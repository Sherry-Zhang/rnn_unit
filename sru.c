 /**
  * @file           sru.c
  * @author         zhangshu(shu.zhang@intel.com)
  * @date           2017-10-19 15:50:03
  * 
  * @brief          SRU for inference
  *
  * @formula list:  x_wave_t = w * x_t
  *                 f_t = sigmoid(w_f * x_t + b_f)
  *                 r_t = sigmoid(w_r * x_t + b_r)
  *                 c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
  *                 h_t = r_t · g(c_t) + (1 - r_t) · x_t 
  *
  **/

//printf("[%s:%d]\n", __FILE__, __LINE__);
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mkl.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

typedef struct {
    float **A;
    float **B;
    float **C;
    float* x_wave_t;
    float* f_t;
    float* r_t;
    float* c_t;
    float* h_t;
    float* x_tmp_t;
}sru_handle;

unsigned long sru_get_size(int batch_size, int hidden_dim, int time_step) {
    return time_step * 12 * sizeof(float*) + time_step * batch_size * hidden_dim * 6 * sizeof(float);
}


void sru_create_instance(void* buf, sru_handle* handle, int hidden_dim, int batch_size, int time_step) {
    handle->A = (float**)buf;
    handle->B = (float**)(buf + 4 * time_step * sizeof(float*));
    handle->C = (float**)(buf + 8 * time_step * sizeof(float*));
    handle->x_wave_t = (float*)(buf + 12 * time_step * sizeof(float*));
    handle->f_t = (float*)((void*)handle->x_wave_t + time_step * batch_size * hidden_dim * sizeof(float));
    handle->r_t = (float*)((void*)handle->f_t + time_step * batch_size * hidden_dim * sizeof(float));
    handle->c_t = (float*)((void*)handle->r_t + time_step * batch_size * hidden_dim * sizeof(float));
    handle->h_t = (float*)((void*)handle->c_t + time_step * batch_size * hidden_dim * sizeof(float));
    handle->x_tmp_t = (float*)((void*)handle->h_t + time_step * batch_size * hidden_dim * sizeof(float));
}

void sru_batch_gemm(int batch_size,     //N
                    int time_step,      //T
                    int input_dim,      //I
                    int hidden_dim,     //H
                    float* w_x,         //H*I
                    float* w_f,         //H*I
                    float* w_r,         //H*I
                    float* w_tmp,       //H*I, if hidden_dim != input_dim, add one more linear transform: w_tmp * x_t
                    float* b_f,         //H*N
                    float* b_r,         //H*N
                    float* c_0,         //H*N
                    float* x_t,         //T*I*N
                    float* x_wave_t,    //T*H*N  
                    float* x_tmp_t,     //T*H*N, if hidden_dim != input_dim, x_tmp_t = w_tmp * x_t, else x_tmp_t = x_t
                    float* f_t,         //T*H*N 
                    float* r_t,         //T*H*N 
                    float* c_t,         //T*H*N 
                    float* h_t,         //T*H*N 
                    float** A,          //temp buffer for gemm
                    float** B,
                    float** C) {

    //printf("C   SRU_batch_gemm called.\n");
    MKL_INT grp_size = 3 * time_step;

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = hidden_dim;
    MKL_INT k = input_dim;
    MKL_INT n = batch_size;
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;

    float alpha = 1.0f;
    float beta = 1.0f;
    
    int i = 0;
    int j = 0;
    int p = 0;

    int input_size = input_dim * batch_size;
    int hidden_size = hidden_dim * batch_size;
    int all_size = time_step * hidden_size;
    // x_wave_t = w_x * x_t
    // f_t = sigmoid(w_f * x_t + b_f)
    // r_t = sigmoid(w_r * x_t + b_r)
    # pragma omp parallel for
    for(i = 0; i < time_step; ++i) {
        j = i * 3;
        p = i * hidden_size;
        A[j] = w_x;
        A[j + 1] = w_f;
        A[j + 2] = w_r;
        B[j] = B[j + 1] = B[j + 2] = x_t + i * input_size;
        memcpy(f_t + p, b_f, hidden_size * sizeof(float));
        memcpy(r_t + p, b_r, hidden_size * sizeof(float));
        C[j] = x_wave_t + p;
        C[j + 1] = f_t + p; 
        C[j + 2] = r_t + p;
    }
    if (input_dim != hidden_dim)     //add one more linear transform
    {
        # pragma omp parallel for
        for(i = 0; i < time_step; ++i) {
            A[grp_size + i] = w_tmp;
            B[grp_size + i] = x_t + i * input_size;
            C[grp_size + i] = x_tmp_t + i * hidden_size;
        }
        grp_size = 4 * time_step;
    }
    else
    {
        memcpy(x_tmp_t, x_t, all_size * sizeof(float));
    }
    cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, &grp_size);
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }
    //c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
    for (i = 0; i < time_step; ++i) {
        p = i * hidden_size;
        # pragma omp parallel for
        for(j = 0; j < hidden_size; ++j) {
            c_t[p + j] = c_0[j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
        c_0 = c_t + p;
    }
    //h_t = r_t · g(c_t) + (1 - r_t) · x_t 
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_tmp_t[i];   //choose tanh as activation function
    }
}
void sru_sequential_gemm(int batch_size,     //N
                         int time_step,      //T
                         int input_dim,      //I
                         int hidden_dim,     //H
                         float* w_x,         //H*I
                         float* w_f,         //H*I
                         float* w_r,         //H*I
                         float* w_tmp,       //H*I, if hidden_dim != input_dim, add one more linear transform: w_tmp * x_t
                         float* b_f,         //H*N
                         float* b_r,         //H*N
                         float* c_0,         //H*N
                         float* x_t,         //T*I*N
                         float* x_wave_t,    //T*H*N  
                         float* x_tmp_t,     //T*H*N, if hidden_dim != input_dim, x_tmp_t = w_tmp * x_t, else x_tmp_t = x_t
                         float* f_t,         //T*H*N 
                         float* r_t,         //T*H*N 
                         float* c_t,         //T*H*N 
                         float* h_t          //T*H*N 
                         ) {

    //printf("C   SRU_sequential_gemm called.\n");

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = hidden_dim;
    MKL_INT k = input_dim;
    MKL_INT n = batch_size;
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;

    float alpha = 1.0f;
    float beta = 1.0f;

    int i = 0;
    int j = 0;
    int p = 0;

    int input_size = input_dim * batch_size;
    int hidden_size = hidden_dim * batch_size;
    int all_size = time_step * hidden_size;

    # pragma omp parallel for
    for(i = 0; i < time_step; ++i) {
        j = i * hidden_size;
        memcpy(f_t + j, b_f, hidden_size * sizeof(float));
        memcpy(r_t + j, b_r, hidden_size * sizeof(float));
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_x, lda, x_t + j, ldb, beta, x_wave_t + j, ldc);
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_f, lda, x_t + j, ldb, beta, f_t + j, ldc);
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_r, lda, x_t + j, ldb, beta, r_t + j, ldc);
        if (input_dim != hidden_dim) {
            cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_tmp, lda, x_t + j, ldb, beta, x_tmp_t + j, ldc);
        }
    }
    if (input_dim == hidden_dim) {
        memcpy(x_tmp_t, x_t, all_size * sizeof(float));
    }
    
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }

    //c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
    for (i = 0; i < time_step; ++i) {
        p = i * hidden_size;
        # pragma omp parallel for
        for(j = 0; j < hidden_size; ++j) {
            c_t[p + j] = c_0[j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
        c_0 = c_t + p;
    }
    //h_t = r_t · g(c_t) + (1 - r_t) · x_t 
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_tmp_t[i];   //choose tanh as activation function
    }
}
void sru_pack_gemm(int batch_size,     //N
                   int time_step,      //T
                   int input_dim,      //I
                   int hidden_dim,     //H
                   float* w_x,         //H*I
                   float* w_f,         //H*I
                   float* w_r,         //H*I
                   float* w_tmp,       //H*I, if hidden_dim != input_dim, add one more linear transform: w_tmp * x_t
                   float* b_f,         //H*N
                   float* b_r,         //H*N
                   float* c_0,         //H*N
                   float* x_t,         //T*I*N
                   float* x_wave_t,    //T*H*N  
                   float* x_tmp_t,     //T*H*N, if hidden_dim != input_dim, x_tmp_t = w_tmp * x_t, else x_tmp_t = x_t
                   float* f_t,         //T*H*N 
                   float* r_t,         //T*H*N 
                   float* c_t,         //T*H*N 
                   float* h_t          //T*H*N 
                   ) {
//    printf("C   SRU_pack_gemm called.\n");

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = hidden_dim;
    MKL_INT k = input_dim;
    MKL_INT n = batch_size;
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;

    float alpha = 1.0f;
    float beta = 1.0f;
    int i = 0;
    int j = 0;
    int p = 0;

    int input_size = input_dim * batch_size;
    int hidden_size = hidden_dim * batch_size;
    int all_size = time_step * hidden_size;

    float* w_x_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    float* w_f_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    float* w_r_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    float* w_tmp_pack = NULL;
    if (w_x_pack == NULL || w_f_pack == NULL || w_r_pack == NULL) {
        printf("[%s:%d]Can't alloc memory for w_pack\n", __FILE__, __LINE__);
        return;
    }
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_x, lda, w_x_pack);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_f, lda, w_f_pack);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_r, lda, w_r_pack);
    if (input_dim != hidden_dim) {
        w_tmp_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
        if (w_x_pack == NULL) {
            printf("Can't alloc memory for w_pack\n");
            return;
        }
        cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_tmp, lda, w_tmp_pack);
    }
    # pragma omp parallel for
    for(i = 0; i < time_step; ++i) {
        j = i * hidden_size;
        memcpy(f_t + j, b_f, hidden_size * sizeof(float));
        memcpy(r_t + j, b_r, hidden_size * sizeof(float));
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_x_pack, lda, x_t + j, ldb, beta, x_wave_t + j, ldc);
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_f_pack, lda, x_t + j, ldb, beta, f_t + j, ldc);
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_r_pack, lda, x_t + j, ldb, beta, r_t + j, ldc);
        if (input_dim != hidden_dim) {
            cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_tmp_pack, lda, x_t + j, ldb, beta, x_tmp_t + j, ldc);
        }
    }
    if (input_dim == hidden_dim) {
        memcpy(x_tmp_t, x_t, all_size * sizeof(float));
    }
    
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }

    //c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
    for (i = 0; i < time_step; ++i) {
        p = i * hidden_size;
        # pragma omp parallel for
        for(j = 0; j < hidden_size; ++j) {
            c_t[p + j] = c_0[j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
        c_0 = c_t + p;
    }
    //h_t = r_t · g(c_t) + (1 - r_t) · x_t 
    # pragma omp parallel for
    for (i = 0; i < all_size; ++i) {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_tmp_t[i];   //choose tanh as activation function
    }
    cblas_sgemm_free(w_x_pack);
    cblas_sgemm_free(w_f_pack);
    cblas_sgemm_free(w_r_pack);
}


int sru_inference(void* buf,
                  int batch_size,     //N
                  int time_step,      //T
                  int input_dim,      //I
                  int hidden_dim,     //H
                  float* w_x,         //H*I
                  float* w_f,         //H*I
                  float* w_r,         //H*I
                  float* w_tmp,       //H*I, if hidden_dim not equal to input_dim, add one more linear transform: w_tmp * x_t
                  float* b_f,         //H*N
                  float* b_r,         //H*N
                  float* c_0,         //H*N
                  float* x_t,         //T*I*N
                  float* h_out,       //if return_sequences == true, size = T*H*N, else size = H*N
                  bool return_sequences,
                  int mode) {
    memset(buf, 0, sru_get_size(batch_size, hidden_dim, time_step));
    sru_handle* han = (sru_handle*)mkl_calloc(1 ,sizeof(sru_handle), 64);
    if (NULL == han)
    {
        printf("[%s:%d] Can't alloc memory\n", __FILE__, __LINE__);
        return -1;
    }
    sru_create_instance(buf, han, hidden_dim, batch_size, time_step);
    float** A = han->A;
    float** B = han->B;
    float** C = han->C;
    float* f_t = han->f_t;
    float* r_t = han->r_t;
    float* c_t = han->c_t;
    float* h_t = han->h_t;
    float* x_wave_t = han->x_wave_t;
    float* x_tmp_t = han->x_tmp_t;
    switch(mode) {
        case 0:
            sru_batch_gemm(batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                           c_0, x_t, x_wave_t, x_tmp_t, f_t, r_t, c_t, h_t, A, B, C);
            break;
        case 1:
            sru_sequential_gemm(batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                           c_0, x_t, x_wave_t, x_tmp_t, f_t, r_t, c_t, h_t);
            break;
        case 2:
            sru_pack_gemm(batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                           c_0, x_t, x_wave_t, x_tmp_t, f_t, r_t, c_t, h_t);
            break;
        default:
            sru_batch_gemm(batch_size, time_step, input_dim, hidden_dim, w_x, w_f, w_r, w_tmp, b_f, b_r, 
                           c_0, x_t, x_wave_t, x_tmp_t, f_t, r_t, c_t, h_t, A, B, C);
            
    }
    int hidden_size = hidden_dim * batch_size;
    int all_size = time_step * hidden_size;
    if (return_sequences) {
        memcpy(h_out, h_t, sizeof(float) * all_size);
    } 
    else {
        memcpy(h_out, h_t + all_size - hidden_size, sizeof(float) * hidden_size);
    }
    mkl_free(han);
    return 0;
}
