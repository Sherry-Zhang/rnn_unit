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
  * @references     [Training RNNs as Fast as CNNs](https://arxiv.org/pdf/1709.02755.pdf)
  **/

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <time.h>
#include <omp.h>
#include <stdbool.h>

    //printf("[%s:%d]\n", __FILE__, __LINE__);

void SRU_batch_gemm(int batch_size, int time_step, int input_dim, 
                    float* w_x, float* w_f, float* w_r, float* b_f, float* b_r,
                    float* x_wave_t, float* f_t, float* r_t, float* c_t, float* h_t,
                    float* c_0, float* x_t, /*outfloat* y, bool return_sequences,*/
                    const float** A, const float** B, float** C)
{/*{{{*/
    //printf("C   SRU_batch_gemm called.\n");
    MKL_INT grp_size = 3 * time_step;

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = input_dim;
    MKL_INT k = input_dim;
    MKL_INT n = batch_size;
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;

    float alpha = 1.0f;
    float beta = 1.0f;
    
    // x_wave_t = w_x * x_t
    // f_t = sigmoid(w_f * x_t + b_f)
    // r_t = sigmoid(w_r * x_t + b_r)
    int i = 0;
    int j = 0;
    int p = 0;
    int cal_size = batch_size * input_dim;
    # pragma omp parallel for
    for(i = 0; i < time_step; ++i)
    {
        j = i * 3;
        p = i * cal_size;
        A[j] = w_x;
        A[j + 1] = w_f;
        A[j + 2] = w_r;
        B[j] = B[j + 1] = B[j + 2] = x_t + p;
        memcpy(f_t + p, b_f, cal_size * sizeof(float));
        memcpy(r_t + p, b_r, cal_size * sizeof(float));
        C[j] = x_wave_t + p;
        C[j + 1] = f_t + p; 
        C[j + 2] = r_t + p;
    }
    cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, &grp_size);
    # pragma omp parallel for
    for (i = 0; i < time_step * cal_size; ++i)
    {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }
    //c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
    # pragma omp parallel for
    for (i = 0; i < cal_size; ++i)
    {
        c_t[i] = c_0[i] * f_t[i] + (1 - f_t[i]) * x_wave_t[i];
    }
    for (i = 1; i < time_step; ++i)
    {
        p = i * cal_size;
        # pragma omp parallel for
        for(j = 0; j < cal_size; ++j)
        {
            c_t[p + j] = c_t[p - cal_size + j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
    }
    //h_t = r_t · g(c_t) + (1 - r_t) · x_t 
    # pragma omp parallel for
    for (i = 0; i < cal_size * time_step; ++i)
    {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_t[i];   //choose tanh as activation function
    }
//    if (return_sequences)
//    {
//        y = h_t[cal_size * (time_step - 1)];
//    }
//    else
//    {
//        y = h_t;
//    }
}/*}}}*/
void SRU_sequential_gemm(int batch_size, int time_step, int input_dim,
                         float* w_x, float* w_f, float* w_r, float* b_f, float* b_r,
                         float* x_wave_t, float* f_t, float* r_t, float* c_t, float* h_t,
                         float* c_0, float* x_t)
{/*{{{*/
    //printf("C   SRU_sequential_gemm called.\n");
    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = input_dim;
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

    int cal_size = batch_size * input_dim;

    # pragma omp parallel for
    for(i = 0; i < time_step; ++i)
    {
        j = i * cal_size;
        memcpy(f_t + j, b_f, cal_size * sizeof(float));
        memcpy(r_t + j, b_r, cal_size * sizeof(float));
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_x, lda, x_t + j, ldb, beta, x_wave_t + j, ldc);
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_f, lda, x_t + j, ldb, beta, f_t + j, ldc);
        cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_r, lda, x_t + j, ldb, beta, r_t + j, ldc);
    }
    
    # pragma omp parallel for
    for (i = 0; i < time_step * cal_size; ++i)
    {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }

    # pragma omp parallel for
    for (i = 0; i < cal_size; ++i)
    {
        c_t[i] = c_0[i] * f_t[i] + (1 - f_t[i]) * x_wave_t[i];
    }
    for (i = 1; i < time_step; ++i)
    {
        p = i * cal_size;
        # pragma omp parallel for
        for(j = 0; j < cal_size; ++j)
        {
            c_t[p + j] = c_t[p - cal_size + j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
    }
    # pragma omp parallel for
    for (i = 0; i < cal_size * time_step; ++i)
    {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_t[i];   //choose tanh as activation function
    }
}/*}}}*/
void SRU_pack_gemm(int batch_size, int time_step, int input_dim,
                   float* w_x, float* w_f, float* w_r, float* b_f, float* b_r,
                   float* x_wave_t, float* f_t, float* r_t, float* c_t, float* h_t, 
                   float* c_0, float* x_t)
{/*{{{*/
//    printf("C   SRU_pack_gemm called.\n");

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = input_dim;
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

    int cal_size = batch_size * input_dim;
    float* w_x_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    float* w_f_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    float* w_r_pack = cblas_sgemm_alloc(CblasAMatrix, m, n, k);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_x, lda, w_x_pack);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_f, lda, w_f_pack);
    cblas_sgemm_pack(CblasRowMajor, CblasAMatrix, CblasNoTrans, m, n, k, alpha, w_r, lda, w_r_pack);
    if (w_r_pack == NULL || w_f_pack == NULL || w_r_pack == NULL)
    {
        printf("Can't alloc memory for w_pack\n");
        return;
    }
    # pragma omp parallel for
    for(i = 0; i < time_step; ++i)
    {
        j = i * cal_size;
        memcpy(f_t + j, b_f, cal_size * sizeof(float));
        memcpy(r_t + j, b_r, cal_size * sizeof(float));
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_x_pack, lda, x_t + j, ldb, beta, x_wave_t + j, ldc);
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_f_pack, lda, x_t + j, ldb, beta, f_t + j, ldc);
        cblas_sgemm_compute(CblasRowMajor, CblasPacked, CblasNoTrans, m, n, k, w_r_pack, lda, x_t + j, ldb, beta, r_t + j, ldc);
    }

    # pragma omp parallel for
    for (i = 0; i < time_step * cal_size; ++i)
    {
        f_t[i] = 1.0f / (1 + (float)exp(0 - f_t[i]));
        r_t[i] = 1.0f / (1 + (float)exp(0 - r_t[i]));
    }

    # pragma omp parallel for
    for (i = 0; i < cal_size; ++i)
    {
        c_t[i] = c_0[i] * f_t[i] + (1 - f_t[i]) * x_wave_t[i];
    }
    for (i = 1; i < time_step; ++i)
    {
        p = i * cal_size;
        # pragma omp parallel for
        for(j = 0; j < cal_size; ++j)
        {
            c_t[p + j] = c_t[p - cal_size + j] * f_t[p + j] + (1 - f_t[p + j]) * x_wave_t[p + j];
        }
    }
    # pragma omp parallel for
    for (i = 0; i < cal_size * time_step; ++i)
    {
        h_t[i] = r_t[i] * tanh(c_t[i]) + (1 - r_t[i]) * x_t[i];   //choose tanh as activation function
    }
    cblas_sgemm_free(w_x_pack);
    cblas_sgemm_free(w_f_pack);
    cblas_sgemm_free(w_r_pack);
}/*}}}*/

