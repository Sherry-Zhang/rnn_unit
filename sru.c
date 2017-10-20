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

float **A = NULL;
float **B = NULL;
float **C = NULL;

float *x_wave_t = NULL;
float *f_t = NULL;
float *r_t = NULL;
float *c_t = NULL;
          
float *b_f = NULL;
float *b_r = NULL;

float *w_x = NULL;
float *w_f = NULL;
float *w_r = NULL;

void print(float *array, int time_step, int row, int col)
{
    int i, j, k;
    for(i = 0; i < time_step; ++i)
    {
        printf("timestep: %d\n", i);
        for(j = 0; j < row; ++j)
        {
            for(k = 0; k < col; ++k)
            {
                printf("%f ", array[i * row * col + j * col + k]);
            }
            printf("\n");
        }
        printf("\n");
    }

}
void random_fill(float *parray, int len)
{
    int i;
    for(i = 0; i < len; ++i)
    {   
        parray[i] = (float)rand() / (float)RAND_MAX;
    }   
}

void sigmoid(float *parray, int len)
{
    # pragma omp parallel for
    int i;
    for (i = 0; i < len; ++i)
    {
        parray[i] = 1.0 / (1 + (float)exp(0 - parray[i]));
    }
}

void sru_forward(int batch_size, int time_step, int input_dim, float *c_0, float *x_t, float *h_t)
{
    
    MKL_INT grp_size = 3 * time_step;

    CBLAS_TRANSPOSE transA = CblasNoTrans;
    CBLAS_TRANSPOSE transB = CblasNoTrans;

    MKL_INT m = input_dim;
    MKL_INT k = input_dim;
    MKL_INT n = batch_size;
    MKL_INT lda = k;
    MKL_INT ldb = n;
    MKL_INT ldc = n;

    float alpha = 1.0;
    float beta = 1.0;
    
    // x_wave_t = w_x * x_t
    // f_t = sigmoid(w_f * x_t + b_f)
    // r_t = sigmoid(w_r * x_t + b_r)
    int i, j;
    int cal_size = batch_size * input_dim;
    # pragma omp parallel for
    for(i = 0; i < time_step; ++i)
    {
        j = i * 3;
        A[j] = w_x;
        A[j + 1] = w_f;
        A[j + 2] = w_r;

        B[j] = B[j + 1] = B[j + 2] = x_t + i * cal_size;

        C[j] = x_wave_t + i * cal_size;
        cblas_saxpy(cal_size, 1.0, b_f, 1, f_t + i * cal_size, 1);
        C[j + 1] = f_t + i * cal_size; 
        cblas_saxpy(cal_size, 1.0, b_r, 1, r_t + i * cal_size, 1);
        C[j + 2] = r_t + i * cal_size;
    }
    cblas_sgemm_batch(CblasRowMajor, &transA, &transB, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, &grp_size);
    //cblas_sgemm(CblasRowMajor, transA, transB, m, n, k, alpha, w_x, lda, x_t, ldb, beta, x_wave_t, ldc);
    sigmoid(f_t, time_step * cal_size);
    sigmoid(r_t, time_step * cal_size);
    
    //c_t = f_t · c_tm1 + (1 - f_t) · x_wave_t
    # pragma omp parallel for
    for (i = 0; i < cal_size; ++i)
    {
        c_t[i] = c_0[i] * f_t[i] + (1 - f_t[i]) * x_wave_t[i];
    }
    int p;
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
    print(h_t, time_step, input_dim, batch_size);
}
int main(int argc, char *argv[])
{
    int time_step = 3;
    int batch_size = 2;
    int input_dim = 5;     

    float *c_0 = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64); 
    float *input = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);      
    float *output = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64); 
    
    x_wave_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);
    f_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64); 
    r_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);
    c_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);    

    b_f = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64); 
    b_r = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64);
 
    w_x = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64); 
    w_f = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64); 
    w_r = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64);

    random_fill(input, time_step * batch_size * input_dim);
    random_fill(c_0, batch_size * input_dim);
    //random_fill(b_f, batch_size * input_dim);
    //random_fill(b_r, batch_size * input_dim);

    random_fill(w_x, input_dim * input_dim);
    random_fill(w_f, input_dim * input_dim);
    random_fill(w_r, input_dim * input_dim);

    //A、B、C为参与cblas运算的临时指针，AB = C
    A = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
    B = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
    C = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
   
    sru_forward(batch_size, time_step, input_dim, c_0, input, output); //output is out parameter
   
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    mkl_free(c_0);
    mkl_free(x_wave_t);
    mkl_free(f_t);
    mkl_free(r_t);
    mkl_free(c_t);

    mkl_free(b_f);
    mkl_free(b_r);
    mkl_free(w_x);
    mkl_free(w_f);
    mkl_free(w_r);
    mkl_free(input);
    mkl_free(output);
    return 0;
}
