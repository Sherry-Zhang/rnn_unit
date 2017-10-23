 /**
  * @file      test_sru.c
  * @author    zhangshu(shu.zhang@intel.com)
  * @date      2017-10-23 14:37:08
  * @brief
  **/
#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <time.h>
#include <omp.h>

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

int main(int argc, char *argv[])
{
    int time_step = 3;
    int batch_size = 32;
    int input_dim = 128;     

    float *c_0 = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64); 
    float *input = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);      
    float *h_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64); 
    
    float *x_wave_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);
    float *f_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64); 
    float *r_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);
    float *c_t = (float*)mkl_calloc(time_step * batch_size * input_dim, sizeof(float), 64);    

    float *b_f = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64); 
    float *b_r = (float*)mkl_calloc(batch_size * input_dim, sizeof(float), 64);
 
    float *w_x = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64); 
    float *w_f = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64); 
    float *w_r = (float*)mkl_calloc(input_dim * input_dim, sizeof(float), 64);

    random_fill(input, time_step * batch_size * input_dim);
    random_fill(c_0, batch_size * input_dim);
    //random_fill(b_f, batch_size * input_dim);
    //random_fill(b_r, batch_size * input_dim);

    random_fill(w_x, input_dim * input_dim);
    random_fill(w_f, input_dim * input_dim);
    random_fill(w_r, input_dim * input_dim);

    float **A = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
    float **B = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
    float **C = (float**)mkl_calloc(3*time_step, sizeof(float*), 64);
   
    SRU_batch_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input, A, B, C);
    printf("batch_gemm called.\n");
    //print(h_t, time_step, input_dim, batch_size);
    // test performance
    int i = 0, count = 10000;
    double begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        SRU_batch_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input, A, B, C);
    }
    double end = dsecnd();
    printf("time:%lfms\n", (end-begin)*1000.0/count);

    memset(x_wave_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(f_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(r_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(c_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(h_t, 0, time_step * batch_size * input_dim * sizeof(float));
    SRU_sequential_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input);
    printf("sequential_gemm called.\n");
    //print(h_t, time_step, input_dim, batch_size);

    begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        SRU_sequential_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input);
    }
    end = dsecnd();
    printf("time:%lfms\n", (end-begin)*1000.0/count);
    memset(x_wave_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(f_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(r_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(c_t, 0, time_step * batch_size * input_dim * sizeof(float));
    memset(h_t, 0, time_step * batch_size * input_dim * sizeof(float));

    SRU_pack_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input);
    printf("pack_gemm called.\n");
    //print(h_t, time_step, input_dim, batch_size);

    begin = dsecnd();
    for(i = 0; i < count; ++i)
    {
        SRU_pack_gemm(batch_size, time_step, input_dim, w_x, w_f, w_r, b_f, b_r, x_wave_t, f_t, r_t, c_t, h_t, c_0, input);
    }
    end = dsecnd();
    printf("time:%lfms\n", (end-begin)*1000.0/count);
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);

    mkl_free(input);
    mkl_free(c_0);
    mkl_free(x_wave_t);
    mkl_free(f_t);
    mkl_free(r_t);
    mkl_free(c_t);
    mkl_free(h_t);

    mkl_free(b_f);
    mkl_free(b_r);
    mkl_free(w_x);
    mkl_free(w_f);
    mkl_free(w_r);
    return 0;
}
