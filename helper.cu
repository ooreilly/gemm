#include <stdlib.h>
#include <cublas_v2.h>
#include <cusparse.h>

#define cublasErrCheck(stat) { cublasErrCheck_((stat), __FILE__, __LINE__); }
void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
   if (stat != CUBLAS_STATUS_SUCCESS) {
      fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
   }
}

#define cusparseErrCheck(stat) { cusparseErrCheck_((stat), __FILE__, __LINE__); }
void cusparseErrCheck_(cusparseStatus_t stat, const char *file, int line) {
   if (stat != CUSPARSE_STATUS_SUCCESS) {
      fprintf(stderr, "cuSPARSE Error: %d %s %d\n", stat, file, line);
   }
}

#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if( stat == cudaErrorCudartUnloading ) {
          // Ignore this error
  }
  else if( stat != cudaSuccess) {                                                   
        fprintf(stderr, "CUDA error in %s:%i %s.\n",                          
          file, line, cudaGetErrorString(stat) );            
        fflush(stderr);                                                             
  }
}

#define dump_rows(A, m, n) print_row_matrix(#A, A, m, n)
template <typename TV>
void print_row_matrix(const char *A, TV *a, int m, int n)
{
        printf("%s = [\n", A);
        for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
                printf("%2.2f ", (double)a[j + i * n]);
        }
        printf("\n");
        }
        printf("] \n");

}

#define dump_rows32(A, m, n) print_row_matrix32(#A, A, m, n)
template <typename TV>
void print_row_matrix32(const char *A, TV *a, int m, int n)
{
        printf("%s = [\n", A);
        for (int i = 0; i < m; ++i) {
        for (int j = 0; j < 3; ++j) {
                printf("%2.2f ", (double)a[j + i * n]);

        }
        printf(" ... ");
        for (int j = 29; j < 32; ++j) {
                printf("%2.2f ", (double)a[j + i * n]);
        }
        printf("\n");
        }
        printf("] \n");

}

#define dump_cols(A, m, n) print_col_matrix(#A, A, m, n)
template <typename TV>
void print_col_matrix(const char *A, TV *a, int m, int n)
{
        printf("%s = [\n", A);
        for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
                printf("%f ", a[i + j * m]);
        }
        printf("\n");
        }
        printf("\n");

}


template<typename Tv>
void fill(Tv *A, int n) {
        for (int i = 0; i < n; ++i) {
                A[i] = rand() *  1.0 / INT_MAX;
        }
}

template<typename Tv>
void ones(Tv *A, int n) {
        for (int i = 0; i < n; ++i) {
                A[i] = 1.0;
        }
}

template<typename Tv>
void fill_idx(Tv *A, int n) {
        for (int i = 0; i < n; ++i) {
                A[i] = 1.0 * (i % 100000);
        }
}

template<typename Tv>
void identity(Tv *A, int m, int n)
{
        for (int i = 0; i < m; ++i) { 
                for (int j = 0; j < n; ++j) 
                        A[j + i * n] = 0;
                A[i + i * n] = 1;
        }
}

template<typename Tv>
void zeros(Tv *A, int n) {
        for (int i = 0; i < n; ++i) {
                A[i] = 0.0;
        }
}

template<typename Tv>
__global__ void _dzeros(Tv *A, int n) {
        int tx = threadIdx.x + blockDim.x * blockIdx.x;
        A[tx] = 0.0; 
}


template<typename Tv>
void dzeros(Tv *A, int n) {
        dim3 threads (32 * 32, 1, 1);
        dim3 blocks ((n - 1) / threads.x + 1, 1, 1);
        _dzeros<<<blocks, threads>>>(A, n);
}

template<typename Ta, typename Tb>
__global__ void _dcopy(Ta *A, Tb *B, int n) {
        int tx = threadIdx.x + blockDim.x * blockIdx.x;
        for (int i = tx; i < n; i += blockDim.x * gridDim.x) {
                A[i] = (Ta)B[i]; 
        }
}


template<typename Ta, typename Tb>
void copy(Ta *A, Tb *B, int n) {
        dim3 threads (32 * 32, 1, 1);
        dim3 blocks ((n - 1) / threads.x + 1, 1, 1);
        _dcopy<<<blocks, threads>>>(A, B, n);
}

template <typename Ta, typename Tb>
double compare(Ta *A, Tb *B, int m, int n)
{
        double max_err = 0.0;
        for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
                double diff = fabs((int)A[j + i * n] - (int)B[j + i * n]);
                if (max_err < diff) {
                        printf("error at i, j = %d, %d, diff=%g, A=%g, B=%g,  \n", i, j, diff,
                                        (double)A[j + i * n],
                                        (double)B[j + i * n]);
                        max_err = diff;
                }
        }
        }
        return max_err;
}

template <typename TV>
double relative_diff(TV *A, TV *B, int m, int n)
{
        double max = 0.0;
        for (int i = 0; i < m * n; ++i) {
                double a = (double)A[i];
                double b = (double)B[i];
                double r = a / b;
                if (r > max) max = r;
        }
        return max;
}

template <typename TV>
void diff(TV *A, TV *B, TV *C, int m, int n)
{
        for (int i = 0; i < m * n; ++i) {
                C[i] = (TV)fabs((double)A[i] - (double)B[i]);
        }
}

template <typename Cv, typename Av, typename Ti>
void transpose(Cv *AT, Av *A, Ti m, Ti n)
{

        for (int i = 0; i < m; ++i) {
                for (int j = 0; j < n; ++j) {
                        Cv temp = (Cv)A[j + i * n];
                        AT[i + j * m] = temp;
                }
        }


}

#define time(func, blocks, threads, args, val) \
{ \
                float gpu_time; \
                cudaEvent_t gpu_start, gpu_stop; \
                cudaEventCreate(&gpu_start); \
                cudaEventCreate(&gpu_stop);  \
                cudaEventRecord(gpu_start, 0); \
                func<<<blocks, threads>>>args; \
                cudaEventRecord(gpu_stop, 0); \
                cudaEventSynchronize(gpu_stop); \
                cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);\
                printf("%s took:  %3.1f ms \n", #func, gpu_time); \
                *val = gpu_time; \
}

class Time
{
        cudaEvent_t gpu_start, gpu_stop; 
        public:

        Time() {
        cudaEventCreate(&gpu_start); \
        cudaEventCreate(&gpu_stop);  \
        }

        void start(void) {
                cudaEventRecord(gpu_start, 0); 
        }

        float stop(const char *msg) {
                float gpu_time;
                cudaEventRecord(gpu_stop, 0); 
                cudaEventSynchronize(gpu_stop);
                cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);
                printf("%s took:  %g ms \n", msg, gpu_time); 
                return gpu_time;
        }

};

#define time2(func, blocks, threads, args, val) \
{ \
                float gpu_time; \
                cudaEvent_t gpu_start, gpu_stop; \
                cudaEventCreate(&gpu_start); \
                cudaEventCreate(&gpu_stop);  \
                cudaEventRecord(gpu_start, 0); \
                func<TILE><<<blocks, threads>>>args; \
                cudaEventRecord(gpu_stop, 0); \
                cudaEventSynchronize(gpu_stop); \
                cudaEventElapsedTime(&gpu_time, gpu_start, gpu_stop);\
                printf("%s took:  %3.1f ms \n", #func, gpu_time); \
                *val = gpu_time; \
}

double gflops(const char *label, int64_t size, double ms_elapsed)
{

       double gf = (double) size / (ms_elapsed * 1e6);
       printf("%s: %g GFLOPs \n",  label, gf);
       return gf;
}
