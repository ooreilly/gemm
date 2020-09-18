#include <stdio.h>
#include <assert.h>
#include <cuda_profiler_api.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "helper.cu"
#include "printing.cuh"
#include "baseline.cuh"
#include "optimized.cuh"

typedef float Tv; 
typedef int Ti; 

void print_usage_and_exit(const char *program) {
        printf("usage: %s M N K \n", program);
        printf("Computes the matrix-matrix multiplication A * B = C in single precision, \n" \
        "where A is of size M * K, B is of size K * N, and hence C is of size M * N. \n" \
        "The optimized kernel needs both M and N to be divisible by 4 to run.\n"); 
}

int main(int argc, char **argv)
{

        if (argc != 4) {
                print_usage_and_exit(argv[0]);
                exit(1);
        }
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int K = atoi(argv[3]);

        Tv *A, *B, *C, *C2, *AT, *BT, *CT;
        Tv *d_A, *d_B, *d_C, *d_AT, *d_BT, *d_CT;

        A = (Tv*) malloc(sizeof(A) * M * K);
        B = (Tv*) malloc(sizeof(B) * K * N);
        C = (Tv*) malloc(sizeof(C) * M * N);
        C2 = (Tv*) malloc(sizeof(C2) * M * N);
        AT = (Tv*) malloc(sizeof(A) * M * K);
        BT = (Tv*) malloc(sizeof(B) * K * N);
        CT = (Tv*) malloc(sizeof(C) * M * N);

        cudaErrCheck(cudaMalloc((void**)&d_A, sizeof(A) * M * K));
        cudaErrCheck(cudaMalloc((void**)&d_B, sizeof(B) * K * N));
        cudaErrCheck(cudaMalloc((void**)&d_C, sizeof(C) * M * N));
        cudaErrCheck(cudaMalloc((void**)&d_AT, sizeof(A) * M * K));
        cudaErrCheck(cudaMalloc((void**)&d_BT, sizeof(B) * K * N));
        cudaErrCheck(cudaMalloc((void**)&d_CT, sizeof(C) * M * N));

        identity<Tv>(A, M , K);
        identity<Tv>(B, K , N);
        fill_idx<Tv>(A, M * K);
        fill_idx<Tv>(B, K * N);

        transpose(AT, A, M, K);
        transpose(BT, B, N, K);

        cudaErrCheck(cudaMemcpy(d_A, A, sizeof(A) * M * K, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(d_B, B, sizeof(B) * K * N, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(d_C, C, sizeof(C) * M * N, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(d_AT, AT, sizeof(A) * M * K, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(d_BT, BT, sizeof(B) * K * N, cudaMemcpyHostToDevice));
        cudaErrCheck(cudaMemcpy(d_CT, CT, sizeof(C) * M * N, cudaMemcpyHostToDevice));

        // Baseline
        {
                Time t = Time();
                dim3 threads(32, 32, 1);
                dim3 blocks((N - 1) / threads.x + 1,
                            (M - 1) / threads.y + 1, 1);
                t.start();

                gemm_rrr<Tv, Tv, Ti><<<blocks, threads>>>(d_C, d_A, d_B, M, N, K);
                float elapsed = t.stop("gemm_baseline");
                cudaErrCheck(cudaMemcpy(C, d_C, sizeof(C) * M * N,
                                        cudaMemcpyDeviceToHost));
        }

        // Optimized
        {
                Time t = Time();
                dzeros(d_C, M * N);
                t.start();
                gemm_16x16(d_C, d_AT, d_B, M, N, K);
                float elapsed = t.stop("gemm_16x16");
                cudaErrCheck(cudaMemcpy(C2, d_C, sizeof(C) * M * N,
                                        cudaMemcpyDeviceToHost));
                double rdiff = compare(C, C2, M, N);
                printf("Maximum difference: %g \n", rdiff);
        }

        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_AT);
        cudaFree(d_BT);
        cudaFree(d_CT);

        free(A);
        free(B);
        free(C);
        free(AT);
        free(BT);
        free(CT);

        return 0;
}

