
template <int ldA, int ldB, int K>
__inline__ __device__ __host__ void mma(float *a, float *b, float *c)
{

#pragma unroll
        for (int i = 0; i < ldA; i++)
#pragma unroll
        for (int j = 0; j < ldB; j++)
#pragma unroll
        for (int k = 0; k < K; k++)
                c[j + ldB * i] += a[i + ldA * k] * b[j + ldB * k];

}

__global__ void dgemm_16x16(float *C, float *A, float *B, int M, int N, int K)
{

        const int warpSize = 32;
        const int numBuffers = 2;

        const int threadDim_x = 4;
        const int threadDim_y = 8;

        const int warpDim_x = 4;
        const int warpDim_y = 2;
        const int numFrags = 16;

        const int outputDim_x = 8;
        const int outputDim_y = 8;

        int tx = threadIdx.x % 32;
        int ty = threadIdx.x / 32;


        const int sDim = 32;
        extern __shared__ float smem[];


        int threadIdx_x = tx % threadDim_x;
        int threadIdx_y = tx / threadDim_x;

        int warpIdx_x = ty % warpDim_x;
        int warpIdx_y = ty / warpDim_x;
        
        float4 *sA = (float4*)smem;
        float4 *sB = &((float4*)smem)[2 * sDim * numFrags * numBuffers];
        // the C output matrix will overwrite any loaded data into shared memory
        float4 *sC = sA;

        int idxA = tx + blockIdx.y * warpSize;
        int idxB = tx + blockIdx.x * warpSize;

        float4 *load_ptr, *store_ptr;

        int loadIncrement;


        // Registers for holding thin outer product slices
        const int aDim = outputDim_y / 4;
        const int bDim = outputDim_x / 4;
        const int cDim_x = outputDim_x / 4;
        float4 a[aDim * numFrags];
        float4 b[bDim * numFrags];
        float c[outputDim_x * outputDim_y];

        const int idx = threadIdx_x + warpIdx_x * threadDim_x;
        const int idy = threadIdx_y + warpIdx_y * threadDim_y;

                
        if (warpIdx_y == 0) {
                store_ptr = &sA[tx + sDim * warpIdx_x];
                load_ptr = &((float4*)A)[idxA + (M / 4) * warpIdx_x];              
                loadIncrement = (M / 4) * warpDim_x;
        }
        if (warpIdx_y == 1) {
                store_ptr = &sB[tx + sDim * warpIdx_x];
                load_ptr = &((float4*)B)[idxB + (N / 4) * warpIdx_x];              
                loadIncrement = (N / 4) * warpDim_x;
        }

        float4 *currStore_ptr = store_ptr;
        float4 *nextStore_ptr = store_ptr + sDim * numFrags; 


        for (int load = 0; load < numFrags; load += warpDim_x) {
                *store_ptr = *load_ptr;
                load_ptr += loadIncrement;
        }

        for (int i = 0; i < outputDim_y; ++i)
        for (int j = 0; j < outputDim_x; ++j)
                c[j + i * outputDim_x] = 0.0f;


        int iter = 0;
        for (int tilePos = numFrags; tilePos <= K; tilePos +=
                        numFrags) {
                
                store_ptr = iter % 2 == 0 ? nextStore_ptr : currStore_ptr;
                int bufferIncrement = iter % 2 == 1 ? sDim * numFrags : 0;

                __syncthreads();

                for (int k = 0; k < numFrags; ++k) {
                for (int i = 0; i < aDim; ++i)
                        a[i + aDim * k] = sA[idy * aDim + i + bufferIncrement +
                        sDim * k];

                for (int i = 0; i < bDim; ++i)
                        b[i + bDim *k] = sB[idx * bDim + i + bufferIncrement +
                        sDim * k];

                }
                mma<outputDim_x, outputDim_y, numFrags>((float*)a, (float*)b, c);

                
                if (tilePos < K) {
                        for (int load = 0; load < numFrags; load += warpDim_x) {
                                *store_ptr = *load_ptr;
                                load_ptr += loadIncrement;
                        }
                }
                

                iter++;
        }

        
        // Write output to shared memory to coalesce writes
        store_ptr = &sC[outputDim_x / 4 * idx + sDim * idy];
        load_ptr = &sC[tx + warpDim_y * sDim * ty];
        nextStore_ptr = store_ptr + sDim * warpDim_y * threadDim_y;
        currStore_ptr = store_ptr;

        float4 *currLoad_ptr = load_ptr;
        float4 *nextLoad_ptr = load_ptr + sDim * warpDim_y * threadDim_y;
        iter = 0;

        for (int j = 0; j < cDim_x; ++j)
                *(store_ptr + j) = ((float4*)c)[j];

        for (int i = 0; i < outputDim_y; ++i) {

                store_ptr = iter % 2 == 0 ? nextStore_ptr : currStore_ptr;
                load_ptr = iter % 2 == 0 ? currLoad_ptr : nextLoad_ptr;
                
                __syncthreads();

                // Write to global memory
                const int gidx_x = tx + blockIdx.x * warpSize;
                const int gidx_y = warpDim_y * outputDim_y * ty + i +
                                   blockIdx.y * warpSize * 4;

                for (int k = 0; k < warpDim_y; ++k) {
                        if (gidx_x < N && gidx_y < M)
                                ((float4 *)
                                     C)[gidx_x +
                                        N / 4 * (gidx_y + outputDim_y * k)] =
                                    *(load_ptr + k * sDim);
                }
        

                for (int j = 0; j < cDim_x; ++j)
                        *(store_ptr + j) = ((float4*)c)[j + (i + 1) * cDim_x];


                iter++;
        }



}

void gemm_16x16(float *C, float *AT, float *B, int M, int N, int K)
{
                dim3 threads(32 * 8, 1, 1);
                dim3 blocks((N - 1) / 128 + 1,
                            (M - 1) / 128 + 1, 1);

                assert( M % 4 == 0);
                assert( N % 4 == 0);

                int maxbytes = 65536;  // 64 KB
                cudaFuncSetAttribute(
                    dgemm_16x16,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    dgemm_16x16,
                    cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
                dgemm_16x16<<<blocks, threads, maxbytes>>>(C, AT, B, M, N, K);
}
