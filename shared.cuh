
// 4 x 4 matrix multiplication and accumulation
__inline__ __device__ __host__ void mma4(float4 *A, float4 *B, float4 *C)
{
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
                C[k].x += A[k].x * B[0].x + A[k].y * B[1].x + A[k].z * B[2].x +
                          A[k].w * B[3].x;
                C[k].y += A[k].x * B[0].y + A[k].y * B[1].y + A[k].z * B[2].y +
                          A[k].w * B[3].y;
                C[k].z += A[k].x * B[0].z + A[k].y * B[1].z + A[k].z * B[2].z +
                          A[k].w * B[3].z;
                C[k].w += A[k].x * B[0].w + A[k].y * B[1].w + A[k].z * B[2].w +
                          A[k].w * B[3].w;
        }
}

__inline__ __device__ __host__ void op8(float *a, float *b, float *c)
{
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
                c[j + 8 * i] += a[i] * b[j];

}

template <int dim>
__inline__ __device__ __host__ void mma(float *a, float *b, float *c)
{

#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
        for (int j = 0; j < 8; j++)
#pragma unroll
        for (int k = 0; k < dim; k++)
                c[j + 8 * i] += a[i + 8 * k] * b[j + 8 * k];

}

template <int size, int dim>
__inline__ __device__ __host__ void mma(float *a, float *b, float *c)
{

#pragma unroll
        for (int i = 0; i < size; i++)
#pragma unroll
        for (int j = 0; j < size; j++)
#pragma unroll
        for (int k = 0; k < dim; k++)
                c[j + size * i] += a[i + size * k] * b[j + size * k];

}

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


template <int tile_size>
//__launch_bounds__(256)
__global__ void dgemm_blocked(float *C, float *A, float *B, int M, int N, int K)
{
        extern __shared__ float sm[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int idx = tx + blockDim.x * blockIdx.x;
        int idy = ty + blockDim.y * blockIdx.y;
        int bidx = blockIdx.x;
        int bidy = blockIdx.y;
        const int warp_y = 4;
        const int ws = 32;
        const int wsh = 16;
        const int pad = 0;
        const int sx = ws + pad;



        // Number of A and B rows/columns that fit into shared memory
        const int numFrags = 8;
        const int fragDim = numFrags * warp_y;
        const int sharedDimA = fragDim * sx;

        float4 *sA = ((float4*)sm);
        float4 *sB = &(((float4*)sm)[sharedDimA]);

        // Pointer into shared memory after A, B fragments. This pointer is used
        // for debugging purposes.
        float *sC = &sm[2 * 4 * fragDim];


        int Kf = K / 4;
        int Mf = M / 4;
        int Nf = N / 4;
        float4 *fA = (float4*)A;
        float4 *fB = (float4*)B;

        // Layout threads 8 x 4 tiles
        const int threadDim_x = 4;
        const int threadDim_y = 8;
        int ctx = tx % threadDim_x;
        int cty = tx / threadDim_x;

        // Layout warps in 2 x 4 tiles
        const int warpDim_x = 4;
        const int warpDim_y = 2;
        int wx = ty % threadDim_x;
        int wy = ty / threadDim_x;

        int cidx = ctx + wx * threadDim_x;
        int cidy = cty + wy * threadDim_y;


        const int fragSize_x = 8;
        const int fragSize_y = 8;



        int lty = ty % 4;

        // Split warps into two groups, group 0 loads A, and group 1 loads B

        float4 *store_frag_ptr = ty < 4 ? &sA[tx + sx * lty] : &sB[tx + sx * lty];
        float4 *load_frag_ptr = ty < 4 ? &fA[tx + blockIdx.y * blockDim.x + Mf * lty] : &fB[idx + Nf * lty];
        int inc = ty < 4 ? Mf : Nf;

        float4 *store_frag_ptr_start = store_frag_ptr;

        float cout[64];
        #pragma unroll
        for (int i = 0; i < 64; ++i)
                cout[i] = 0.0f;


        // Loop over all tiles
        for (int tile = 0; tile < K; tile+= fragDim) {

                for (int frag = 0; frag < numFrags; frag++) {
                        *store_frag_ptr = *load_frag_ptr;
                      store_frag_ptr += sx * warpDim_x;
                      load_frag_ptr += inc * warpDim_x;
                }

                __syncthreads();

                // Load data from shared memory into registers

                // Process a 8 x 8 tile as skinny outer products
                // 8 x 1 * 1 x 8
                const int dim = 4;
                float4 atile[dim][2];
                float4 btile[dim][2];


                #pragma unroll
                for (int frag = 0; frag < fragDim / dim; frag++) {
                        #pragma unroll
                        for (int d = 0; d < dim; d++) {
                        atile[d][0] = sA[2 * cidy  + sx * (dim * frag + d)];
                        atile[d][1] = sA[2 * cidy + 1  + sx * (dim * frag + d)];
                        btile[d][0] = sB[2 * cidx  + sx * (dim * frag + d)];
                        btile[d][1] = sB[2 * cidx + 1  + sx * (dim * frag + d)];
                        }

                        mma<dim>((float*)atile, (float*)btile, cout);

                //if (ctx == 0 && cty == 0 && wx == 0 && wy == 0 && bidx == 0 &&
                //                bidy == 0) {
                //        //dump_matrix((float*)atile, 8, 1);
                //        dump_matrix((float*)btile, 8, 1);
                //        //dump_matrix(cout, 8, 8);
                //        printf("Thread: %d %d Tile: %d Frag: %d Block: %d %d \n", tx,
                //                        ty, tile, frag, bidx, bidy);
                //        //dump_matrix128((float*)sA, 4 * sx, tile_size);  
                //        //dump_matrix128((float*)sm, 128, 36);  
                //}

                }




                // Reset shared memory pointer
                store_frag_ptr = store_frag_ptr_start;

                __syncthreads();
        }
        if (cidx == 0 & bidx == 0 && bidy == 0)
                printf("cidy = %d \n", cidy);



        // Copy local output tile back to shared memory

        // Working solution but that contains shared memory bank conflicts
        int px = 2 * cidx;
        int py = 8 * cidy;

        //#pragma unroll
        //for (int i = 0; i < 8; ++i)  {
        //#pragma unroll
        //for (int j = 0; j < 2; ++j)  {
        //        sA[2 * cidx + j + sx * (py + i)] = ((float4*)cout)[2*i + j];
        //}
        //}

        //__syncthreads();

        //int gidx = tx + 32 * bidx; 
        //int gidy = ty + 8 * bidy; 
        //for (int i = 0; i < 16; ++i) {
        //    ((float4*)C)[gidx + Nf * (16 * gidy + i)] = 
        //    sA[tx + sx * (16 * ty + i)];
        //}

        int gidx = 8 * cidx + (4 * 4 * 8) * bidx; 
        int gidy = 8 * cidy + (4 * 4 * 8) * bidy; 
        for (int i = 0; i < 8; ++i)
        for (int j = 0; j < 8; ++j)
                C[(gidx + j) + N * (gidy + i)] = cout[j + i * 8];


        //return;


        //#pragma unroll
        //for (int i = 0; i < 8; ++i)  {
        //#pragma unroll
        //for (int j = 0; j < 8; ++j)  {
        //        ((float*)sA)[cty + 16 * j +  136 * (ctx + 16 * i)] =
        //                ((float*)cout)[8*i + j];
        //}
        //}




        //if (cidy < 8) {

        //#pragma unroll
        //for (int i = 0; i < 8; ++i)  {
        //#pragma unroll
        //for (int j = 0; j < 8; ++j)  {
        //        ((float*)sA)[(8 + 1) * tx + j + 144 * (8*cidy + i)] = ((float*)cout)[8*i + j];
        //}
        //}

        //}
        //

        //        
        //__syncthreads();

        //int gidx = tx + 32 * bidx; 
        //int gidy = ty + 8 * bidy; 

        //for (int i = 0; i < 8; ++i) {
        //    ((float4*)C)[gidx + Nf * (16 * gidy + i)] = 
        //    sA[tx + sx * (16 * ty + i)];
        //}


        //if (cidy >= 8) {
        //#pragma unroll
        //for (int i = 0; i < 4; ++i)  {
        //#pragma unroll
        //for (int j = 0; j < 8; ++j)  {
        //        ((float*)sA)[(8 + 1) * tx + j + 144 * (8*(cidy - 8) + i)] =
        //                ((float*)cout)[8*(i + 4) + j];
        //}
        //}
        //}
        //        
        //__syncthreads();

        //for (int i = 8; i < 16; ++i) {
        //    ((float4*)C)[gidx + Nf * (16 * gidy + i)] = 
        //    sA[tx + sx * (16 * ty + i)];
        //}




        //for (int i = 0; i < 16; ++i) {
        //for (int j = 0; j < 2; ++j) {
        //    ((float2*)C)[2 * gidx + j + N / 2 * (16 * gidy + i)] = 
        //    ((float2*)sA)[2 * tx + j + 2 * sx * (16 * ty + i)];
        //}
        //}

        //if (ctx == 0 && cty == 0 && wx == 0 && wy == 0 && bidx == 0 &&
        //                        bidy == 0) {
        //                //dump_matrix((float*)atile, 8, 1);
        //                //dump_matrix((float*)btile, 8, 1);
        //                dump_matrix(cout, 8, 8);
        //                printf("Thread: %d %d Tile: %d Frag: %d Block: %d %d \n", tx,
        //                                ty, 0, 0, bidx, bidy);
        //                dump_matrix128((float*)sA, 4 * sx, 1 + tile_size);  
        //                //dump_matrix128((float*)sm, 128, 36);  
        //}


        // Write result back to global memory

        //int gidx = 8 * cidx + (4 * 4 * 8) * bidx; 
        //int gidy = 8 * cidy + (4 * 4 * 8) * bidy; 
        //for (int i = 0; i < 8; ++i)
        //for (int j = 0; j < 8; ++j)
        //        C[(gidx + j) + N * (gidy + i)] = cout[j + i * 8];

}

void gemm_blocked(float *C, float *AT, float *B, int M, int N, int K)
{
                const int tile_size = 32;
                dim3 threads(32, 8, 1);
                dim3 blocks((N - 1) / 128 + 1,
                            (M - 1) / 128 + 1, 1);

                assert( M % 4 == 0);
                assert( N % 4 == 0);

                int maxbytes = 65536;  // 64 KB
                cudaFuncSetAttribute(
                    dgemm_blocked<tile_size>,
                    cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
                int carveout = cudaSharedmemCarveoutMaxShared;
                cudaErrCheck(cudaFuncSetAttribute(
                    dgemm_blocked<tile_size>,
                    cudaFuncAttributePreferredSharedMemoryCarveout, carveout));
                cudaErrCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
                dgemm_blocked<tile_size><<<blocks, threads, maxbytes>>>(C, AT, B, M, N, K);
}
