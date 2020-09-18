/*
 Matrix-Matrix multiplication C = A * B, for matrices A, B 
 Dimensions:
   A: (*m, k)
   B: (*k, n)
   C: (*m, n)
 
 Ordering:
   A: row-major
   B: row-major
   C: row-major
 */
template <typename Cv, typename Av, typename Ti>
__global__ void gemm_rrr(Cv * __restrict__ C, 
                         const Av * __restrict__ A, 
                         const Av * __restrict__ B, Ti m, Ti n, Ti k)
{

        // row index
        int i = threadIdx.y + blockIdx.y * blockDim.y;
        // col index
        int j = threadIdx.x + blockIdx.x * blockDim.x;

        if (i >= m || j >= n) return;

        Cv temp = 0.0f;
        for (int l = 0; l < k; ++l) {
                temp += A[l + k * i] * B[j + n * l]; 
        }
        C[j + n * i] = temp;


}

