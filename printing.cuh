#pragma once

typedef struct {
        dim3 n;
        dim3 p;
        dim3 m;
        int l;
        int r;
} align_t;

#define dump_align(a) _dump_align(a, #a)
void _dump_align(const align_t& a, const char *str)
{
           printf("%s = {n = [%d %d %d],"\
               " p = [%d %d %d], "\
               " m = [%d %d %d], l = %d, r = %d}\n", str, a.n.x, a.n.y, a.n.z, 
               a.p.x, a.p.y, a.p.z, a.m.x, a.m.y, a.m.z, a.l, a.r);
}

#define dump_matrix_pitch(A, nx, ny, pitch) _dump_matrix(A, #A, nx, ny, pitch)
#define dump_matrix(A, nx, ny) _dump_matrix(A, #A, nx, ny)
#define dump_matrix128(A, nx, ny) _dump_matrix(A, #A, nx, ny, nx, 4)

template <typename Tv>
__host__ __device__ void _dump_matrix(Tv *A, const char *str, const int nx,
                const int ny, const int pitch=-1, const int stride=1)
{

        int mx = nx;
        if (pitch != -1)
                mx = pitch;
        printf("%s = [\n", str);
        // Header row
        printf("     ");
        for (int j = 0; j < nx; j+=stride) 
                printf("%03d ", j);
        printf("\n");
        printf("----");
        for (int j = 0; j < nx; j+=stride) 
                printf("---");
        printf("\n");

        for (int i = 0; i < ny; i+=1) {
                printf("%03d | ", i);
        for (int j = 0; j < nx; j+=stride) {
                printf("%03d ", (int)A[j + i * mx] );

        }
        printf("\n");
        }
        printf("] \n");

}


#define dump_2d(A, a, ny) _dump_2d(A, a, #A, ny)
template <typename Tv>
void _dump_2d(Tv *A, const align_t& a, const char *str, const int ny) {
        printf("%s = [\n", str);
        for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < 4; ++j) {
                printf("%2.2f ", (double)A[j + i * a.m.x]);

        }
        printf(" ... ");
        for (int j = a.n.x-4; j < a.n.x; ++j) {
                printf("%2.2f ", (double)A[j + i * a.m.x]);
        }
        printf("\n");
        }
        printf("] \n");

}

#define dump_2d_full(A, a, nx, ny) _dump_2d_full(A, a, #A, nx, ny)
template <typename Tv>
void _dump_2d_full(Tv *A, const align_t& a, const char *str, const int nx, 
                const int ny) {
        printf("%s = [\n", str);
        for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
                printf("%2.2f ", (double)A[j + i * a.m.x]);

        }
        printf("\n");
        }
        printf("] \n");

}
