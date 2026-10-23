// cpu_part.cpp
#include "cpu_part.h"
#include <omp.h>

void gemm_cpu_part(const float *A, const float *B, float *C,
                   int N, int row_start, int row_end)
{
    #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int i = row_start; i < row_end; i++) {
        for (int k = 0; k < N; k++) {
            float a_val = A[i * N + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_val * B[k * N + j];
            }
        }
    }
}

int gemm_cpu_num_threads(void) {
    return omp_get_max_threads();
}