/*
 * spmv_part.cpp
 * CPU partial SpMV worker — used by the heterogeneous driver.
 * Mirrors the role of prefix_part.cpp in the prefix-sum workload.
 *
 * Each row of SpMV is fully independent (no carry-in dependency
 * unlike prefix sum), so the OMP parallelism here is a simple
 * parallel-for over the assigned row range — no multi-pass needed.
 */

#include "spmv_part.h"
#include <omp.h>

void spmv_cpu_part(const CSRMatrix *A,
                   const float     *x,
                         float     *y,
                   int row_start,
                   int row_end)
{
    if (row_start >= row_end) return;

    /*
     * Each thread owns a contiguous slice of rows.
     * Access to val[] and col_idx[] is sequential within each row
     * (good cache behaviour). Access to x[] is random (same as GPU),
     * but the CPU's large L3 cache amortises this better than the GPU
     * for small-to-medium matrices.
     */
    #pragma omp parallel for schedule(static)
    for (int i = row_start; i < row_end; i++) {
        float sum = 0.0f;
        int rs = A->row_ptr[i];
        int re = A->row_ptr[i + 1];
        for (int j = rs; j < re; j++)
            sum += (float)A->val[j] * x[A->col_idx[j]];
        y[i] = sum;
    }
}