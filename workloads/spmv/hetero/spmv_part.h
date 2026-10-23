/*
 * spmv_part.h
 * CPU partial SpMV worker — used by the heterogeneous driver.
 * Mirrors the role of prefix_part.h in the prefix-sum workload.
 *
 * Computes y[row_start .. row_end) = A[row_start..row_end) * x
 * using OpenMP parallelism over the assigned row range.
 */
#pragma once
#include "../init_csr.h"   /* CSRMatrix */

/*
 * spmv_cpu_part()
 * Runs an OpenMP-parallel SpMV on rows [row_start, row_end).
 * Writes results directly into y[] at the correct global indices.
 *
 *  A         -- full CSR matrix (host)
 *  x         -- full input vector (length N)
 *  y         -- full output vector (length M); only [row_start,row_end) written
 *  row_start -- first row this CPU partition owns
 *  row_end   -- one past last row this CPU partition owns
 */
void spmv_cpu_part(const CSRMatrix *A,
                   const float     *x,
                         float     *y,
                   int row_start,
                   int row_end);