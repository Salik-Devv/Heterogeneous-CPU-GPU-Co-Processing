// cpu_part.h
#pragma once

// Computes a row-partitioned slice of C = A * B
// A, B, C  — full N×N matrices (row-major)
// N        — matrix dimension
// row_start, row_end — half-open row range this call owns [row_start, row_end)
void gemm_cpu_part(const float *A, const float *B, float *C,
                   int N, int row_start, int row_end);

// Returns omp_get_max_threads() — call before any parallel region
// so the value is available to main() for logging and CSV
int gemm_cpu_num_threads(void);