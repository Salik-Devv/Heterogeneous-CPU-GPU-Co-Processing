#pragma once
/*
 * init_csr.h
 * CSR sparse matrix + dense vector initialisation helpers.
 * Mirrors the role of init_matrix.h in the GEMM workload.
 */

#include <cstdlib>

/* ── CSR matrix struct ───────────────────────────────────────────────────── */
struct CSRMatrix {
    int    *row_ptr;   /* length M+1  */
    int    *col_idx;   /* length nnz  */
    double *val;       /* length nnz  */
    int     M, N;
    long    nnz;
};

/*
 * init_csr()
 * Allocate and fill a CSR matrix with exactly nnz_per_row non-zeros per row.
 * Column indices are chosen uniformly at random; values are in (0, 1].
 * Caller must free with free_csr().
 */
inline CSRMatrix *init_csr(int M, int N, int nnz_per_row,
                            unsigned int seed = 42) {
    CSRMatrix *A = new CSRMatrix;
    A->M       = M;
    A->N       = N;
    A->nnz     = (long)M * nnz_per_row;
    A->row_ptr = new int   [M + 1];
    A->col_idx = new int   [A->nnz];
    A->val     = new double[A->nnz];

    srand(seed);

    A->row_ptr[0] = 0;
    for (int i = 0; i < M; i++) {
        A->row_ptr[i + 1] = A->row_ptr[i] + nnz_per_row;
        long base = (long)i * nnz_per_row;
        for (int k = 0; k < nnz_per_row; k++) {
            A->col_idx[base + k] = (int)((unsigned)rand() % (unsigned)N);
            A->val    [base + k] = static_cast<double>(rand()) / RAND_MAX + 1e-6;
        }
    }
    return A;
}

/*
 * free_csr()
 * Release all memory owned by a CSRMatrix returned by init_csr().
 */
inline void free_csr(CSRMatrix *A) {
    delete[] A->row_ptr;
    delete[] A->col_idx;
    delete[] A->val;
    delete A;
}

/*
 * init_vector()
 * Fill a dense vector of length n with uniform random values in (0, 1].
 * Mirrors init_matrix() but for 1-D arrays.
 */
inline void init_vector(double *v, int n) {
    for (int i = 0; i < n; i++)
        v[i] = static_cast<double>(rand()) / RAND_MAX + 1e-6;
}