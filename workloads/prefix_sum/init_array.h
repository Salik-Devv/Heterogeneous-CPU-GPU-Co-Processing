#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define ARRAY_SEED 42

// ── Use double to avoid float32 precision loss in prefix sum ──
static inline void init_array_random(double *arr, long N, unsigned int seed) {
    srand(seed);
    for (long i = 0; i < N; i++)
        arr[i] = (double)((rand() % 10) + 1);  // values 1..10
}

static inline void init_array_constant(double *arr, long N, double val) {
    for (long i = 0; i < N; i++)
        arr[i] = val;
}

static inline void init_array_zero(double *arr, long N) {
    memset(arr, 0, N * sizeof(double));
}

static inline double* alloc_array(long N) {
    double *arr = (double*)malloc(N * sizeof(double));
    if (!arr) {
        fprintf(stderr, "alloc_array: out of memory for N=%ld\n", N);
        exit(EXIT_FAILURE);
    }
    return arr;
}

static inline int verify_prefix(const double *ref,
                                  const double *test,
                                  long N, double tol) {
    double max_err = 0.0;
    long   err_idx = -1;
    for (long i = 0; i < N; i++) {
        double diff = fabs(ref[i] - test[i]);
        if (diff > max_err) { max_err = diff; err_idx = i; }
    }
    if (max_err > tol) {
        printf("  MISMATCH: max_err=%.6f at idx=%ld | ref=%.6f got=%.6f\n",
               max_err, err_idx, ref[err_idx], test[err_idx]);
        return 0;
    }
    printf("  Verification PASSED ✓ (max_err=%.2e)\n", max_err);
    return 1;
}

static inline double compute_checksum_array(const double *arr, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++)
        sum += arr[i];
    return sum;
}

static inline void print_array_preview(const char *label,
                                        const double *arr, long N) {
    int preview = (N < 8) ? (int)N : 4;
    printf("[%s] N=%ld | first %d: ", label, N, preview);
    for (int i = 0; i < preview; i++) printf("%.4f ", arr[i]);
    printf("... last %d: ", preview);
    for (long i = N - preview; i < N; i++) printf("%.4f ", arr[i]);
    printf("\n");
}

static const long PREFIX_SIZES[] = {
    1 << 16,
    1 << 20,
    1 << 22,
    1 << 24,
    1 << 26,
    1 << 27
};
static const int NUM_PREFIX_SIZES = 6;