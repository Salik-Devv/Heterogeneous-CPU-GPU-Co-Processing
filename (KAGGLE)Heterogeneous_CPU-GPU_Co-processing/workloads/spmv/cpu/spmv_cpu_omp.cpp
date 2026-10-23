/*
 * spmv_cpu_omp.cpp
 * Sparse Matrix-Vector Multiplication (SpMV) — CPU OpenMP
 * Format: Compressed Sparse Row (CSR)
 *
 *   y = A * x
 *   A : M x N sparse matrix in CSR format
 *   x : dense input  vector (length N)
 *   y : dense output vector (length M)
 *
 * Each row of the output is independent => trivially parallelisable.
 * Parallelism: omp parallel for over rows (i-loop).
 *
 * Usage:
 *   ./spmv_cpu_omp <M> <N> <nnz_per_row> <threads>
 *
 * Example:
 *   ./spmv_cpu_omp 131072 131072 32 8
 *
 * Compile:
 *   g++ -O3 -march=native -fopenmp -o spmv_cpu_omp spmv_cpu_omp.cpp -lm
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <omp.h>
#include "../init_csr.h"   /* CSRMatrix, init_csr(), init_vector(), free_csr() */
#include "../spmv_csv.h"   /* save_spmv_csv() */

/* ── Benchmark knobs ─────────────────────────────────────────────────────── */
#define N_RUNS   8
#define N_WARMUP 1

/* ── Core SpMV kernel — OpenMP parallelised over rows ────────────────────── */
static void spmv_omp(const CSRMatrix *A,
                      const double   *x,
                            double   *y) {
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < A->M; i++) {
        double sum = 0.0;
        int row_start = A->row_ptr[i];
        int row_end   = A->row_ptr[i + 1];
        for (int j = row_start; j < row_end; j++)
            sum += A->val[j] * x[A->col_idx[j]];
        y[i] = sum;
    }
}

/* ── Statistics helpers ──────────────────────────────────────────────────── */
static double mean_d(const double *a, int n) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += a[i];
    return s / n;
}
static double stddev_d(const double *a, int n, double m) {
    double s = 0.0;
    for (int i = 0; i < n; i++) s += (a[i] - m) * (a[i] - m);
    return sqrt(s / n);
}

/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[]) {
    if (argc < 5) {
        printf("Usage: %s <M> <N> <nnz_per_row> <threads>\n", argv[0]);
        printf("Example: %s 131072 131072 32 8\n", argv[0]);
        return 1;
    }

    int M           = atoi(argv[1]);
    int N           = atoi(argv[2]);
    int nnz_per_row = atoi(argv[3]);
    int threads     = atoi(argv[4]);

    if (M <= 0 || N <= 0 || nnz_per_row <= 0 || nnz_per_row > N || threads <= 0) {
        fprintf(stderr, "Invalid arguments.\n");
        return 1;
    }

    omp_set_num_threads(threads);
    long nnz = (long)M * nnz_per_row;

    printf("SpMV CPU OMP -- CSR format\n");
    printf("  Matrix  : %d x %d,  nnz = %ld,  nnz/row = %d\n",
           M, N, nnz, nnz_per_row);
    printf("  Density : %.4f%%\n", 100.0 * nnz / ((double)M * N));
    printf("  Threads : %d\n", threads);

    /* ── Allocate & initialise (via init_csr.h) ── */
    printf("Generating matrix... ");
    fflush(stdout);
    CSRMatrix *A = init_csr(M, N, nnz_per_row, /*seed=*/42);
    printf("done.\n");

    double *x = new double[N];
    double *y = new double[M];

    srand(0);
    init_vector(x, N);
    memset(y, 0, M * sizeof(double));

    /* ── Memory traffic & arithmetic intensity ── */
    double bytes = (M + 1) * 4.0   /* row_ptr */
                 + nnz * 4.0        /* col_idx */
                 + nnz * 8.0        /* val     */
                 + nnz * 8.0        /* x (worst-case: no cache reuse) */
                 + M   * 8.0;       /* y       */
    double flops = 2.0 * nnz;       /* 1 mul + 1 add per non-zero */

    printf("  Est. AI : %.4f FLOP/byte\n\n", flops / bytes);

    /* ── Warmup ── */
    for (int r = 0; r < N_WARMUP; r++)
        spmv_omp(A, x, y);

    /* ── Benchmark ── */
    double times[N_RUNS];
    for (int r = 0; r < N_RUNS; r++) {
        memset(y, 0, M * sizeof(double));
        double t0 = omp_get_wtime();
        spmv_omp(A, x, y);
        double t1 = omp_get_wtime();
        times[r] = (t1 - t0) * 1e3;   /* ms */
    }

    double m_time = mean_d(times, N_RUNS);
    double sd     = stddev_d(times, N_RUNS, m_time);
    double bw     = (bytes / 1e9) / (m_time / 1e3);   /* GB/s   */
    double gfl    = (flops / 1e9) / (m_time / 1e3);   /* GFLOPS */

    double checksum = 0.0;
    for (int i = 0; i < M; i++) checksum += y[i];

    /* ── Print (mirrors gemm_cpu_omp.cpp printf style) ── */
    printf("CPU-OMP SpMV | M=%d | N=%d | nnz=%ld | nnz_row=%d | Threads=%d | "
           "Time=%.2f ms | BW=%.4f GB/s | GFLOPS=%.6f | Checksum=%.6f\n",
           M, N, nnz, nnz_per_row, threads, m_time, bw, gfl, checksum);

    printf("\nPer-run times (ms):\n  ");
    for (int r = 0; r < N_RUNS; r++) printf("%.3f  ", times[r]);
    printf("\n  Std-dev: %.3f ms\n", sd);

    printf("\nFirst 4 output values: %.6f  %.6f  %.6f  %.6f\n",
           y[0], y[1], y[2], y[3]);

    /* ── Save to CSV (mirrors save_gemm_csv() call in gemm_cpu_omp.cpp) ── */
    save_spmv_csv(
        "CPU_OMP",
        M, N, nnz, nnz_per_row,
        /*threads=*/       threads,
        /*cpu_fraction=*/  1.0,
        /*cpu_rows=*/      M,
        /*gpu_rows=*/      0,
        /*compute_ms=*/    m_time,
        /*wall_total=*/    m_time,
        /*cpu_part_ms=*/   m_time,
        /*gpu_kernel_ms=*/ -1.0,
        /*h2d_ms=*/        -1.0,
        /*d2h_ms=*/        -1.0,
        /*bandwidth_GBs=*/ bw,
        /*gflops=*/        gfl,
        /*checksum=*/      checksum,
        /*bottleneck=*/    "Memory"
    );

    delete[] x;
    delete[] y;
    free_csr(A);
    return 0;
}