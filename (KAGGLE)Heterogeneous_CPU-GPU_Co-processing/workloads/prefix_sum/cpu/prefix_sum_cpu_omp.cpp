#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "../init_array.h"
#include "../prefix_sum_csv.h"

void prefix_sum_omp(const double *in, double *out, long N, int num_threads) {
    double *chunk_sums = (double*)calloc(num_threads, sizeof(double));

    // Phase 1: local inclusive scan per chunk
    #pragma omp parallel num_threads(num_threads)
    {
        int  tid   = omp_get_thread_num();
        long chunk = (N + num_threads - 1) / num_threads;
        long start = (long)tid * chunk;
        long end   = start + chunk;
        if (end > N) end = N;

        if (start < N) {
            out[start] = in[start];
            for (long i = start + 1; i < end; i++)
                out[i] = out[i-1] + in[i];
            chunk_sums[tid] = out[end - 1];
        }
    }

    // Phase 2: serial scan of chunk totals
    for (int t = 1; t < num_threads; t++)
        chunk_sums[t] += chunk_sums[t-1];

    // Phase 3: add correction to each chunk
    #pragma omp parallel num_threads(num_threads)
    {
        int  tid   = omp_get_thread_num();
        long chunk = (N + num_threads - 1) / num_threads;
        long start = (long)tid * chunk;
        long end   = start + chunk;
        if (end > N) end = N;

        if (tid > 0 && start < N) {
            double offset = chunk_sums[tid - 1];
            for (long i = start; i < end; i++)
                out[i] += offset;
        }
    }

    free(chunk_sums);
}

void run_benchmark(long N, int num_threads) {
    double *in  = alloc_array(N);
    double *out = alloc_array(N);
    double *ref = alloc_array(N);

    init_array_random(in, N, ARRAY_SEED);

    // Serial reference
    ref[0] = in[0];
    for (long i = 1; i < N; i++)
        ref[i] = ref[i-1] + in[i];

    double t0 = omp_get_wtime();
    prefix_sum_omp(in, out, N, num_threads);
    double ms = (omp_get_wtime() - t0) * 1e3;

    double checksum = compute_checksum_array(out, N);
    double gb_s     = (2.0 * N * sizeof(double)) / (ms * 1e6);

    printf("CPU Prefix Sum (OMP)    | N=%-10ld | Threads=%2d | "
           "Time=%8.3f ms | BW=%6.2f GB/s | Checksum=%.6f\n",
           N, num_threads, ms, gb_s, checksum);

    if (N <= (1 << 24))
        verify_prefix(ref, out, N, 1e-6);

    // ── Save results to CSV ──
    save_prefix_csv(
        "CPU_OMP",
        N,
        /*threads=*/          num_threads,
        /*cpu_fraction=*/     1.0,
        /*cpu_elems=*/        N,
        /*gpu_elems=*/        0,
        /*compute_ms=*/       ms,
        /*wall_total_ms=*/   -1.0,   // same as compute
        /*cpu_part_ms=*/      ms,
        /*gpu_kernel_ms=*/   -1.0,   // N/A
        /*sync_overhead_ms=*/-1.0,   // N/A
        /*h2d_ms=*/          -1.0,   // N/A
        /*d2h_ms=*/          -1.0,   // N/A
        gb_s, checksum,
        /*bottleneck=*/       "CPU"
    );

    free(in); free(out); free(ref);
}

int main(int argc, char *argv[]) {
    int max_threads = omp_get_max_threads();

    if (argc >= 2) {
        long N      = atol(argv[1]);
        int threads = (argc >= 3) ? atoi(argv[2]) : max_threads;
        printf("=== CPU Prefix Sum (OMP) | N=%ld | Threads=%d ===\n\n",
               N, threads);
        run_benchmark(N, threads);
    } else {
        printf("=== CPU Prefix Sum (OMP) — Size + Thread Sweep ===\n\n");
        int thread_counts[] = { 1, max_threads/2, max_threads };
        for (int t = 0; t < 3; t++) {
            int threads = thread_counts[t] < 1 ? 1 : thread_counts[t];
            printf("-- Threads = %d --\n", threads);
            for (int i = 0; i < NUM_PREFIX_SIZES; i++)
                run_benchmark(PREFIX_SIZES[i], threads);
            printf("\n");
        }
    }
    return 0;
}