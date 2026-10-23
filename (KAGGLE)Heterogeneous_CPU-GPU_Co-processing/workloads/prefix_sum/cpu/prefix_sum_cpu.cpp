#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../init_array.h"
#include "../prefix_sum_csv.h"

void prefix_sum_inclusive(const double *in, double *out, long N) {
    out[0] = in[0];
    for (long i = 1; i < N; i++)
        out[i] = out[i-1] + in[i];
}

void run_benchmark(long N) {
    double *in  = alloc_array(N);
    double *out = alloc_array(N);

    init_array_random(in, N, ARRAY_SEED);

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    prefix_sum_inclusive(in, out, N);
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double ms  = (t1.tv_sec - t0.tv_sec)*1e3 + (t1.tv_nsec - t0.tv_nsec)/1e6;
    double gb_s = (2.0 * N * sizeof(double)) / (ms * 1e6);
    double checksum = compute_checksum_array(out, N);

    printf("CPU Prefix Sum (Serial) | N=%-10ld | Time=%8.3f ms | "
           "BW=%6.2f GB/s | Checksum=%.6f\n", N, ms, gb_s, checksum);

    if (N <= (1 << 20)) {
        double *ones = alloc_array(N);
        double *ref  = alloc_array(N);
        init_array_constant(ones, N, 1.0);
        prefix_sum_inclusive(ones, ref, N);
        int pass = 1;
        for (long i = 0; i < N; i++) {
            if (fabs(ref[i] - (double)(i+1)) > 0.5) {
                printf("  FAILED at i=%ld\n", i); pass = 0; break;
            }
        }
        if (pass) printf("  Correctness PASSED ✓ (ones array gives [1..N])\n");
        free(ones); free(ref);
    }

    // ── Save results to CSV ──
    save_prefix_csv(
        "CPU_Serial",
        N,
        /*threads=*/          1,
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

    free(in); free(out);
}

int main(int argc, char *argv[]) {
    if (argc >= 2) {
        long N = atol(argv[1]);
        printf("=== CPU Prefix Sum (Serial) | N=%ld ===\n\n", N);
        run_benchmark(N);
    } else {
        printf("=== CPU Prefix Sum (Serial) — Size Sweep ===\n\n");
        for (int i = 0; i < NUM_PREFIX_SIZES; i++)
            run_benchmark(PREFIX_SIZES[i]);
    }
    return 0;
}