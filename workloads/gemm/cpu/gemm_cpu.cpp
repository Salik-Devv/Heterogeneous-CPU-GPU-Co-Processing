#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "../gemm_csv.h"

void gemm_serial(float *A, float *B, float *C, int N) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    srand(0);

    // C++-correct memory allocation
    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N*N; i++) {
        A[i] = (float)rand() / RAND_MAX;
        B[i] = (float)rand() / RAND_MAX;
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    gemm_serial(A, B, C, N);

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) * 1e3 +
                     (end.tv_nsec - start.tv_nsec) / 1e6;

    double gflops = (2.0 * N * N * N) / (elapsed * 1e6);

    double checksum = 0.0;
    for (int i = 0; i < N*N; i++)
        checksum += C[i];

    printf("CPU GEMM | N=%d | Time=%.2f ms | GFLOPS=%.4f | Checksum=%.6f\n",
           N, elapsed, gflops, checksum);

    // ── Save results to CSV ──
    save_gemm_csv(
        "CPU_Serial",
        N,
        /*threads=*/      1,
        /*cpu_fraction=*/ 1.0,
        /*cpu_rows=*/     N,
        /*gpu_rows=*/     0,
        /*compute_ms=*/   elapsed,
        /*wall_total=*/   elapsed,
        /*cpu_part_ms=*/  elapsed,
        /*gpu_kernel=*/  -1.0,
        /*h2d_ms=*/      -1.0,
        /*d2h_ms=*/      -1.0,
        gflops, checksum,
        /*bottleneck=*/   "CPU"
    );

    // C++-correct deallocation
    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}