#include <cstdio>
#include <cstdlib>
#include <omp.h>

#include "../init_matrix.h"
#include "../gemm_csv.h"

void gemm_omp(float *A, float *B, float *C, int N) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++)
                sum += A[i*N + k] * B[k*N + j];
            C[i*N + j] = sum;
        }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <N> <threads>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    int threads = atoi(argv[2]);
    omp_set_num_threads(threads);

    srand(0);

    float *A = new float[N*N];
    float *B = new float[N*N];
    float *C = new float[N*N];

    init_matrix(A, N);
    init_matrix(B, N);

    double start = omp_get_wtime();
    gemm_omp(A, B, C, N);
    double elapsed = (omp_get_wtime() - start) * 1e3;

    double gflops = (2.0 * N * N * N) / (elapsed * 1e6);

    double checksum = 0.0;
    for (int i = 0; i < N*N; i++)
        checksum += C[i];

    printf("CPU-OMP GEMM | N=%d | Threads=%d | Time=%.2f ms | GFLOPS=%.4f | Checksum=%.6f\n",
           N, threads, elapsed, gflops, checksum);

    // ── Save results to CSV ──
    save_gemm_csv(
        "CPU_OMP",
        N,
        /*threads=*/      threads,
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

    delete[] A;
    delete[] B;
    delete[] C;

    return 0;
}