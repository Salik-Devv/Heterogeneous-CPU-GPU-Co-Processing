// gemm_hetero.cu
// CPU + GPU co-processing GEMM
// Splits row work between CPU (OpenMP parallel) and GPU (tiled kernel)
// Both run simultaneously via POSIX thread + CUDA stream
//
// Usage: ./gemm_hetero <N> <cpu_fraction 0.0-1.0>
// Example: ./gemm_hetero 1024 0.3   → CPU gets 30% of rows, GPU gets 70%

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../init_matrix.h"
#include "cpu_part.h"
#include "../gemm_csv.h"

#define TILE_SIZE 16

// ─────────────────────────────────────────────
// Error macros
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",               \
                    __FILE__, __LINE__, cudaGetErrorString(err));        \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define CUBLAS_CHECK(call)                                              \
    do {                                                                 \
        cublasStatus_t st = (call);                                      \
        if (st != CUBLAS_STATUS_SUCCESS) {                               \
            fprintf(stderr, "cuBLAS error at %s:%d — code %d\n",        \
                    __FILE__, __LINE__, st);                             \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)


// ─────────────────────────────────────────────
// GPU tiled kernel
// ─────────────────────────────────────────────
__global__ void gemm_tiled(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; t++) {
        int aCol = t * TILE_SIZE + tx;
        sA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;

        int bRow = t * TILE_SIZE + ty;
        sB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            sum += sA[ty][k] * sB[k][tx];
        __syncthreads();
    }

    if (row < N && col < N)
        C[row * N + col] = sum;
}


// ─────────────────────────────────────────────
// CPU thread argument struct
// ─────────────────────────────────────────────
typedef struct {
    const float *A;
    const float *B;
    float       *C;
    int          N;
    int          row_start;
    int          row_end;
    double       elapsed_ms;
} CpuThreadArgs;


// ─────────────────────────────────────────────
// CPU worker thread — runs gemm_cpu_part() (OMP inside)
// ─────────────────────────────────────────────
void *cpu_worker(void *arg) {
    CpuThreadArgs *a = (CpuThreadArgs*)arg;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    gemm_cpu_part(a->A, a->B, a->C, a->N, a->row_start, a->row_end);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    a->elapsed_ms = (t1.tv_sec  - t0.tv_sec)  * 1e3 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return NULL;
}


// ─────────────────────────────────────────────
// Checksum over full C matrix
// ─────────────────────────────────────────────
double compute_checksum(float *M, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; i++)
        sum += (double)M[i];
    return sum;
}


// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <N> <cpu_fraction>\n", argv[0]);
        printf("  cpu_fraction: 0.0 to 1.0 (e.g. 0.3 = CPU gets 30%% of rows)\n");
        return 1;
    }

    int   N            = atoi(argv[1]);
    float cpu_fraction = atof(argv[2]);

    // Read OMP thread count before any parallel region
    int omp_threads = gemm_cpu_num_threads();

    // Row split
    int cpu_rows = (int)(N * cpu_fraction);
    int gpu_rows = N - cpu_rows;

    int gpu_row_start = cpu_rows;

    size_t bytes     = (size_t)N * N * sizeof(float);
    size_t gpu_bytes = (size_t)gpu_rows * N * sizeof(float);

    printf("=== GEMM Hetero | N=%d | CPU=%.0f%% (%d rows, %d OMP threads) | GPU=%.0f%% (%d rows) ===\n\n",
           N, cpu_fraction*100, cpu_rows, omp_threads,
           (1.0f-cpu_fraction)*100, gpu_rows);

    // ── Allocate host matrices ──
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    memset(h_C, 0, bytes);

    srand(0);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    // ── Device allocation (only GPU partition of A and C) ──
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, gpu_bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, gpu_bytes));

    // ── H→D transfer (only GPU's share) ──
    cudaEvent_t xfer_start, xfer_stop;
    CUDA_CHECK(cudaEventCreate(&xfer_start));
    CUDA_CHECK(cudaEventCreate(&xfer_stop));

    CUDA_CHECK(cudaEventRecord(xfer_start));
    CUDA_CHECK(cudaMemcpy(d_A, h_A + (size_t)gpu_row_start * N,
                          gpu_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(xfer_stop));
    CUDA_CHECK(cudaEventSynchronize(xfer_stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, xfer_start, xfer_stop));
    printf("[H→D xfer] %.3f ms\n\n", h2d_ms);

    // ─────────────────────────────────────────
    // LAUNCH CPU (OMP parallel) AND GPU SIMULTANEOUSLY
    // ─────────────────────────────────────────
    CpuThreadArgs cpu_args = {
        .A          = h_A,
        .B          = h_B,
        .C          = h_C,
        .N          = N,
        .row_start  = 0,
        .row_end    = cpu_rows,
        .elapsed_ms = 0.0
    };

    struct timespec wall_start, wall_end;
    clock_gettime(CLOCK_MONOTONIC, &wall_start);

    pthread_t cpu_thread;
    if (cpu_rows > 0)
        pthread_create(&cpu_thread, NULL, cpu_worker, &cpu_args);

    cudaEvent_t gpu_start, gpu_stop;
    float gpu_ms = 0.0f;

    if (gpu_rows > 0) {
        CUDA_CHECK(cudaEventCreate(&gpu_start));
        CUDA_CHECK(cudaEventCreate(&gpu_stop));
        CUDA_CHECK(cudaMemset(d_C, 0, gpu_bytes));

        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N + TILE_SIZE-1)/TILE_SIZE,
                  (gpu_rows + TILE_SIZE-1)/TILE_SIZE);

        CUDA_CHECK(cudaEventRecord(gpu_start));
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);
        CUDA_CHECK(cudaEventRecord(gpu_stop));
        CUDA_CHECK(cudaEventSynchronize(gpu_stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, gpu_start, gpu_stop));
    }

    if (cpu_rows > 0)
        pthread_join(cpu_thread, NULL);

    clock_gettime(CLOCK_MONOTONIC, &wall_end);
    double wall_ms = (wall_end.tv_sec  - wall_start.tv_sec)  * 1e3 +
                     (wall_end.tv_nsec - wall_start.tv_nsec) / 1e6;

    // ── D→H transfer ──
    float d2h_ms = 0.0f;
    if (gpu_rows > 0) {
        CUDA_CHECK(cudaEventRecord(xfer_start));
        CUDA_CHECK(cudaMemcpy(h_C + (size_t)gpu_row_start * N,
                              d_C, gpu_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(xfer_stop));
        CUDA_CHECK(cudaEventSynchronize(xfer_stop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, xfer_start, xfer_stop));
    }

    double checksum = compute_checksum(h_C, N);
    double gflops   = (2.0 * N * N * N) / (wall_ms * 1e6);

    printf("GPU GEMM (Hetero) | N=%d | Time=%.2f ms | GFLOPS=%.4f | Checksum=%.6f\n",
           N, wall_ms, gflops, checksum);
    printf("\n--- Breakdown ---\n");
    printf("  CPU part  : %d rows | Time=%.2f ms  (%d OMP threads)\n",
           cpu_rows, cpu_args.elapsed_ms, omp_threads);
    printf("  GPU part  : %d rows | Kernel=%.2f ms\n",
           gpu_rows, gpu_ms);
    printf("  D→H xfer  : %.3f ms\n", d2h_ms);
    printf("  Wall time : %.2f ms\n", wall_ms);

    // ── Bottleneck ──
    const char *bottleneck = (cpu_args.elapsed_ms > gpu_ms + d2h_ms) ? "CPU" : "GPU";
    printf("\n--- Bottleneck ---\n");
    printf("  >> %s was the bottleneck (CPU=%.2f ms vs GPU+D2H=%.2f ms)\n",
           bottleneck, cpu_args.elapsed_ms, gpu_ms + d2h_ms);

    // ── Save results to CSV ──
    save_gemm_csv(
        "CPU_GPU_Hetero",
        N,
        /*threads=*/      omp_threads,      // actual OMP thread count
        /*cpu_fraction=*/ cpu_fraction,
        /*cpu_rows=*/     cpu_rows,
        /*gpu_rows=*/     gpu_rows,
        /*compute_ms=*/   wall_ms,
        /*wall_total=*/   wall_ms + d2h_ms,
        /*cpu_part_ms=*/  cpu_args.elapsed_ms,
        /*gpu_kernel=*/   gpu_ms,
        /*h2d_ms=*/       h2d_ms,
        /*d2h_ms=*/       d2h_ms,
        gflops, checksum,
        bottleneck
    );

    // ── Cleanup ──
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(xfer_start); cudaEventDestroy(xfer_stop);
    if (gpu_rows > 0) {
        cudaEventDestroy(gpu_start);
        cudaEventDestroy(gpu_stop);
    }
    free(h_A); free(h_B); free(h_C);

    return 0;
}