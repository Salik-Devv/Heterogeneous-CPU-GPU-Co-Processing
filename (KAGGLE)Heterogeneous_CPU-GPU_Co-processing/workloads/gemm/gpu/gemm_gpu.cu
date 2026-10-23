// gemm_gpu.cu
// Usage: ./gemm_gpu <N> [mode: 0=naive 1=tiled 2=cublas 3=all]

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../init_matrix.h"
#include "../gemm_csv.h"

#define TILE_SIZE 16

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
// Checksum — sum of all elements in C
// ─────────────────────────────────────────────
double compute_checksum(float *M, int N) {
    double sum = 0.0;
    for (int i = 0; i < N * N; i++)
        sum += M[i];
    return sum;
}


// ─────────────────────────────────────────────
// Kernel 1: Naive
// ─────────────────────────────────────────────
__global__ void gemm_naive(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}


// ─────────────────────────────────────────────
// Kernel 2: Tiled shared-memory
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
// Run kernel, time it, print result, save CSV
// h2d_ms passed in from main (measured once)
// ─────────────────────────────────────────────
void run_and_time(
    const char *label,
    float *d_A, float *d_B, float *d_C,
    float *h_C,
    int N, int mode,
    cublasHandle_t handle,
    float h2d_ms)             // H→D time measured once in main
{
    size_t bytes = (size_t)N * N * sizeof(float);

    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaMemset(d_C, 0, bytes));

    CUDA_CHECK(cudaEventRecord(ev_start));

    if (mode == 0) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, N);

    } else if (mode == 1) {
        dim3 block(TILE_SIZE, TILE_SIZE);
        dim3 grid((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, N);

    } else {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N, N, N,
            &alpha,
            d_B, N,
            d_A, N,
            &beta,
            d_C, N
        ));
    }

    CUDA_CHECK(cudaEventRecord(ev_stop));
    CUDA_CHECK(cudaEventSynchronize(ev_stop));
    CUDA_CHECK(cudaGetLastError());

    float gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, ev_start, ev_stop));

    // ── D→H transfer timing ──
    cudaEvent_t d2h_start, d2h_stop;
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&d2h_start));
    CUDA_CHECK(cudaEventCreate(&d2h_stop));
    CUDA_CHECK(cudaEventRecord(d2h_start));
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_stop));
    CUDA_CHECK(cudaEventSynchronize(d2h_stop));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_start, d2h_stop));

    double checksum   = compute_checksum(h_C, N);
    double gflops     = (2.0 * N * N * N) / (gpu_kernel_ms * 1e6);
    double wall_total = h2d_ms + gpu_kernel_ms + d2h_ms;

    printf("GPU GEMM (%s) | N=%d | Kernel=%.2f ms | H2D=%.2f ms | D2H=%.2f ms"
           " | Total=%.2f ms | GFLOPS=%.4f | Checksum=%.6f\n",
           label, N, gpu_kernel_ms, h2d_ms, d2h_ms, wall_total, gflops, checksum);

    // ── Save results to CSV ──
    char impl_name[32];
    snprintf(impl_name, sizeof(impl_name), "GPU_%s", label);

    save_gemm_csv(
        impl_name,
        N,
        /*threads=*/      1,
        /*cpu_fraction=*/ 0.0,
        /*cpu_rows=*/     0,
        /*gpu_rows=*/     N,
        /*compute_ms=*/   gpu_kernel_ms,
        /*wall_total=*/   wall_total,
        /*cpu_part_ms=*/ -1.0,
        /*gpu_kernel=*/   gpu_kernel_ms,
        /*h2d_ms=*/       h2d_ms,
        /*d2h_ms=*/       d2h_ms,
        gflops, checksum,
        /*bottleneck=*/   "GPU"
    );

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
}


int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s <N> [mode: 0=naive 1=tiled 2=cublas 3=all]\n", argv[0]);
        return 1;
    }

    int N    = atoi(argv[1]);
    int mode = (argc >= 3) ? atoi(argv[2]) : 3;
    size_t bytes = (size_t)N * N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    srand(0);
    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // ── Measure H→D transfer (shared across all kernel runs) ──
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
    printf("[H→D xfer] %.3f ms\n\n", h2d_ms);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    if (mode == 0 || mode == 3) run_and_time("Naive",  d_A, d_B, d_C, h_C, N, 0, handle, h2d_ms);
    if (mode == 1 || mode == 3) run_and_time("Tiled",  d_A, d_B, d_C, h_C, N, 1, handle, h2d_ms);
    if (mode == 2 || mode == 3) run_and_time("cuBLAS", d_A, d_B, d_C, h_C, N, 2, handle, h2d_ms);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    free(h_A); free(h_B); free(h_C);

    return 0;
}
