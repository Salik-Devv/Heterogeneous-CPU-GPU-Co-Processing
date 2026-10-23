/*
 * spmv_hetero.cu
 * CPU + GPU co-processing Sparse Matrix-Vector Multiplication (SpMV)
 *
 *   y = A * x
 *   A : M x N sparse matrix in CSR format
 *   x : dense input  vector (length N)
 *   y : dense output vector (length M)
 *
 * Split strategy (by rows — SpMV rows are fully independent):
 *   CPU handles rows [0,        cpu_rows)  -- float32, OMP parallel
 *   GPU handles rows [cpu_rows, M)         -- float32, warp-reduction kernel
 *
 * Timeline:
 *   1. H->D  : transfer GPU partition of CSR + full x vector (async)
 *   2. PARALLEL: CPU scans its rows (pthread+OMP) || GPU kernel runs
 *      -- No sync dependency between partitions (unlike prefix sum)
 *         because SpMV rows are fully independent.
 *   3. D->H  : copy GPU output partition back
 *   4. y[] is now complete: CPU wrote [0, cpu_rows), GPU wrote [cpu_rows, M)
 *
 * Usage:
 *   ./spmv_hetero <M> <N> <nnz_per_row> <cpu_fraction 0.0-1.0>
 *
 * Compile (manual):
 *   nvcc -O3 -arch=sm_89 -Xcompiler -fopenmp -o spmv_hetero spmv_hetero.cu
 *        spmv_part.cpp -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../init_csr.h"   /* CSRMatrix, init_csr(), free_csr() */
#include "../spmv_csv.h"   /* save_spmv_csv()                   */
#include "spmv_part.h"     /* spmv_cpu_part()                   */

/* ── CUDA error guard ────────────────────────────────────────────────────── */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(_e));             \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* ── Launch config ───────────────────────────────────────────────────────── */
#define WARP_SIZE  32
#define BLOCK_SIZE 256   /* 256/32 = 8 warps per block = 8 rows per block  */


/* ══════════════════════════════════════════════════════════════════════════
 * GPU kernel -- warp reduction (one warp per row)
 * Identical to spmv_gpu.cu: GPU partition is [0, gpu_rows) in device arrays
 * which maps to [cpu_rows, M) in the global y[].
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void spmv_warp(
    const int   * __restrict__ row_ptr,
    const int   * __restrict__ col_idx,
    const float * __restrict__ val,
    const float * __restrict__ x,
          float * __restrict__ y,
    int gpu_rows)
{
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    if (warp_id >= gpu_rows) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    float sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        sum += val[j] * x[col_idx[j]];

    sum += __shfl_down_sync(0xffffffff, sum, 16);
    sum += __shfl_down_sync(0xffffffff, sum,  8);
    sum += __shfl_down_sync(0xffffffff, sum,  4);
    sum += __shfl_down_sync(0xffffffff, sum,  2);
    sum += __shfl_down_sync(0xffffffff, sum,  1);

    if (lane == 0)
        y[warp_id] = sum;
}


/* ── Warmup kernel ───────────────────────────────────────────────────────── */
__global__ void warmup_kernel(float *buf) { buf[threadIdx.x] = 0.0f; }


/* ══════════════════════════════════════════════════════════════════════════
 * CPU worker (pthread) — mirrors cpu_worker() in prefix_sum_hetero.cu
 * ══════════════════════════════════════════════════════════════════════════ */
typedef struct {
    const CSRMatrix *A;
    const float     *x;
          float     *y;
    int              row_start;
    int              row_end;
    double           elapsed_ms;
} CpuArgs;

static void *cpu_worker(void *arg)
{
    CpuArgs *a = (CpuArgs*)arg;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    spmv_cpu_part(a->A, a->x, a->y, a->row_start, a->row_end);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    a->elapsed_ms = (t1.tv_sec  - t0.tv_sec)  * 1e3
                  + (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return NULL;
}


/* ── Timing helper ───────────────────────────────────────────────────────── */
static inline double now_ms(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}


/* ── CPU double-precision reference for verification ─────────────────────── */
static void cpu_ref(const CSRMatrix *A, const float *x, double *y_ref)
{
    for (int i = 0; i < A->M; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
            sum += A->val[j] * (double)x[A->col_idx[j]];
        y_ref[i] = sum;
    }
}


/* ══════════════════════════════════════════════════════════════════════════
 * run_benchmark()
 * ══════════════════════════════════════════════════════════════════════════ */
static void run_benchmark(int M, int N, int nnz_per_row, float cpu_fraction)
{
    int cpu_rows = (int)(M * cpu_fraction);
    int gpu_rows = M - cpu_rows;
    int omp_threads = omp_get_max_threads();
    long nnz      = (long)M * nnz_per_row;
    long gpu_nnz  = (long)gpu_rows * nnz_per_row;

    printf("=== SpMV Hetero | M=%d | N=%d | nnz/row=%d | "
           "CPU=%.0f%% (%d rows, %d OMP threads) | GPU=%.0f%% (%d rows) ===\n\n",
           M, N, nnz_per_row,
           cpu_fraction * 100.0f, cpu_rows, omp_threads,
           (1.0f - cpu_fraction) * 100.0f, gpu_rows);

    /* ── Build full CSR on host ── */
    CSRMatrix *A = init_csr(M, N, nnz_per_row, /*seed=*/42);

    /*
     * Convert to float for GPU.
     * Pinned host memory for async transfers (mirrors prefix hetero).
     * h_x and h_y are full-length; GPU only uses [cpu_rows, M) slice.
     */
    float *h_val, *h_x, *h_y;
    CUDA_CHECK(cudaMallocHost(&h_val, nnz * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_x,   N   * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_y,   M   * sizeof(float)));

    srand(0);
    for (long i = 0; i < nnz; i++) h_val[i] = (float)A->val[i];
    for (int  i = 0; i < N;   i++)
        h_x[i] = (float)((double)rand() / RAND_MAX + 1e-6);
    memset(h_y, 0, M * sizeof(float));

    /* CPU double reference */
    double *h_ref = (double*)malloc(M * sizeof(double));
    cpu_ref(A, h_x, h_ref);

    /*
     * GPU CSR partition:
     *   row_ptr : A->row_ptr[cpu_rows .. M]  length = gpu_rows+1
     *             but offsets must be rebased to start from 0
     *   col_idx : A->col_idx[gpu_nnz_offset .. nnz]
     *   val     : h_val[gpu_nnz_offset .. nnz]
     *
     * The GPU kernel sees a local CSR of gpu_rows rows starting at row 0.
     */
    long gpu_nnz_offset = (long)A->row_ptr[cpu_rows]; /* byte offset in col_idx/val */
    long gpu_nnz_count  = nnz - gpu_nnz_offset;

    /* Rebase row_ptr for GPU partition */
    int *h_gpu_row_ptr = (int*)malloc((gpu_rows + 1) * sizeof(int));
    for (int i = 0; i <= gpu_rows; i++)
        h_gpu_row_ptr[i] = A->row_ptr[cpu_rows + i] - (int)gpu_nnz_offset;

    /* ── Device allocations ── */
    int   *d_row_ptr, *d_col_idx;
    float *d_val, *d_x, *d_y_gpu;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (gpu_rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, gpu_nnz_count  * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,     gpu_nnz_count  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,       N              * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y_gpu,   gpu_rows       * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t t0, t1, k0, k1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));

    /* ══════════════════════════════════════════════════════════════════════
     * PHASE 1: H->D (GPU partition only, async)
     * ══════════════════════════════════════════════════════════════════════ */
    CUDA_CHECK(cudaEventRecord(t0, stream));
    if (gpu_rows > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, h_gpu_row_ptr,
                                   (gpu_rows + 1) * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_col_idx,
                                   A->col_idx + gpu_nnz_offset,
                                   gpu_nnz_count * sizeof(int),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_val,
                                   h_val + gpu_nnz_offset,
                                   gpu_nnz_count * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_x, h_x,
                                   N * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
    printf("[H->D xfer] %.3f ms\n\n", h2d_ms);

    CUDA_CHECK(cudaMemset(d_y_gpu, 0, gpu_rows * sizeof(float)));

    /* ══════════════════════════════════════════════════════════════════════
     * PHASE 2: CPU || GPU (true parallel — no sync dependency)
     *
     * Key difference from prefix sum hetero:
     * SpMV rows are independent — CPU and GPU partitions need NO
     * communication with each other. Both just write their slice of y[].
     * There is no "carry-in offset" step.
     * ══════════════════════════════════════════════════════════════════════ */
    CpuArgs cpu_args = {
        .A          = A,
        .x          = h_x,
        .y          = h_y,
        .row_start  = 0,
        .row_end    = cpu_rows,
        .elapsed_ms = 0.0
    };

    double wall_start = now_ms();

    /* Launch CPU thread */
    pthread_t cpu_thread;
    if (cpu_rows > 0)
        pthread_create(&cpu_thread, NULL, cpu_worker, &cpu_args);

    /* Launch GPU kernel simultaneously */
    float gpu_kernel_ms = 0.0f;
    if (gpu_rows > 0) {
        int warps_per_block = BLOCK_SIZE / WARP_SIZE;
        int grid = (gpu_rows + warps_per_block - 1) / warps_per_block;

        CUDA_CHECK(cudaEventRecord(k0, stream));
        spmv_warp<<<grid, BLOCK_SIZE, 0, stream>>>(
            d_row_ptr, d_col_idx, d_val, d_x, d_y_gpu, gpu_rows);
        CUDA_CHECK(cudaEventRecord(k1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, k0, k1));
    }

    /* Wait for CPU */
    if (cpu_rows > 0)
        pthread_join(cpu_thread, NULL);

    double phase2_ms = now_ms() - wall_start;

    /* ══════════════════════════════════════════════════════════════════════
     * PHASE 3: D->H (GPU output partition)
     * ══════════════════════════════════════════════════════════════════════ */
    CUDA_CHECK(cudaEventRecord(t0, stream));
    if (gpu_rows > 0)
        CUDA_CHECK(cudaMemcpyAsync(
            h_y + cpu_rows, d_y_gpu,
            gpu_rows * sizeof(float),
            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, t0, t1));

    double wall_total_ms = now_ms() - wall_start;

    /* ── Metrics ── */
    double bytes = (double)(M + 1) * sizeof(int)
                 + (double)nnz     * sizeof(int)
                 + (double)nnz     * sizeof(float)
                 + (double)nnz     * sizeof(float)
                 + (double)M       * sizeof(float);
    double bw  = (bytes / 1e9) / (wall_total_ms / 1e3);
    double gfl = (2.0 * nnz / 1e9) / (wall_total_ms / 1e3);

    double checksum = 0.0;
    for (int i = 0; i < M; i++) checksum += (double)h_y[i];

    /* ── Print ── */
    printf("SpMV Hetero | M=%d | N=%d | nnz=%ld | cpu_frac=%.2f | "
           "Time=%.2f ms | BW=%.4f GB/s | GFLOPS=%.6f | Checksum=%.6f\n",
           M, N, nnz, cpu_fraction, wall_total_ms, bw, gfl, checksum);

    printf("\n--- Breakdown ---\n");
    printf("  CPU SpMV     : %d rows | Time=%.3f ms  (%d OMP threads)\n",
           cpu_rows, cpu_args.elapsed_ms, omp_threads);
    printf("  GPU kernel   : %d rows | Time=%.3f ms  (warp reduction)\n",
           gpu_rows, gpu_kernel_ms);
    printf("  Phase 2 wall : %.3f ms  (CPU || GPU, no sync dependency)\n",
           phase2_ms);
    printf("  H->D xfer    : %.3f ms\n", h2d_ms);
    printf("  D->H xfer    : %.3f ms\n", d2h_ms);
    printf("  Total wall   : %.3f ms\n", wall_total_ms);

    /* ── Bottleneck ── */
    const char *bottleneck;
    printf("\n--- Bottleneck ---\n");
    if (cpu_rows == 0) {
        printf("  >> GPU only\n");
        bottleneck = "GPU";
    } else if (gpu_rows == 0) {
        printf("  >> CPU only\n");
        bottleneck = "CPU";
    } else if (cpu_args.elapsed_ms > gpu_kernel_ms) {
        printf("  >> CPU was bottleneck (%.3f ms vs GPU %.3f ms)\n",
               cpu_args.elapsed_ms, gpu_kernel_ms);
        bottleneck = "CPU";
    } else {
        printf("  >> GPU was bottleneck (%.3f ms vs CPU %.3f ms)\n",
               gpu_kernel_ms, cpu_args.elapsed_ms);
        bottleneck = "GPU";
    }

    /* ── Verification ── */
    printf("\n--- Verification ---\n");
    double max_rel = 0.0;
    int bad = 0;
    for (int i = 0; i < M; i++) {
        double ref  = h_ref[i];
        double got  = (double)h_y[i];
        double rel  = (fabs(ref) > 1e-9) ? fabs(ref - got) / fabs(ref)
                                          : fabs(ref - got);
        if (rel > max_rel) max_rel = rel;
        if (rel > 1e-3) bad++;
    }
    if (bad > 0)
        printf("  MISMATCH: %d elements exceed 0.1%% tolerance | "
               "max_rel=%.2e\n", bad, max_rel);
    else
        printf("  Verification PASSED (max_rel_err=%.2e)\n", max_rel);

    /* ── Save to CSV ── */
    save_spmv_csv(
        "CPU_GPU_Hetero",
        M, N, nnz, nnz_per_row,
        /*threads=*/       omp_threads,
        /*cpu_fraction=*/  (double)cpu_fraction,
        /*cpu_rows=*/      cpu_rows,
        /*gpu_rows=*/      gpu_rows,
        /*compute_ms=*/    phase2_ms,
        /*wall_total=*/    wall_total_ms,
        /*cpu_part_ms=*/   cpu_args.elapsed_ms,
        /*gpu_kernel_ms=*/ (double)gpu_kernel_ms,
        /*h2d_ms=*/        (double)h2d_ms,
        /*d2h_ms=*/        (double)d2h_ms,
        /*bandwidth_GBs=*/ bw,
        /*gflops=*/        gfl,
        /*checksum=*/      checksum,
        /*bottleneck=*/    bottleneck
    );

    printf("\n");

    /* ── Cleanup ── */
    free(h_ref);
    free(h_gpu_row_ptr);
    free_csr(A);

    CUDA_CHECK(cudaFreeHost(h_val));
    CUDA_CHECK(cudaFreeHost(h_x));
    CUDA_CHECK(cudaFreeHost(h_y));

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y_gpu));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaEventDestroy(k0)); CUDA_CHECK(cudaEventDestroy(k1));
}


/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
    if (argc < 5) {
        printf("Usage: %s <M> <N> <nnz_per_row> <cpu_fraction>\n", argv[0]);
        printf("Example: %s 131072 131072 32 0.3\n", argv[0]);
        return 1;
    }

    int   M           = atoi(argv[1]);
    int   N           = atoi(argv[2]);
    int   nnz_per_row = atoi(argv[3]);
    float cpu_fraction = (float)atof(argv[4]);

    if (cpu_fraction < 0.0f || cpu_fraction > 1.0f) {
        fprintf(stderr, "cpu_fraction must be in [0.0, 1.0]\n");
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs=%d | Mem=%.0f MB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    /* Warmup */
    {
        float *d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, BLOCK_SIZE * sizeof(float)));
        warmup_kernel<<<1, BLOCK_SIZE>>>(d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_dummy));
        printf("[Warmup] done\n\n");
    }

    run_benchmark(M, N, nnz_per_row, cpu_fraction);

    return 0;
}