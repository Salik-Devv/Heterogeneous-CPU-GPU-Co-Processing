/*
 * spmv_gpu.cu
 * Sparse Matrix-Vector Multiplication (SpMV) -- GPU, two kernels
 *
 *   y = A * x
 *   A : M x N sparse matrix in CSR format (float on GPU)
 *   x : dense input  vector (length N)
 *   y : dense output vector (length M)
 *
 * Kernels
 * -------
 *  GPU_Naive  : one thread  per row  -- simple, uncoalesced x[] reads
 *  GPU_Warp   : one warp    per row  -- threads in a warp cooperate on
 *               the dot product then reduce with __shfl_down_sync;
 *               coalesced val[]/col_idx[] reads, better occupancy
 *
 * SpMV is strongly memory-bandwidth-bound (~0.25 FLOP/byte).
 * The warp kernel improves BW utilisation at the cost of more threads.
 *
 * Usage:
 *   ./spmv_gpu <M> <N> <nnz_per_row>    (single run, both kernels)
 *   ./spmv_gpu                           (size sweep, both kernels)
 *
 * Compile (manual):
 *   nvcc -O3 -arch=sm_89 -o spmv_gpu spmv_gpu.cu -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#include "../init_csr.h"
#include "../spmv_csv.h"

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

/* ── Constants ───────────────────────────────────────────────────────────── */
#define WARP_SIZE  32
#define BLOCK_SIZE 256   /* threads per block for both kernels              */
                         /* naive  : 256 threads = 256 rows per block       */
                         /* warp   : 256 threads =   8 warps = 8 rows/block */

/* ── Default size sweep ──────────────────────────────────────────────────── */
static const int NUM_SIZES  = 3;
static const int M_SIZES[]  = { 65536, 131072, 262144 };
static const int N_SIZES[]  = { 65536, 131072, 262144 };
static const int NNZ_ROWS[] = {    32,     32,     32 };


/* ══════════════════════════════════════════════════════════════════════════
 * KERNEL 1 -- Naive: one thread per row
 *
 * Each thread independently walks its row's non-zeros.
 * Access to x[col_idx[j]] is random => uncoalesced => BW-limited.
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void spmv_naive(
    const int   * __restrict__ row_ptr,
    const int   * __restrict__ col_idx,
    const float * __restrict__ val,
    const float * __restrict__ x,
          float * __restrict__ y,
    int M)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    float sum = 0.0f;
    for (int j = row_ptr[row]; j < row_ptr[row + 1]; j++)
        sum += val[j] * x[col_idx[j]];
    y[row] = sum;
}


/* ══════════════════════════════════════════════════════════════════════════
 * KERNEL 2 -- Warp reduction: one warp (32 threads) per row
 *
 * All 32 threads in a warp share one row. Each thread handles every
 * 32nd element of that row (stride = warp size), so consecutive threads
 * read consecutive val[]/col_idx[] entries => coalesced loads.
 * The 32 partial sums are reduced with warp shuffle (__shfl_down_sync)
 * -- no shared memory needed, no __syncthreads().
 * Lane 0 writes the final result to y[row].
 *
 * Launch config:
 *   block = BLOCK_SIZE threads = BLOCK_SIZE/WARP_SIZE warps per block
 *   grid  = ceil(M / warps_per_block) blocks
 * ══════════════════════════════════════════════════════════════════════════ */
__global__ void spmv_warp(
    const int   * __restrict__ row_ptr,
    const int   * __restrict__ col_idx,
    const float * __restrict__ val,
    const float * __restrict__ x,
          float * __restrict__ y,
    int M)
{
    /* Which row does this warp own? */
    int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    int lane    = threadIdx.x % WARP_SIZE;

    if (warp_id >= M) return;

    int row_start = row_ptr[warp_id];
    int row_end   = row_ptr[warp_id + 1];

    /* Each lane accumulates every WARP_SIZE-th element */
    float sum = 0.0f;
    for (int j = row_start + lane; j < row_end; j += WARP_SIZE)
        sum += val[j] * x[col_idx[j]];

    /* Warp-level reduction via shuffle -- no shared memory */
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


/* ── CPU double-precision reference ──────────────────────────────────────── */
static void cpu_ref(const CSRMatrix *A, const float *x, double *y_ref)
{
    for (int i = 0; i < A->M; i++) {
        double sum = 0.0;
        for (int j = A->row_ptr[i]; j < A->row_ptr[i + 1]; j++)
            sum += A->val[j] * (double)x[A->col_idx[j]];
        y_ref[i] = sum;
    }
}


/* ── Verify GPU vs CPU reference ─────────────────────────────────────────── */
static void verify(const double *ref, const float *gpu, int M,
                   const char *label)
{
    double max_err = 0.0;
    int    err_idx = -1;
    for (int i = 0; i < M; i++) {
        double diff = fabs(ref[i] - (double)gpu[i]);
        if (diff > max_err) { max_err = diff; err_idx = i; }
    }
    double rel = (err_idx >= 0 && fabs(ref[err_idx]) > 1e-9)
                 ? max_err / fabs(ref[err_idx]) : 0.0;
    if (rel > 1e-3)
        printf("  [%s] MISMATCH: max_rel=%.2e at idx=%d\n",
               label, rel, err_idx);
    else
        printf("  [%s] Verification PASSED (max_rel_err=%.2e)\n",
               label, rel);
}


/* ══════════════════════════════════════════════════════════════════════════
 * time_kernel()
 * Runs ONE kernel variant: records H->D once, times kernel, records D->H.
 * Returns kernel_ms; writes h2d_ms and d2h_ms via pointers.
 * The H->D transfer is shared between both kernels (called once before).
 * ══════════════════════════════════════════════════════════════════════════ */
typedef void (*KernelFn)(const int*, const int*, const float*,
                         const float*, float*, int);

static float time_kernel(
    KernelFn        kernel_fn,
    int             grid,
    int             block,
    cudaStream_t    stream,
    const int      *d_row_ptr,
    const int      *d_col_idx,
    const float    *d_val,
    const float    *d_x,
    float          *d_y,
    float          *h_y,
    int             M,
    float          *out_d2h_ms)
{
    cudaEvent_t k0, k1, d0, d1;
    CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));
    CUDA_CHECK(cudaEventCreate(&d0)); CUDA_CHECK(cudaEventCreate(&d1));

    CUDA_CHECK(cudaMemset(d_y, 0, M * sizeof(float)));

    /* Kernel */
    CUDA_CHECK(cudaEventRecord(k0, stream));
    kernel_fn<<<grid, block, 0, stream>>>(
        d_row_ptr, d_col_idx, d_val, d_x, d_y, M);
    CUDA_CHECK(cudaEventRecord(k1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k0, k1));

    /* D->H */
    CUDA_CHECK(cudaEventRecord(d0, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_y, d_y, M * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(d1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    CUDA_CHECK(cudaEventElapsedTime(out_d2h_ms, d0, d1));

    CUDA_CHECK(cudaEventDestroy(k0)); CUDA_CHECK(cudaEventDestroy(k1));
    CUDA_CHECK(cudaEventDestroy(d0)); CUDA_CHECK(cudaEventDestroy(d1));

    return kernel_ms;
}


/* ══════════════════════════════════════════════════════════════════════════
 * run_benchmark() -- runs BOTH kernels for one problem size
 * ══════════════════════════════════════════════════════════════════════════ */
static void run_benchmark(int M, int N, int nnz_per_row)
{
    long nnz = (long)M * nnz_per_row;

    printf("=== GPU SpMV | M=%d | N=%d | nnz=%ld | nnz/row=%d ===\n",
           M, N, nnz, nnz_per_row);

    /* ── Build host data ── */
    CSRMatrix *A   = init_csr(M, N, nnz_per_row, /*seed=*/42);
    float  *h_val  = (float*)malloc(nnz * sizeof(float));
    float  *h_x    = (float*)malloc(N   * sizeof(float));
    float  *h_y    = (float*)malloc(M   * sizeof(float));
    double *h_ref  = (double*)malloc(M  * sizeof(double));

    if (!h_val || !h_x || !h_y || !h_ref) {
        fprintf(stderr, "Host malloc failed\n"); exit(EXIT_FAILURE);
    }

    srand(0);
    for (long i = 0; i < nnz; i++) h_val[i] = (float)A->val[i];
    for (int  i = 0; i < N;   i++)
        h_x[i] = (float)((double)rand() / RAND_MAX + 1e-6);

    cpu_ref(A, h_x, h_ref);

    /* Memory traffic (same for both kernels -- problem doesn't change) */
    double bytes = (double)(M + 1) * sizeof(int)
                 + (double)nnz     * sizeof(int)
                 + (double)nnz     * sizeof(float)
                 + (double)nnz     * sizeof(float)
                 + (double)M       * sizeof(float);
    double flops = 2.0 * nnz;

    /* ── Device allocations ── */
    int   *d_row_ptr, *d_col_idx;
    float *d_val, *d_x, *d_y;
    CUDA_CHECK(cudaMalloc(&d_row_ptr, (M + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, nnz     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_val,     nnz     * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x,       N       * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y,       M       * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    /* ── H->D (timed, shared by both kernels) ── */
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_row_ptr, A->row_ptr,
                               (M + 1) * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_col_idx, A->col_idx,
                               nnz * sizeof(int),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_val, h_val,
                               nnz * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_x, h_x,
                               N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    /* ════════════════════════════════════════════════════════════════════
     * Benchmark 1 -- Naive
     * ════════════════════════════════════════════════════════════════════ */
    {
        int grid_naive = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;

        float d2h_ms    = 0.0f;
        float kernel_ms = time_kernel(
            spmv_naive, grid_naive, BLOCK_SIZE, stream,
            d_row_ptr, d_col_idx, d_val, d_x, d_y,
            h_y, M, &d2h_ms);

        double bw         = (bytes / 1e9) / (kernel_ms / 1e3);
        double gfl        = (flops / 1e9) / (kernel_ms / 1e3);
        double wall_total = (double)h2d_ms + (double)kernel_ms
                          + (double)d2h_ms;

        double checksum = 0.0;
        for (int i = 0; i < M; i++) checksum += (double)h_y[i];

        printf("GPU_Naive  | Kernel=%.3f ms | BW=%.2f GB/s | "
               "GFLOPS=%.6f | Checksum=%.6f\n",
               kernel_ms, bw, gfl, checksum);
        printf("  H->D=%.3f ms | D->H=%.3f ms | Total=%.3f ms\n",
               h2d_ms, d2h_ms, wall_total);
        verify(h_ref, h_y, M, "GPU_Naive");

        save_spmv_csv(
            "GPU_Naive",
            M, N, nnz, nnz_per_row,
            /*threads=*/       1,
            /*cpu_fraction=*/  0.0,
            /*cpu_rows=*/      0,
            /*gpu_rows=*/      M,
            /*compute_ms=*/    (double)kernel_ms,
            /*wall_total=*/    wall_total,
            /*cpu_part_ms=*/  -1.0,
            /*gpu_kernel_ms=*/ (double)kernel_ms,
            /*h2d_ms=*/        (double)h2d_ms,
            /*d2h_ms=*/        (double)d2h_ms,
            /*bandwidth_GBs=*/ bw,
            /*gflops=*/        gfl,
            /*checksum=*/      checksum,
            /*bottleneck=*/    "Memory"
        );
    }

    /* ════════════════════════════════════════════════════════════════════
     * Benchmark 2 -- Warp reduction
     *
     * One warp (32 threads) per row:
     *   warps_per_block = BLOCK_SIZE / WARP_SIZE = 8
     *   grid = ceil(M / warps_per_block)
     * ════════════════════════════════════════════════════════════════════ */
    {
        int warps_per_block = BLOCK_SIZE / WARP_SIZE;
        int grid_warp       = (M + warps_per_block - 1) / warps_per_block;

        float d2h_ms    = 0.0f;
        float kernel_ms = time_kernel(
            spmv_warp, grid_warp, BLOCK_SIZE, stream,
            d_row_ptr, d_col_idx, d_val, d_x, d_y,
            h_y, M, &d2h_ms);

        double bw         = (bytes / 1e9) / (kernel_ms / 1e3);
        double gfl        = (flops / 1e9) / (kernel_ms / 1e3);
        double wall_total = (double)h2d_ms + (double)kernel_ms
                          + (double)d2h_ms;

        double checksum = 0.0;
        for (int i = 0; i < M; i++) checksum += (double)h_y[i];

        printf("GPU_Warp   | Kernel=%.3f ms | BW=%.2f GB/s | "
               "GFLOPS=%.6f | Checksum=%.6f\n",
               kernel_ms, bw, gfl, checksum);
        printf("  H->D=%.3f ms | D->H=%.3f ms | Total=%.3f ms\n",
               h2d_ms, d2h_ms, wall_total);
        verify(h_ref, h_y, M, "GPU_Warp");

        save_spmv_csv(
            "GPU_Warp",
            M, N, nnz, nnz_per_row,
            /*threads=*/       1,
            /*cpu_fraction=*/  0.0,
            /*cpu_rows=*/      0,
            /*gpu_rows=*/      M,
            /*compute_ms=*/    (double)kernel_ms,
            /*wall_total=*/    wall_total,
            /*cpu_part_ms=*/  -1.0,
            /*gpu_kernel_ms=*/ (double)kernel_ms,
            /*h2d_ms=*/        (double)h2d_ms,
            /*d2h_ms=*/        (double)d2h_ms,
            /*bandwidth_GBs=*/ bw,
            /*gflops=*/        gfl,
            /*checksum=*/      checksum,
            /*bottleneck=*/    "Memory"
        );
    }

    printf("\n");

    /* ── Cleanup ── */
    free(h_val); free(h_x); free(h_y); free(h_ref);
    free_csr(A);
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_row_ptr));
    CUDA_CHECK(cudaFree(d_col_idx));
    CUDA_CHECK(cudaFree(d_val));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}


/* ── Main ────────────────────────────────────────────────────────────────── */
int main(int argc, char *argv[])
{
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

    if (argc >= 4) {
        run_benchmark(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]));
    } else {
        printf("=== GPU SpMV -- Size Sweep ===\n\n");
        for (int i = 0; i < NUM_SIZES; i++)
            run_benchmark(M_SIZES[i], N_SIZES[i], NNZ_ROWS[i]);
    }

    return 0;
}