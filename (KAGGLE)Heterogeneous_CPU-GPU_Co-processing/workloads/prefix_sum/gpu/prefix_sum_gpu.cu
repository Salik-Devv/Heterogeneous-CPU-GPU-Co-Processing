// prefix_sum_gpu.cu
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "../init_array.h"
#include "../prefix_sum_csv.h"

#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define BLOCK_SIZE 512

// ─────────────────────────────────────────────
// Kernel 1: Per-block exclusive scan
// ─────────────────────────────────────────────
__global__ void scan_blocks(
    const float *in, float *out,
    float *block_sums, long N)
{
    __shared__ float smem[BLOCK_SIZE * 2];

    int  tid  = threadIdx.x;
    int  bid  = blockIdx.x;
    long base = (long)bid * BLOCK_SIZE * 2;
    long idx0 = base + tid;
    long idx1 = base + tid + BLOCK_SIZE;

    float v0 = (idx0 < N) ? in[idx0] : 0.0f;
    float v1 = (idx1 < N) ? in[idx1] : 0.0f;
    smem[tid]              = v0;
    smem[tid + BLOCK_SIZE] = v1;
    __syncthreads();

    for (int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE * 2)
            smem[index] += smem[index - stride];
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[bid]           = smem[BLOCK_SIZE * 2 - 1];
        smem[BLOCK_SIZE * 2 - 1]  = 0.0f;
    }
    __syncthreads();

    for (int stride = BLOCK_SIZE; stride >= 1; stride >>= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE * 2) {
            float tmp            = smem[index - stride];
            smem[index - stride] = smem[index];
            smem[index]         += tmp;
        }
        __syncthreads();
    }

    if (idx0 < N) out[idx0] = smem[tid];
    if (idx1 < N) out[idx1] = smem[tid + BLOCK_SIZE];
}


// ─────────────────────────────────────────────
// Kernel 2: Add scanned block offsets + input
// ─────────────────────────────────────────────
__global__ void add_block_offsets(
    const float *in, float *out,
    const float *block_sums_scanned,
    long N)
{
    int  tid    = threadIdx.x;
    int  bid    = blockIdx.x;
    long base   = (long)bid * BLOCK_SIZE * 2;
    long idx0   = base + tid;
    long idx1   = base + tid + BLOCK_SIZE;

    float offset = block_sums_scanned[bid];

    if (idx0 < N) out[idx0] = out[idx0] + offset + in[idx0];
    if (idx1 < N) out[idx1] = out[idx1] + offset + in[idx1];
}


// ─────────────────────────────────────────────
// CPU scan of block sums (small array)
// ─────────────────────────────────────────────
void cpu_exclusive_scan(const float *in, float *out, int N) {
    out[0] = 0.0f;
    for (int i = 1; i < N; i++)
        out[i] = out[i-1] + in[i-1];
}


// ─────────────────────────────────────────────
// Warmup
// ─────────────────────────────────────────────
__global__ void warmup(float *buf) { buf[threadIdx.x] = 0.0f; }


// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
double checksum_float(const float *arr, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += (double)arr[i];
    return sum;
}

void verify_against_cpu(const double *cpu_ref,
                         const float  *gpu_out, long N) {
    double max_err = 0.0;
    long   err_idx = -1;
    for (long i = 0; i < N; i++) {
        double diff = fabs(cpu_ref[i] - (double)gpu_out[i]);
        if (diff > max_err) { max_err = diff; err_idx = i; }
    }
    double rel = (err_idx >= 0 && fabs(cpu_ref[err_idx]) > 1e-9)
                 ? max_err / fabs(cpu_ref[err_idx]) : 0.0;
    if (rel > 1e-3)
        printf("  MISMATCH: max_err=%.2f at idx=%ld "
               "| cpu=%.2f gpu=%.2f | rel=%.2e\n",
               max_err, err_idx,
               cpu_ref[err_idx], (double)gpu_out[err_idx], rel);
    else
        printf("  Verification PASSED ✓ (max_rel_err=%.2e)\n", rel);
}


// ─────────────────────────────────────────────
// Run benchmark
// ─────────────────────────────────────────────
void run_benchmark(long N) {
    size_t bytes_d = N * sizeof(double);
    size_t bytes_f = N * sizeof(float);

    int num_blocks = (int)((N + (long)BLOCK_SIZE * 2 - 1)
                           / ((long)BLOCK_SIZE * 2));

    double *h_in_d;
    float  *h_in_f, *h_out_f;
    CUDA_CHECK(cudaMallocHost(&h_in_d,  bytes_d));
    CUDA_CHECK(cudaMallocHost(&h_in_f,  bytes_f));
    CUDA_CHECK(cudaMallocHost(&h_out_f, bytes_f));

    init_array_random(h_in_d, N, ARRAY_SEED);
    for (long i = 0; i < N; i++)
        h_in_f[i] = (float)h_in_d[i];

    double *h_ref = (double*)malloc(bytes_d);
    h_ref[0] = h_in_d[0];
    for (long i = 1; i < N; i++)
        h_ref[i] = h_ref[i-1] + h_in_d[i];

    float *d_in, *d_out, *d_block_sums, *d_block_sums_scanned;
    CUDA_CHECK(cudaMalloc(&d_in,                 bytes_f));
    CUDA_CHECK(cudaMalloc(&d_out,                bytes_f));
    CUDA_CHECK(cudaMalloc(&d_block_sums,         num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_block_sums_scanned, num_blocks * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t t0, t1, k0, k1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));

    // ── H→D ──
    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_in_f, bytes_f,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));

    CUDA_CHECK(cudaMemset(d_out,                0, bytes_f));
    CUDA_CHECK(cudaMemset(d_block_sums,         0, num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_block_sums_scanned, 0, num_blocks * sizeof(float)));

    // ── Timed scan ──
    CUDA_CHECK(cudaEventRecord(k0, stream));

    scan_blocks<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, d_block_sums, N);
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float *h_block_sums         = (float*)malloc(num_blocks * sizeof(float));
    float *h_block_sums_scanned = (float*)malloc(num_blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums,
                          num_blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));
    cpu_exclusive_scan(h_block_sums, h_block_sums_scanned, num_blocks);
    CUDA_CHECK(cudaMemcpy(d_block_sums_scanned, h_block_sums_scanned,
                          num_blocks * sizeof(float),
                          cudaMemcpyHostToDevice));

    add_block_offsets<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
        d_in, d_out, d_block_sums_scanned, N);

    CUDA_CHECK(cudaEventRecord(k1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    float kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, k0, k1));

    // ── D→H ──
    CUDA_CHECK(cudaEventRecord(t0, stream));
    CUDA_CHECK(cudaMemcpyAsync(h_out_f, d_out, bytes_f,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, t0, t1));

    double checksum = checksum_float(h_out_f, N);
    double gb_s     = (2.0 * N * sizeof(float)) / (kernel_ms * 1e6);
    double wall_total = h2d_ms + kernel_ms + d2h_ms;

    printf("GPU Prefix Sum          | N=%-10ld | Time=%8.3f ms | "
           "BW=%6.2f GB/s | Checksum=%.6f\n",
           N, kernel_ms, gb_s, checksum);
    printf("  H→D=%.3f ms | D→H=%.3f ms | Total=%.3f ms | Passes=3\n",
           h2d_ms, d2h_ms, wall_total);

    if (N <= (1 << 24))
        verify_against_cpu(h_ref, h_out_f, N);

    // ── Save results to CSV ──
    save_prefix_csv(
        "GPU",
        N,
        /*threads=*/          1,
        /*cpu_fraction=*/     0.0,
        /*cpu_elems=*/        0,
        /*gpu_elems=*/        N,
        /*compute_ms=*/       kernel_ms,
        /*wall_total_ms=*/    wall_total,
        /*cpu_part_ms=*/     -1.0,   // N/A
        /*gpu_kernel_ms=*/    kernel_ms,
        /*sync_overhead_ms=*/-1.0,   // N/A
        /*h2d_ms=*/           h2d_ms,
        /*d2h_ms=*/           d2h_ms,
        gb_s, checksum,
        /*bottleneck=*/       "GPU"
    );

    free(h_block_sums); free(h_block_sums_scanned); free(h_ref);
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_block_sums); cudaFree(d_block_sums_scanned);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaFreeHost(h_in_d); cudaFreeHost(h_in_f); cudaFreeHost(h_out_f);
}


// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs=%d | Mem=%.0f MB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    {
        float *d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, BLOCK_SIZE * sizeof(float)));
        warmup<<<1, BLOCK_SIZE>>>(d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_dummy));
        printf("[Warmup] done\n\n");
    }

    if (argc >= 2) {
        long N = atol(argv[1]);
        printf("=== GPU Prefix Sum | N=%ld ===\n\n", N);
        run_benchmark(N);
    } else {
        printf("=== GPU Prefix Sum — Size Sweep ===\n\n");
        for (int i = 0; i < NUM_PREFIX_SIZES; i++)
            run_benchmark(PREFIX_SIZES[i]);
    }

    return 0;
}
