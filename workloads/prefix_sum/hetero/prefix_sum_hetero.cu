// prefix_sum_hetero.cu
// CPU + GPU co-processing Prefix Sum
//
// Split strategy:
//   CPU handles [0,         cpu_elems)  — float32, OMP parallel scan
//   GPU handles [cpu_elems, N)          — float32
//
// Timeline:
//   1. H→D transfer (GPU partition only, pinned async)
//   2. CPU scans its partition || GPU scans its partition (Pass 1)
//   3. SYNC POINT: CPU total known → baked into GPU block offsets
//   4. GPU Pass 2+3: finalize with global offset applied
//   5. D→H transfer
//
// Usage: ./prefix_sum_hetero <N> <cpu_fraction 0.0-1.0>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../init_array.h"
#include "prefix_part.h"
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
// Kernel 1: per-block exclusive scan
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

    smem[tid]              = (idx0 < N) ? in[idx0] : 0.0f;
    smem[tid + BLOCK_SIZE] = (idx1 < N) ? in[idx1] : 0.0f;
    __syncthreads();

    for (int stride = 1; stride <= BLOCK_SIZE; stride <<= 1) {
        int index = (tid + 1) * stride * 2 - 1;
        if (index < BLOCK_SIZE * 2)
            smem[index] += smem[index - stride];
        __syncthreads();
    }

    if (tid == 0) {
        block_sums[bid]          = smem[BLOCK_SIZE * 2 - 1];
        smem[BLOCK_SIZE * 2 - 1] = 0.0f;
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
// Kernel 2: finalize
// ─────────────────────────────────────────────
__global__ void add_block_offsets(
    const float *in, float *out,
    const float *block_sums_scanned,
    long N)
{
    int  tid  = threadIdx.x;
    int  bid  = blockIdx.x;
    long base = (long)bid * BLOCK_SIZE * 2;
    long idx0 = base + tid;
    long idx1 = base + tid + BLOCK_SIZE;

    float offset = block_sums_scanned[bid];

    if (idx0 < N) out[idx0] = out[idx0] + offset + in[idx0];
    if (idx1 < N) out[idx1] = out[idx1] + offset + in[idx1];
}


// ─────────────────────────────────────────────
// Warmup
// ─────────────────────────────────────────────
__global__ void warmup(float *buf) { buf[threadIdx.x] = 0.0f; }


// ─────────────────────────────────────────────
// CPU thread args
// ─────────────────────────────────────────────
typedef struct {
    const float *in;
    float       *out;
    long         start;
    long         end;
    float        cpu_total;
    double       elapsed_ms;
} CpuArgs;

void *cpu_worker(void *arg) {
    CpuArgs *a = (CpuArgs*)arg;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    prefix_sum_cpu_part(a->in, a->out, a->start, a->end, 0.0f);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    a->elapsed_ms = (t1.tv_sec  - t0.tv_sec)  * 1e3 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
    a->cpu_total  = (a->end > a->start) ? a->out[a->end - 1] : 0.0f;
    return NULL;
}


// ─────────────────────────────────────────────
// CPU exclusive scan of block sums (Pass 2)
// ─────────────────────────────────────────────
void cpu_exclusive_scan_blocks(const float *in, float *out, int N) {
    out[0] = 0.0f;
    for (int i = 1; i < N; i++)
        out[i] = out[i-1] + in[i-1];
}


// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
double checksum_float(const float *arr, long N) {
    double sum = 0.0;
    for (long i = 0; i < N; i++) sum += (double)arr[i];
    return sum;
}

static inline double now_ms() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}


// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <N> <cpu_fraction>\n", argv[0]);
        return 1;
    }

    long  N            = atol(argv[1]);
    float cpu_fraction = atof(argv[2]);
    long  cpu_elems    = (long)(N * cpu_fraction);
    long  gpu_elems    = N - cpu_elems;

    // Read OMP thread count before any parallel region
    int omp_threads = omp_get_max_threads();

    printf("=== Prefix Sum Hetero | N=%ld | "
           "CPU=%.0f%% (%ld elems, %d OMP threads) | GPU=%.0f%% (%ld elems) ===\n\n",
           N, cpu_fraction*100, cpu_elems, omp_threads,
           (1.0f-cpu_fraction)*100, gpu_elems);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | SMs=%d\n\n", prop.name, prop.multiProcessorCount);

    {
        float *d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, BLOCK_SIZE * sizeof(float)));
        warmup<<<1, BLOCK_SIZE>>>(d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_dummy));
        printf("[Warmup] done\n\n");
    }

    int num_blocks = (gpu_elems > 0)
        ? (int)((gpu_elems + (long)BLOCK_SIZE*2-1) / ((long)BLOCK_SIZE*2))
        : 1;

    size_t bytes_all = (size_t)N         * sizeof(float);
    size_t bytes_gpu = (size_t)gpu_elems * sizeof(float);

    double *h_in_d;
    float  *h_in_f, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_in_d, (size_t)N * sizeof(double)));
    CUDA_CHECK(cudaMallocHost(&h_in_f, bytes_all));
    CUDA_CHECK(cudaMallocHost(&h_out,  bytes_all));
    memset(h_out, 0, bytes_all);

    init_array_random(h_in_d, N, ARRAY_SEED);
    for (long i = 0; i < N; i++)
        h_in_f[i] = (float)h_in_d[i];

    double *h_ref = (double*)malloc((size_t)N * sizeof(double));
    h_ref[0] = h_in_d[0];
    for (long i = 1; i < N; i++)
        h_ref[i] = h_ref[i-1] + h_in_d[i];

    float *d_in, *d_out, *d_bs, *d_bs_scan;
    CUDA_CHECK(cudaMalloc(&d_in,      bytes_gpu));
    CUDA_CHECK(cudaMalloc(&d_out,     bytes_gpu));
    CUDA_CHECK(cudaMalloc(&d_bs,      num_blocks * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bs_scan, num_blocks * sizeof(float)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    cudaEvent_t t0, t1, k0, k1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventCreate(&k0)); CUDA_CHECK(cudaEventCreate(&k1));

    // ── PHASE 1: H→D ──
    CUDA_CHECK(cudaEventRecord(t0, stream));
    if (gpu_elems > 0)
        CUDA_CHECK(cudaMemcpyAsync(
            d_in, h_in_f + cpu_elems, bytes_gpu,
            cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
    printf("[H→D xfer] %.3f ms\n\n", h2d_ms);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes_gpu));
    CUDA_CHECK(cudaMemset(d_bs,  0, num_blocks * sizeof(float)));

    // ── PHASE 2: CPU (OMP parallel) || GPU simultaneous ──
    CpuArgs cpu_args = {
        .in         = h_in_f,
        .out        = h_out,
        .start      = 0,
        .end        = cpu_elems,
        .cpu_total  = 0.0f,
        .elapsed_ms = 0.0
    };

    double wall_start = now_ms();

    pthread_t cpu_thread;
    if (cpu_elems > 0)
        pthread_create(&cpu_thread, NULL, cpu_worker, &cpu_args);

    float gpu_pass1_ms = 0.0f;
    if (gpu_elems > 0) {
        CUDA_CHECK(cudaEventRecord(k0, stream));
        scan_blocks<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_in, d_out, d_bs, gpu_elems);
        CUDA_CHECK(cudaEventRecord(k1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&gpu_pass1_ms, k0, k1));
    }

    if (cpu_elems > 0)
        pthread_join(cpu_thread, NULL);

    double phase2_ms = now_ms() - wall_start;

    // ── SYNC POINT ──
    float cpu_total = (cpu_elems > 0) ? cpu_args.cpu_total : 0.0f;

    double sync_start = now_ms();

    float  *h_bs      = (float*)malloc(num_blocks * sizeof(float));
    float  *h_bs_scan = (float*)malloc(num_blocks * sizeof(float));
    float   gpu_pass23_ms = 0.0f;

    if (gpu_elems > 0) {
        CUDA_CHECK(cudaMemcpy(h_bs, d_bs,
                              num_blocks * sizeof(float),
                              cudaMemcpyDeviceToHost));
        cpu_exclusive_scan_blocks(h_bs, h_bs_scan, num_blocks);

        for (int i = 0; i < num_blocks; i++)
            h_bs_scan[i] += cpu_total;

        CUDA_CHECK(cudaMemcpy(d_bs_scan, h_bs_scan,
                              num_blocks * sizeof(float),
                              cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaEventRecord(k0, stream));
        add_block_offsets<<<num_blocks, BLOCK_SIZE, 0, stream>>>(
            d_in, d_out, d_bs_scan, gpu_elems);
        CUDA_CHECK(cudaEventRecord(k1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&gpu_pass23_ms, k0, k1));
    }

    double sync_ms = now_ms() - sync_start;

    // ── PHASE 4: D→H ──
    CUDA_CHECK(cudaEventRecord(t0, stream));
    if (gpu_elems > 0)
        CUDA_CHECK(cudaMemcpyAsync(
            h_out + cpu_elems, d_out, bytes_gpu,
            cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, t0, t1));

    double wall_total_ms = now_ms() - wall_start;
    float  gpu_total_ms  = gpu_pass1_ms + gpu_pass23_ms;

    double checksum = checksum_float(h_out, N);
    double gb_s     = (2.0 * N * sizeof(float)) / (wall_total_ms * 1e6);

    printf("Prefix Sum (Hetero) | N=%-10ld | Time=%.2f ms | "
           "BW=%6.2f GB/s | Checksum=%.6f\n",
           N, wall_total_ms, gb_s, checksum);

    printf("\n--- Breakdown ---\n");
    printf("  CPU scan     : %ld elems | Time=%.3f ms  (%d OMP threads)\n",
           cpu_elems, cpu_args.elapsed_ms, omp_threads);
    printf("  GPU Pass 1   : %ld elems | Time=%.3f ms  (local scan)\n",
           gpu_elems, gpu_pass1_ms);
    printf("  GPU Pass 2+3 : finalize  | Time=%.3f ms  (add offsets)\n",
           gpu_pass23_ms);
    printf("  GPU total    : %.3f ms\n", gpu_total_ms);
    printf("  Phase 2 wall : %.3f ms  (CPU || GPU pass1)\n", phase2_ms);
    printf("  Sync overhead: %.3f ms  (D→H block sums, scan, H→D, pass2+3)\n",
           sync_ms);
    printf("  H→D xfer     : %.3f ms\n", h2d_ms);
    printf("  D→H xfer     : %.3f ms\n", d2h_ms);
    printf("  Total wall   : %.3f ms\n", wall_total_ms);

    // ── Bottleneck ──
    const char *bottleneck;
    printf("\n--- Bottleneck ---\n");
    if (cpu_args.elapsed_ms > gpu_pass1_ms) {
        printf("  >> CPU was bottleneck (%.3f ms vs GPU %.3f ms)\n",
               cpu_args.elapsed_ms, gpu_pass1_ms);
        bottleneck = "CPU";
    } else {
        printf("  >> GPU was bottleneck (%.3f ms vs CPU %.3f ms)\n",
               gpu_pass1_ms, cpu_args.elapsed_ms);
        bottleneck = "GPU";
    }
    printf("  >> Sync overhead: %.3f ms — serial dependency cost\n", sync_ms);

    if (N <= (1 << 24)) {
        printf("\n--- Verification ---\n");
        double tol = 5e-3;
        double max_rel = 0.0;
        long   err_idx = -1;
        int    bad = 0;
        for (long i = 0; i < N; i++) {
            double ref = h_ref[i];
            double got = (double)h_out[i];
            double rel = (fabs(ref) > 1.0)
                         ? fabs(ref - got) / fabs(ref)
                         : fabs(ref - got);
            if (rel > max_rel) { max_rel = rel; err_idx = i; }
            if (rel > tol) bad++;
        }
        if (bad > 0)
            printf("  MISMATCH: %d elements exceed %.1f%% tolerance | "
                   "worst rel=%.2e at idx=%ld\n"
                   "  ref=%.2f got=%.2f\n",
                   bad, tol*100, max_rel, err_idx,
                   h_ref[err_idx], (double)h_out[err_idx]);
        else
            printf("  Verification PASSED ✓  "
                   "(float32, max_rel_err=%.2e, N=%ld)\n", max_rel, N);
        printf("  Note: GPU partition uses float32 — "
               "relative error up to ~0.5%% is expected for N=%ld\n", N);
    }

    // ── Save results to CSV ──
    save_prefix_csv(
        "CPU_GPU_Hetero",
        N,
        /*threads=*/          omp_threads,      // actual OMP thread count
        /*cpu_fraction=*/     cpu_fraction,
        /*cpu_elems=*/        cpu_elems,
        /*gpu_elems=*/        gpu_elems,
        /*compute_ms=*/       phase2_ms,
        /*wall_total_ms=*/    wall_total_ms,
        /*cpu_part_ms=*/      cpu_args.elapsed_ms,
        /*gpu_kernel_ms=*/    gpu_total_ms,
        /*sync_overhead_ms=*/ sync_ms,
        /*h2d_ms=*/           h2d_ms,
        /*d2h_ms=*/           d2h_ms,
        gb_s, checksum,
        bottleneck
    );

    free(h_bs); free(h_bs_scan); free(h_ref);
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_bs); cudaFree(d_bs_scan);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    cudaEventDestroy(k0); cudaEventDestroy(k1);
    cudaFreeHost(h_in_d);
    cudaFreeHost(h_in_f);
    cudaFreeHost(h_out);

    return 0;
}