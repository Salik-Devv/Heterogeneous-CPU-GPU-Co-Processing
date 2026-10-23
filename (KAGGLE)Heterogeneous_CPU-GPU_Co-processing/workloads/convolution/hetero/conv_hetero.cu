// conv_hetero.cu
// CPU + GPU co-processing Sobel — optimized & fair measurement
//
// Improvements over v1:
//   - Pinned host memory  → faster PCIe transfers
//   - Async H→D transfer via CUDA stream → overlaps with CPU compute
//   - Warm-up kernel run  → eliminates CUDA context init from timing
//   - Fair wall clock     → starts AFTER data is on device
//   - Same tiled kernel   → identical to conv_gpu.cu
//   - Bottleneck report   → shows which side was the limiter
//   - OMP parallel CPU    → thread count logged and saved to CSV
//
// Usage: ./conv_hetero [input.png] [cpu_fraction 0.0-1.0]

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <pthread.h>
#include <omp.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "conv_part.h"
#include "../conv_csv.h"

#define TILE_W 32
#define TILE_H 32

// ─────────────────────────────────────────────
// Error macro
// ─────────────────────────────────────────────
#define CUDA_CHECK(call)                                                   \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)


// ─────────────────────────────────────────────
// Sobel in constant memory
// ─────────────────────────────────────────────
__constant__ float c_Gx[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};
__constant__ float c_Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


// ─────────────────────────────────────────────
// Tiled Sobel kernel
// ─────────────────────────────────────────────
__global__ void sobel_tiled(
    const float *in_full,
    float       *out_full,
    int          width,
    int          height,
    int          row_offset,
    int          gpu_rows)
{
    __shared__ float smem[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int local_y  = blockIdx.y * TILE_H + ty;
    int global_y = row_offset + local_y;
    int global_x = blockIdx.x * TILE_W + tx;

    int smem_x = tx + 1;
    int smem_y = ty + 1;

    smem[smem_y][smem_x] = (global_x < width && global_y < height)
                            ? in_full[global_y * width + global_x] : 0.0f;

    if (tx == 0)
        smem[smem_y][0] = (global_x-1 >= 0 && global_y < height)
                          ? in_full[global_y * width + (global_x-1)] : 0.0f;

    if (tx == TILE_W-1)
        smem[smem_y][TILE_W+1] = (global_x+1 < width && global_y < height)
                                  ? in_full[global_y * width + (global_x+1)] : 0.0f;

    if (ty == 0)
        smem[0][smem_x] = (global_y-1 >= 0 && global_x < width)
                          ? in_full[(global_y-1) * width + global_x] : 0.0f;

    if (ty == TILE_H-1)
        smem[TILE_H+1][smem_x] = (global_y+1 < height && global_x < width)
                                  ? in_full[(global_y+1) * width + global_x] : 0.0f;

    if (tx == 0 && ty == 0)
        smem[0][0] = (global_x-1>=0 && global_y-1>=0)
                     ? in_full[(global_y-1)*width+(global_x-1)] : 0.0f;

    if (tx == TILE_W-1 && ty == 0)
        smem[0][TILE_W+1] = (global_x+1<width && global_y-1>=0)
                             ? in_full[(global_y-1)*width+(global_x+1)] : 0.0f;

    if (tx == 0 && ty == TILE_H-1)
        smem[TILE_H+1][0] = (global_x-1>=0 && global_y+1<height)
                             ? in_full[(global_y+1)*width+(global_x-1)] : 0.0f;

    if (tx == TILE_W-1 && ty == TILE_H-1)
        smem[TILE_H+1][TILE_W+1] = (global_x+1<width && global_y+1<height)
                                    ? in_full[(global_y+1)*width+(global_x+1)] : 0.0f;

    __syncthreads();

    if (global_x < 1 || global_x >= width-1)  return;
    if (global_y < 1 || global_y >= height-1) return;
    if (local_y  >= gpu_rows)                  return;

    float gx = 0.0f, gy = 0.0f;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            float px = smem[smem_y+ky][smem_x+kx];
            gx += c_Gx[ky+1][kx+1] * px;
            gy += c_Gy[ky+1][kx+1] * px;
        }

    out_full[global_y * width + global_x] = sqrtf(gx*gx + gy*gy);
}


// ─────────────────────────────────────────────
// Warmup kernel
// ─────────────────────────────────────────────
__global__ void warmup_kernel(float *buf) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    buf[i] = 0.0f;
}


// ─────────────────────────────────────────────
// CPU thread
// ─────────────────────────────────────────────
typedef struct {
    const float *in;
    float       *out;
    int          width;
    int          height;
    int          row_start;
    int          row_end;
    double       elapsed_ms;
} CpuArgs;

void *cpu_worker(void *arg) {
    CpuArgs *a = (CpuArgs*)arg;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    sobel_cpu_part(a->in, a->out,
                   a->width, a->height,
                   a->row_start, a->row_end);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    a->elapsed_ms = (t1.tv_sec  - t0.tv_sec)  * 1e3 +
                    (t1.tv_nsec - t0.tv_nsec) / 1e6;
    return NULL;
}


// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────
float* rgb_to_gray_pinned(const unsigned char *rgb, int width, int height) {
    float *gray;
    CUDA_CHECK(cudaMallocHost(&gray, (size_t)width * height * sizeof(float)));
    for (int i = 0; i < width * height; i++)
        gray[i] = 0.299f*rgb[i*3+0] + 0.587f*rgb[i*3+1] + 0.114f*rgb[i*3+2];
    return gray;
}

unsigned char* normalize_to_uchar(const float *buf, int n) {
    float max_val = 0.0f;
    for (int i = 0; i < n; i++)
        if (buf[i] > max_val) max_val = buf[i];
    unsigned char *out = (unsigned char*)malloc(n);
    float scale = (max_val > 0.0f) ? 255.0f / max_val : 1.0f;
    for (int i = 0; i < n; i++)
        out[i] = (unsigned char)(buf[i] * scale);
    return out;
}

double compute_checksum(const float *buf, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += buf[i];
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
    const char *input_path = (argc >= 2) ? argv[1] : "city-view.png";
    float cpu_fraction     = (argc >= 3) ? atof(argv[2]) : 0.3f;

    // Read OMP thread count before any parallel region
    int omp_threads = sobel_cpu_num_threads();

    // ── Load image ──
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load '%s'\n", input_path);
        return 1;
    }

    int cpu_rows = (int)(height * cpu_fraction);
    int gpu_rows = height - cpu_rows;

    if (gpu_rows < 0) gpu_rows = 0;
    if (cpu_rows > height) cpu_rows = height;

    printf("=== Sobel Hetero | %dx%d | CPU=%.0f%% (%d rows, %d OMP threads) GPU=%.0f%% (%d rows) ===\n\n",
           width, height,
           cpu_fraction*100, cpu_rows, omp_threads,
           (1.0f-cpu_fraction)*100, gpu_rows);

    // ── GPU info ──
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU : %s | SMs=%d | Mem=%.0f MB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    size_t n     = (size_t)width * height;
    size_t bytes = n * sizeof(float);

    // ── Pinned host allocation ──
    float *h_gray, *h_out;
    CUDA_CHECK(cudaMallocHost(&h_out,  bytes));
    memset(h_out, 0, bytes);

    h_gray = rgb_to_gray_pinned(img, width, height);
    stbi_image_free(img);

    // ── Device allocation ──
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // ── CUDA stream ──
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // ── Warmup ──
    {
        float *d_dummy;
        CUDA_CHECK(cudaMalloc(&d_dummy, 256 * sizeof(float)));
        warmup_kernel<<<1, 256>>>(d_dummy);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_dummy));
        printf("[Warmup] done\n\n");
    }

    // ── PHASE 1: Async H→D transfer ──
    cudaEvent_t xfer_t0, xfer_t1;
    CUDA_CHECK(cudaEventCreate(&xfer_t0));
    CUDA_CHECK(cudaEventCreate(&xfer_t1));

    CUDA_CHECK(cudaEventRecord(xfer_t0, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_in, h_gray, bytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(xfer_t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, xfer_t0, xfer_t1));
    printf("[H→D xfer ] %.3f ms | %.2f GB/s\n\n",
           h2d_ms, bytes / (h2d_ms * 1e6));

    // ── PHASE 2: Co-processing (OMP CPU || GPU) ──
    CpuArgs cpu_args = {
        .in         = h_gray,
        .out        = h_out,
        .width      = width,
        .height     = height,
        .row_start  = 0,
        .row_end    = cpu_rows,
        .elapsed_ms = 0.0
    };

    cudaEvent_t gpu_t0, gpu_t1;
    float gpu_ms = 0.0f;

    double wall_start = now_ms();

    pthread_t cpu_thread;
    if (cpu_rows > 0)
        pthread_create(&cpu_thread, NULL, cpu_worker, &cpu_args);

    if (gpu_rows > 0) {
        CUDA_CHECK(cudaEventCreate(&gpu_t0));
        CUDA_CHECK(cudaEventCreate(&gpu_t1));

        dim3 block(TILE_W, TILE_H);
        dim3 grid((width    + TILE_W-1) / TILE_W,
                  (gpu_rows + TILE_H-1) / TILE_H);

        CUDA_CHECK(cudaEventRecord(gpu_t0, stream));
        sobel_tiled<<<grid, block, 0, stream>>>(
            d_in, d_out,
            width, height,
            cpu_rows,
            gpu_rows);
        CUDA_CHECK(cudaEventRecord(gpu_t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&gpu_ms, gpu_t0, gpu_t1));
    }

    if (cpu_rows > 0)
        pthread_join(cpu_thread, NULL);

    double wall_compute_ms = now_ms() - wall_start;

    // ── PHASE 3: D→H transfer (GPU rows only) ──
    float d2h_ms = 0.0f;
    if (gpu_rows > 0) {
        cudaEvent_t d2h_t0, d2h_t1;
        CUDA_CHECK(cudaEventCreate(&d2h_t0));
        CUDA_CHECK(cudaEventCreate(&d2h_t1));

        size_t gpu_offset = (size_t)cpu_rows * width;
        size_t gpu_bytes  = (size_t)gpu_rows * width * sizeof(float);

        CUDA_CHECK(cudaEventRecord(d2h_t0, stream));
        CUDA_CHECK(cudaMemcpyAsync(
            h_out  + gpu_offset,
            d_out  + gpu_offset,
            gpu_bytes,
            cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(d2h_t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_t0, d2h_t1));

        cudaEventDestroy(d2h_t0);
        cudaEventDestroy(d2h_t1);
    }

    double wall_total_ms = now_ms() - wall_start;  // includes D→H

    // ── Results ──
    double checksum    = compute_checksum(h_out, (int)n);
    double mpx_per_sec = ((double)n) / (wall_compute_ms * 1e3);

    printf("GPU Sobel (Hetero) | %dx%d | Time=%.2f ms | MP/s=%.4f | Checksum=%.6f\n",
           width, height, wall_compute_ms, mpx_per_sec, checksum);

    printf("\n--- Breakdown ---\n");
    printf("  CPU part    : %d rows | Time=%.2f ms  (%d OMP threads)\n",
           cpu_rows, cpu_args.elapsed_ms, omp_threads);
    printf("  GPU kernel  : %d rows | Time=%.2f ms\n",
           gpu_rows, gpu_ms);
    printf("  H→D xfer    : %.3f ms  (pinned, async)\n", h2d_ms);
    printf("  D→H xfer    : %.3f ms  (pinned, async, GPU rows only)\n", d2h_ms);
    printf("  Compute wall: %.2f ms  (CPU || GPU, transfer excluded)\n",
           wall_compute_ms);
    printf("  Total wall  : %.2f ms  (includes D→H)\n", wall_total_ms);

    // ── Bottleneck indicator ──
    const char *bottleneck;
    printf("\n--- Bottleneck ---\n");
    if (cpu_args.elapsed_ms > gpu_ms + d2h_ms) {
        printf("  >> CPU was the bottleneck (%.2f ms vs GPU %.2f ms)\n",
               cpu_args.elapsed_ms, gpu_ms + d2h_ms);
        bottleneck = "CPU";
    } else {
        printf("  >> GPU was the bottleneck (%.2f ms vs CPU %.2f ms)\n",
               gpu_ms + d2h_ms, cpu_args.elapsed_ms);
        bottleneck = "GPU";
    }
    printf("     Idle time : %.2f ms wasted\n",
           fabsf(cpu_args.elapsed_ms - (gpu_ms + d2h_ms)));

    // ── Save edge image ──
    unsigned char *img_out = normalize_to_uchar(h_out, (int)n);
    char out_path[512];
    const char *dot = strrchr(input_path, '.');
    int base_len = dot ? (int)(dot - input_path) : (int)strlen(input_path);
    snprintf(out_path, sizeof(out_path), "%.*s-edges-hetero.png",
             base_len, input_path);
    stbi_write_png(out_path, width, height, 1, img_out, width);
    printf("\nSaved : %s\n", out_path);

    // ── Save results to CSV ──
    save_conv_csv(
        "CPU_GPU_Hetero",
        width, height,
        /*threads=*/      omp_threads,      // actual OMP thread count
        /*cpu_fraction=*/ cpu_fraction,
        /*cpu_rows=*/     cpu_rows,
        /*gpu_rows=*/     gpu_rows,
        /*compute_ms=*/   wall_compute_ms,
        /*wall_total=*/   wall_total_ms,
        /*cpu_part_ms=*/  cpu_args.elapsed_ms,
        /*gpu_kernel=*/   gpu_ms,
        /*h2d_ms=*/       h2d_ms,
        /*d2h_ms=*/       d2h_ms,
        mpx_per_sec, checksum,
        bottleneck
    );

    // ── Cleanup ──
    cudaStreamDestroy(stream);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(xfer_t0); cudaEventDestroy(xfer_t1);
    if (gpu_rows > 0) {
        cudaEventDestroy(gpu_t0);
        cudaEventDestroy(gpu_t1);
    }
    cudaFreeHost(h_gray);
    cudaFreeHost(h_out);
    free(img_out);

    return 0;
}