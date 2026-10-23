// conv_gpu.cu
// GPU Sobel edge detection on city-view.png
// Kernels: 1) Naive  2) Shared-memory tiled
// Usage: ./conv_gpu [input.png] [mode: 0=naive 1=tiled 2=all]

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "../conv_csv.h"

#define TILE_W 32
#define TILE_H 32

// ─────────────────────────────────────────────
// Error macro
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


// ─────────────────────────────────────────────
// Sobel constants in GPU constant memory
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
// Kernel 1: Naive
// ─────────────────────────────────────────────
__global__ void sobel_naive(
    const float *in, float *out, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;

    float gx = 0.0f, gy = 0.0f;

    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            float px = in[(y+ky) * width + (x+kx)];
            gx += c_Gx[ky+1][kx+1] * px;
            gy += c_Gy[ky+1][kx+1] * px;
        }

    out[y * width + x] = sqrtf(gx*gx + gy*gy);
}


// ─────────────────────────────────────────────
// Kernel 2: Shared-memory tiled
// ─────────────────────────────────────────────
__global__ void sobel_tiled(
    const float *in, float *out, int width, int height)
{
    __shared__ float smem[TILE_H + 2][TILE_W + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int x = blockIdx.x * TILE_W + tx;
    int y = blockIdx.y * TILE_H + ty;

    int smem_x = tx + 1;
    int smem_y = ty + 1;

    if (x < width && y < height)
        smem[smem_y][smem_x] = in[y * width + x];
    else
        smem[smem_y][smem_x] = 0.0f;

    if (tx == 0) {
        int gx = x - 1;
        smem[smem_y][0] = (gx >= 0 && y < height) ? in[y * width + gx] : 0.0f;
    }
    if (tx == TILE_W - 1) {
        int gx = x + 1;
        smem[smem_y][TILE_W+1] = (gx < width && y < height) ? in[y * width + gx] : 0.0f;
    }
    if (ty == 0) {
        int gy = y - 1;
        smem[0][smem_x] = (gy >= 0 && x < width) ? in[gy * width + x] : 0.0f;
    }
    if (ty == TILE_H - 1) {
        int gy = y + 1;
        smem[TILE_H+1][smem_x] = (gy < height && x < width) ? in[gy * width + x] : 0.0f;
    }
    if (tx == 0 && ty == 0)
        smem[0][0] = (x-1>=0 && y-1>=0) ? in[(y-1)*width+(x-1)] : 0.0f;
    if (tx == TILE_W-1 && ty == 0)
        smem[0][TILE_W+1] = (x+1<width && y-1>=0) ? in[(y-1)*width+(x+1)] : 0.0f;
    if (tx == 0 && ty == TILE_H-1)
        smem[TILE_H+1][0] = (x-1>=0 && y+1<height) ? in[(y+1)*width+(x-1)] : 0.0f;
    if (tx == TILE_W-1 && ty == TILE_H-1)
        smem[TILE_H+1][TILE_W+1] = (x+1<width && y+1<height) ? in[(y+1)*width+(x+1)] : 0.0f;

    __syncthreads();

    if (x < 1 || x >= width-1 || y < 1 || y >= height-1) return;

    float gx = 0.0f, gy = 0.0f;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++) {
            float px = smem[smem_y+ky][smem_x+kx];
            gx += c_Gx[ky+1][kx+1] * px;
            gy += c_Gy[ky+1][kx+1] * px;
        }

    out[y * width + x] = sqrtf(gx*gx + gy*gy);
}


// ─────────────────────────────────────────────
// RGB → Grayscale on CPU (before upload)
// ─────────────────────────────────────────────
float* rgb_to_gray(const unsigned char *rgb, int width, int height) {
    float *gray = (float*)malloc((size_t)width * height * sizeof(float));
    for (int i = 0; i < width * height; i++)
        gray[i] = 0.299f*rgb[i*3+0] + 0.587f*rgb[i*3+1] + 0.114f*rgb[i*3+2];
    return gray;
}


// ─────────────────────────────────────────────
// Normalize float → uchar for saving
// ─────────────────────────────────────────────
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


// ─────────────────────────────────────────────
// Checksum
// ─────────────────────────────────────────────
double compute_checksum(const float *buf, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += buf[i];
    return sum;
}


// ─────────────────────────────────────────────
// Run kernel, time it, save output + CSV row
// h2d_ms passed in so it appears in each row
// ─────────────────────────────────────────────
void run_and_time(
    const char  *label,
    const char  *input_path,
    float       *d_in,
    float       *d_out,
    float       *h_out,
    int          width,
    int          height,
    int          mode,
    float        h2d_ms)     // measured once in main, shared by both kernels
{
    size_t n     = (size_t)width * height;
    size_t bytes = n * sizeof(float);

    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    dim3 block(TILE_W, TILE_H);
    dim3 grid((width  + TILE_W-1) / TILE_W,
              (height + TILE_H-1) / TILE_H);

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    CUDA_CHECK(cudaEventRecord(ev0));
    if (mode == 0)
        sobel_naive<<<grid, block>>>(d_in, d_out, width, height);
    else
        sobel_tiled<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaGetLastError());

    float gpu_kernel_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpu_kernel_ms, ev0, ev1));

    // D→H transfer timing
    cudaEvent_t d2h_e0, d2h_e1;
    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventCreate(&d2h_e0));
    CUDA_CHECK(cudaEventCreate(&d2h_e1));
    CUDA_CHECK(cudaEventRecord(d2h_e0));
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(d2h_e1));
    CUDA_CHECK(cudaEventSynchronize(d2h_e1));
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2h_e0, d2h_e1));

    double checksum    = compute_checksum(h_out, (int)n);
    double mpx_per_sec = ((double)n) / (gpu_kernel_ms * 1e3);
    double wall_total  = h2d_ms + gpu_kernel_ms + d2h_ms;

    printf("GPU Sobel (%-6s) | %dx%d | Kernel=%.2f ms | H2D=%.2f ms | D2H=%.2f ms"
           " | Total=%.2f ms | MP/s=%.4f | Checksum=%.6f\n",
           label, width, height,
           gpu_kernel_ms, h2d_ms, d2h_ms, wall_total,
           mpx_per_sec, checksum);

    // Save edge image
    unsigned char *img_out = normalize_to_uchar(h_out, (int)n);
    char out_path[512];
    const char *dot = strrchr(input_path, '.');
    int base_len = dot ? (int)(dot - input_path) : (int)strlen(input_path);
    snprintf(out_path, sizeof(out_path), "%.*s-edges-gpu-%s.png",
             base_len, input_path, label);
    stbi_write_png(out_path, width, height, 1, img_out, width);
    printf("Saved : %s\n", out_path);
    free(img_out);

    // ── Save results to CSV (wide format) ──
    char impl_name[32];
    snprintf(impl_name, sizeof(impl_name), "GPU_%s", label);

    save_conv_csv(
        impl_name,
        width, height,
        /*threads=*/      1,
        /*cpu_fraction=*/ 0.0,
        /*cpu_rows=*/     0,
        /*gpu_rows=*/     height,
        /*compute_ms=*/   gpu_kernel_ms,
        /*wall_total=*/   wall_total,
        /*cpu_part_ms=*/ -1.0,           // N/A
        /*gpu_kernel=*/   gpu_kernel_ms,
        /*h2d_ms=*/       h2d_ms,
        /*d2h_ms=*/       d2h_ms,
        mpx_per_sec, checksum,
        /*bottleneck=*/   "GPU"
    );

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(d2h_e0);
    cudaEventDestroy(d2h_e1);
}


// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const char *input_path = (argc >= 2) ? argv[1] : "../city-view.png";
    int mode               = (argc >= 3) ? atoi(argv[2]) : 2;  // default: both

    // ── Load image ──
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error: cannot load '%s'\n", input_path);
        return 1;
    }
    printf("Loaded : %s | %dx%d | channels=%d\n\n",
           input_path, width, height, channels);

    // ── GPU info ──
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU    : %s | SMs=%d | Mem=%.0f MB\n\n",
           prop.name, prop.multiProcessorCount,
           prop.totalGlobalMem / 1e6);

    size_t n     = (size_t)width * height;
    size_t bytes = n * sizeof(float);

    // ── Grayscale on CPU ──
    float *h_gray = rgb_to_gray(img, width, height);
    float *h_out  = (float*)calloc(n, sizeof(float));
    stbi_image_free(img);

    // ── Device allocation ──
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // ── Measure H→D transfer (shared by both kernels) ──
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(d_in, h_gray, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, t0, t1));
    printf("[H→D xfer] %.3f ms | %.2f GB/s\n\n",
           h2d_ms, bytes / (h2d_ms * 1e6));

    // ── Run kernels ──
    if (mode == 0 || mode == 2)
        run_and_time("Naive", input_path, d_in, d_out, h_out, width, height, 0, h2d_ms);
    if (mode == 1 || mode == 2)
        run_and_time("Tiled", input_path, d_in, d_out, h_out, width, height, 1, h2d_ms);

    // ── Cleanup ──
    cudaFree(d_in);
    cudaFree(d_out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    free(h_gray);
    free(h_out);

    return 0;
}
