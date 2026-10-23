// conv_cpu_omp.cpp
// OpenMP parallel Sobel — same output as serial for verification
// Usage: ./conv_cpu_omp [input.png] [threads]

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "../conv_csv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>

static const int Gx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
static const int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };

float* rgb_to_gray(const unsigned char *rgb, int width, int height) {
    float *gray = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++)
        gray[i] = 0.299f*rgb[i*3] + 0.587f*rgb[i*3+1] + 0.114f*rgb[i*3+2];
    return gray;
}

void sobel_omp(const float *in, float *out, int width, int height) {
    #pragma omp parallel for schedule(static) collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++) {
                    float px = in[(y+ky)*width + (x+kx)];
                    gx += Gx[ky+1][kx+1] * px;
                    gy += Gy[ky+1][kx+1] * px;
                }
            out[y*width+x] = sqrtf(gx*gx + gy*gy);
        }
    }
}

unsigned char* normalize_to_uchar(const float *buf, int width, int height) {
    float max_val = 0.0f;
    int n = width * height;
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

int main(int argc, char *argv[]) {
    const char *input_path = (argc >= 2) ? argv[1] : "../city-view.png";
    int threads            = (argc >= 3) ? atoi(argv[2]) : omp_get_max_threads();
    omp_set_num_threads(threads);

    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) { fprintf(stderr, "Cannot load %s\n", input_path); return 1; }
    printf("Loaded: %s | %dx%d\n", input_path, width, height);

    float *gray  = rgb_to_gray(img, width, height);
    float *edges = (float*)calloc(width * height, sizeof(float));
    stbi_image_free(img);

    double t0 = omp_get_wtime();
    sobel_omp(gray, edges, width, height);
    double ms = (omp_get_wtime() - t0) * 1e3;

    double checksum    = compute_checksum(edges, width * height);
    double mpx_per_sec = ((double)width * height) / (ms * 1e3);

    printf("CPU Sobel (OMP) | %dx%d | Threads=%d | Time=%.2f ms | MP/s=%.4f | Checksum=%.6f\n",
           width, height, threads, ms, mpx_per_sec, checksum);

    unsigned char *out_img = normalize_to_uchar(edges, width, height);

    char out_path[512];
    const char *dot = strrchr(input_path, '.');
    int base_len = dot ? (int)(dot - input_path) : (int)strlen(input_path);
    snprintf(out_path, sizeof(out_path), "%.*s-edges-omp.png", base_len, input_path);
    stbi_write_png(out_path, width, height, 1, out_img, width);
    printf("Saved : %s\n", out_path);

    // ── Save results to CSV (wide format) ──
    save_conv_csv(
        "CPU_OMP",
        width, height,
        /*threads=*/      threads,
        /*cpu_fraction=*/ 1.0,
        /*cpu_rows=*/     height,
        /*gpu_rows=*/     0,
        /*compute_ms=*/   ms,
        /*wall_total=*/   ms,    // no transfer overhead
        /*cpu_part_ms=*/  ms,    // full workload is CPU
        /*gpu_kernel=*/  -1.0,   // N/A
        /*h2d_ms=*/      -1.0,   // N/A
        /*d2h_ms=*/      -1.0,   // N/A
        mpx_per_sec, checksum,
        /*bottleneck=*/   "CPU"
    );

    free(gray); free(edges); free(out_img);
    return 0;
}