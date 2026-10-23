// conv_cpu.cpp
// Sobel edge detection on city-view.png
// Outputs: city-view-edges.png
// Usage: ./conv_cpu [input.png]

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include "../conv_csv.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

// ─────────────────────────────────────────────
// Sobel kernels
// ─────────────────────────────────────────────
static const int Gx[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

static const int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};


// ─────────────────────────────────────────────
// Convert RGB image to grayscale (in-place → new buffer)
// ─────────────────────────────────────────────
float* rgb_to_gray(const unsigned char *rgb, int width, int height) {
    float *gray = (float*)malloc(width * height * sizeof(float));
    for (int i = 0; i < width * height; i++) {
        float r = rgb[i * 3 + 0];
        float g = rgb[i * 3 + 1];
        float b = rgb[i * 3 + 2];
        gray[i] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
    return gray;
}


// ─────────────────────────────────────────────
// Sobel filter — serial
// ─────────────────────────────────────────────
void sobel_serial(const float *in, float *out, int width, int height) {
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    float px = in[(y + ky) * width + (x + kx)];
                    gx += Gx[ky + 1][kx + 1] * px;
                    gy += Gy[ky + 1][kx + 1] * px;
                }
            }

            out[y * width + x] = sqrtf(gx * gx + gy * gy);
        }
    }
}


// ─────────────────────────────────────────────
// Normalize float buffer [0, max] → [0, 255]
// ─────────────────────────────────────────────
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


// ─────────────────────────────────────────────
// Checksum — sum of all output pixel values
// ─────────────────────────────────────────────
double compute_checksum(const float *buf, int width, int height) {
    double sum = 0.0;
    for (int i = 0; i < width * height; i++)
        sum += buf[i];
    return sum;
}


// ─────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────
int main(int argc, char *argv[]) {
    const char *input_path = (argc >= 2) ? argv[1] : "../city-view.png";

    // ── Load image ──
    int width, height, channels;
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 3);
    if (!img) {
        fprintf(stderr, "Error: could not load '%s'\n", input_path);
        return 1;
    }
    printf("Loaded: %s | %dx%d | channels=%d\n", input_path, width, height, channels);

    // ── Grayscale ──
    float *gray = rgb_to_gray(img, width, height);
    stbi_image_free(img);

    // ── Allocate output ──
    float *edges = (float*)calloc(width * height, sizeof(float));

    // ── Serial Sobel ──
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    sobel_serial(gray, edges, width, height);

    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec  - t0.tv_sec)  * 1e3 +
                (t1.tv_nsec - t0.tv_nsec) / 1e6;

    double checksum    = compute_checksum(edges, width, height);
    double mpx_per_sec = ((double)width * height) / (ms * 1e3);

    printf("CPU Sobel (Serial) | %dx%d | Time=%.2f ms | MP/s=%.4f | Checksum=%.6f\n",
           width, height, ms, mpx_per_sec, checksum);

    // ── Save edge image ──
    unsigned char *out_img = normalize_to_uchar(edges, width, height);

    char out_path[512];
    const char *dot = strrchr(input_path, '.');
    if (dot) {
        int base_len = (int)(dot - input_path);
        snprintf(out_path, sizeof(out_path), "%.*s-edges.png", base_len, input_path);
    } else {
        snprintf(out_path, sizeof(out_path), "%s-edges.png", input_path);
    }

    if (stbi_write_png(out_path, width, height, 1, out_img, width)) {
        printf("Saved : %s\n", out_path);
    } else {
        fprintf(stderr, "Error: could not write '%s'\n", out_path);
    }

    // ── Save results to CSV (wide format) ──
    save_conv_csv(
        "CPU_Serial",
        width, height,
        /*threads=*/      1,
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

    free(gray);
    free(edges);
    free(out_img);
    return 0;
}