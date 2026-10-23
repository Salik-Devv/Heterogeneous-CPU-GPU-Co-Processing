// conv_part.cpp
#include "conv_part.h"
#include <math.h>
#include <omp.h>

static const int Gx[3][3] = { {-1,0,1},{-2,0,2},{-1,0,1} };
static const int Gy[3][3] = { {-1,-2,-1},{0,0,0},{1,2,1} };

void sobel_cpu_part(const float *in, float *out,
                    int width, int height,
                    int row_start, int row_end)
{
    // Clamp to valid Sobel range (skip border rows)
    int y_start = (row_start < 1)        ? 1          : row_start;
    int y_end   = (row_end   > height-1) ? height - 1 : row_end;

    #pragma omp parallel for schedule(static) num_threads(omp_get_max_threads())
    for (int y = y_start; y < y_end; y++) {
        for (int x = 1; x < width - 1; x++) {
            float gx = 0.0f, gy = 0.0f;
            for (int ky = -1; ky <= 1; ky++)
                for (int kx = -1; kx <= 1; kx++) {
                    float px = in[(y + ky) * width + (x + kx)];
                    gx += Gx[ky+1][kx+1] * px;
                    gy += Gy[ky+1][kx+1] * px;
                }
            out[y * width + x] = sqrtf(gx*gx + gy*gy);
        }
    }
}

int sobel_cpu_num_threads(void) {
    return omp_get_max_threads();
}