// conv_part.h
#pragma once
 
// Applies Sobel to rows [row_start, row_end) of a full width x height image
// in  — full grayscale input  (width x height)
// out — full output buffer    (width x height), only writes assigned rows
void sobel_cpu_part(const float *in, float *out,
                    int width, int height,
                    int row_start, int row_end);
 
// Returns omp_get_max_threads() — call before any parallel region
// so the value is available to main() for logging and CSV
int sobel_cpu_num_threads(void);
