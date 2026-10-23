#pragma once
// conv_csv.h
// Wide-format CSV logger shared by all convolution variants.
//
// Columns:
//   implementation, width, height, threads, cpu_fraction,
//   cpu_rows, gpu_rows,
//   compute_time_ms, wall_total_ms,
//   cpu_part_ms, gpu_kernel_ms,
//   h2d_ms, d2h_ms,
//   throughput_mpx_per_sec, checksum, bottleneck
//
// Pass -1.0 / -1 for fields that do not apply to a variant —
// they are written as empty cells so the column count stays fixed.

#include <stdio.h>

static const char *CONV_CSV_PATH = "../../../../results/convolution_results.csv";

static const char *CONV_CSV_HEADER =
    "implementation,width,height,threads,cpu_fraction,"
    "cpu_rows,gpu_rows,"
    "compute_time_ms,wall_total_ms,"
    "cpu_part_ms,gpu_kernel_ms,"
    "h2d_ms,d2h_ms,"
    "throughput_mpx_per_sec,checksum,bottleneck\n";

static void save_conv_csv(
    const char *impl,
    int   width,     int    height,
    int   threads,
    double cpu_fraction,          // -1 if N/A
    int   cpu_rows,  int    gpu_rows,   // -1 if N/A
    double compute_time_ms,       // wall time of the compute phase
    double wall_total_ms,         // includes D→H; same as compute if no GPU
    double cpu_part_ms,           // -1 if N/A
    double gpu_kernel_ms,         // -1 if N/A
    double h2d_ms,                // -1 if N/A
    double d2h_ms,                // -1 if N/A
    double mpx_per_sec,
    double checksum,
    const char *bottleneck        // "CPU", "GPU", or "" if N/A
)
{
    // Write header only if file does not yet exist
    FILE *test = fopen(CONV_CSV_PATH, "r");
    int need_header = (test == NULL);
    if (test) fclose(test);

    FILE *fp = fopen(CONV_CSV_PATH, "a");
    if (!fp) {
        fprintf(stderr, "Warning: could not open CSV '%s'\n", CONV_CSV_PATH);
        return;
    }
    if (need_header) fputs(CONV_CSV_HEADER, fp);

    // impl, dimensions, threads
    fprintf(fp, "%s,%d,%d,%d,", impl, width, height, threads);

    // cpu_fraction
    if (cpu_fraction >= 0.0) fprintf(fp, "%.4f,", cpu_fraction); else fprintf(fp, ",");

    // cpu_rows
    if (cpu_rows >= 0) fprintf(fp, "%d,", cpu_rows); else fprintf(fp, ",");
    // gpu_rows
    if (gpu_rows >= 0) fprintf(fp, "%d,", gpu_rows); else fprintf(fp, ",");

    // compute_time_ms (always present)
    fprintf(fp, "%.4f,", compute_time_ms);

    // wall_total_ms
    if (wall_total_ms >= 0.0) fprintf(fp, "%.4f,", wall_total_ms); else fprintf(fp, ",");

    // cpu_part_ms
    if (cpu_part_ms >= 0.0) fprintf(fp, "%.4f,", cpu_part_ms); else fprintf(fp, ",");
    // gpu_kernel_ms
    if (gpu_kernel_ms >= 0.0) fprintf(fp, "%.4f,", gpu_kernel_ms); else fprintf(fp, ",");

    // h2d_ms
    if (h2d_ms >= 0.0) fprintf(fp, "%.4f,", h2d_ms); else fprintf(fp, ",");
    // d2h_ms
    if (d2h_ms >= 0.0) fprintf(fp, "%.4f,", d2h_ms); else fprintf(fp, ",");

    // throughput, checksum, bottleneck
    fprintf(fp, "%.6f,%.6f,%s\n",
            mpx_per_sec, checksum,
            bottleneck ? bottleneck : "");

    fclose(fp);
    printf("Result : %s\n", CONV_CSV_PATH);
}