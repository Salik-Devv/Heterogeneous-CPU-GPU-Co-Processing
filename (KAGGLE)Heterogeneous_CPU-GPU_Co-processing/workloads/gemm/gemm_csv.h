#pragma once
// gemm_csv.h
// Wide-format CSV logger shared by all GEMM variants.
//
// Columns:
//   implementation, N, threads,
//   cpu_fraction, cpu_rows, gpu_rows,
//   compute_time_ms, wall_total_ms,
//   cpu_part_ms, gpu_kernel_ms,
//   h2d_ms, d2h_ms,
//   gflops, checksum, bottleneck
//
// Pass -1.0 / -1 for fields that do not apply to a variant —
// they are written as empty cells so the column count stays fixed.

#include <stdio.h>

static const char *GEMM_CSV_PATH = "/kaggle/working/gemm_results.csv";

static const char *GEMM_CSV_HEADER =
    "implementation,N,threads,"
    "cpu_fraction,cpu_rows,gpu_rows,"
    "compute_time_ms,wall_total_ms,"
    "cpu_part_ms,gpu_kernel_ms,"
    "h2d_ms,d2h_ms,"
    "gflops,checksum,bottleneck\n";

static void save_gemm_csv(
    const char *impl,
    int    N,
    int    threads,
    double cpu_fraction,      // -1 if N/A
    int    cpu_rows,          // -1 if N/A
    int    gpu_rows,          // -1 if N/A
    double compute_time_ms,   // wall time of compute phase (always set)
    double wall_total_ms,     // includes D→H; same as compute if no GPU
    double cpu_part_ms,       // -1 if N/A
    double gpu_kernel_ms,     // -1 if N/A
    double h2d_ms,            // -1 if N/A
    double d2h_ms,            // -1 if N/A
    double gflops,
    double checksum,
    const char *bottleneck    // "CPU", "GPU", or "" if N/A
)
{
    // Write header only if file does not yet exist
    FILE *test = fopen(GEMM_CSV_PATH, "r");
    int need_header = (test == NULL);
    if (test) fclose(test);

    FILE *fp = fopen(GEMM_CSV_PATH, "a");
    if (!fp) {
        fprintf(stderr, "Warning: could not open CSV '%s'\n", GEMM_CSV_PATH);
        return;
    }
    if (need_header) fputs(GEMM_CSV_HEADER, fp);

    // impl, N, threads
    fprintf(fp, "%s,%d,%d,", impl, N, threads);

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
    if (cpu_part_ms  >= 0.0) fprintf(fp, "%.4f,", cpu_part_ms);  else fprintf(fp, ",");
    // gpu_kernel_ms
    if (gpu_kernel_ms >= 0.0) fprintf(fp, "%.4f,", gpu_kernel_ms); else fprintf(fp, ",");

    // h2d_ms
    if (h2d_ms >= 0.0) fprintf(fp, "%.4f,", h2d_ms); else fprintf(fp, ",");
    // d2h_ms
    if (d2h_ms >= 0.0) fprintf(fp, "%.4f,", d2h_ms); else fprintf(fp, ",");

    // gflops, checksum, bottleneck
    fprintf(fp, "%.6f,%.6f,%s\n",
            gflops, checksum,
            bottleneck ? bottleneck : "");

    fclose(fp);
    printf("Result : %s\n", GEMM_CSV_PATH);
}
