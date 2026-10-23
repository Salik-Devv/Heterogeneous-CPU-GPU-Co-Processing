#pragma once
// prefix_sum_csv.h
// Wide-format CSV logger shared by all prefix sum variants.
//
// Columns:
//   implementation, N, threads,
//   cpu_fraction, cpu_elems, gpu_elems,
//   compute_time_ms, wall_total_ms,
//   cpu_part_ms, gpu_kernel_ms, sync_overhead_ms,
//   h2d_ms, d2h_ms,
//   bandwidth_gb_s, checksum, bottleneck
//
// Pass -1.0 / -1 for fields that do not apply to a variant —
// they are written as empty cells so the column count stays fixed.

#include <stdio.h>
#include <stdlib.h>

static const char *PREFIX_CSV_PATH = "/kaggle/working/prefix_results.csv";

static const char *PREFIX_CSV_HEADER =
    "implementation,N,threads,"
    "cpu_fraction,cpu_elems,gpu_elems,"
    "compute_time_ms,wall_total_ms,"
    "cpu_part_ms,gpu_kernel_ms,sync_overhead_ms,"
    "h2d_ms,d2h_ms,"
    "bandwidth_gb_s,checksum,bottleneck\n";

static void save_prefix_csv(
    const char *impl,
    long   N,
    int    threads,
    double cpu_fraction,       // -1 if N/A
    long   cpu_elems,          // -1 if N/A
    long   gpu_elems,          // -1 if N/A
    double compute_time_ms,    // kernel/compute wall time (always set)
    double wall_total_ms,      // full wall including transfers; -1 if same as compute
    double cpu_part_ms,        // -1 if N/A
    double gpu_kernel_ms,      // -1 if N/A
    double sync_overhead_ms,   // -1 if N/A (hetero only)
    double h2d_ms,             // -1 if N/A
    double d2h_ms,             // -1 if N/A
    double bandwidth_gb_s,
    double checksum,
    const char *bottleneck     // "CPU", "GPU", or "" if N/A
)
{
    FILE *test = fopen(PREFIX_CSV_PATH, "r");
    int need_header = (test == NULL);
    if (test) fclose(test);

    FILE *fp = fopen(PREFIX_CSV_PATH, "a");
    if (!fp) {
        fprintf(stderr, "Warning: could not open CSV '%s'\n", PREFIX_CSV_PATH);
        return;
    }
    if (need_header) fputs(PREFIX_CSV_HEADER, fp);

    // impl, N, threads
    fprintf(fp, "%s,%ld,%d,", impl, N, threads);

    // cpu_fraction
    if (cpu_fraction >= 0.0) fprintf(fp, "%.4f,", cpu_fraction); else fprintf(fp, ",");

    // cpu_elems
    if (cpu_elems >= 0) fprintf(fp, "%ld,", cpu_elems); else fprintf(fp, ",");
    // gpu_elems
    if (gpu_elems >= 0) fprintf(fp, "%ld,", gpu_elems); else fprintf(fp, ",");

    // compute_time_ms (always present)
    fprintf(fp, "%.4f,", compute_time_ms);

    // wall_total_ms
    if (wall_total_ms >= 0.0) fprintf(fp, "%.4f,", wall_total_ms);
    else fprintf(fp, "%.4f,", compute_time_ms);  // fall back to compute

    // cpu_part_ms
    if (cpu_part_ms     >= 0.0) fprintf(fp, "%.4f,", cpu_part_ms);     else fprintf(fp, ",");
    // gpu_kernel_ms
    if (gpu_kernel_ms   >= 0.0) fprintf(fp, "%.4f,", gpu_kernel_ms);   else fprintf(fp, ",");
    // sync_overhead_ms
    if (sync_overhead_ms >= 0.0) fprintf(fp, "%.4f,", sync_overhead_ms); else fprintf(fp, ",");

    // h2d_ms
    if (h2d_ms >= 0.0) fprintf(fp, "%.4f,", h2d_ms); else fprintf(fp, ",");
    // d2h_ms
    if (d2h_ms >= 0.0) fprintf(fp, "%.4f,", d2h_ms); else fprintf(fp, ",");

    // bandwidth, checksum, bottleneck
    fprintf(fp, "%.6f,%.6f,%s\n",
            bandwidth_gb_s, checksum,
            bottleneck ? bottleneck : "");

    fclose(fp);
    printf("Result : %s\n", PREFIX_CSV_PATH);
}
