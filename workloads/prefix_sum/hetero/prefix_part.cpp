// prefix_part.cpp
#include "prefix_part.h"
#include <omp.h>
#include <stdlib.h>

void prefix_sum_cpu_part(const float *in, float *out,
                          long start, long end,
                          float offset)
{
    long n = end - start;
    if (n <= 0) return;

    int T = omp_get_max_threads();
    // Ensure we don't spawn more threads than elements
    if ((long)T > n) T = (int)n;

    long chunk = (n + T - 1) / T;
    float *partial = (float*)malloc((size_t)T * sizeof(float));

    // ── Pass 1: each thread builds a local inclusive scan on its chunk ──
    #pragma omp parallel num_threads(T)
    {
        int  tid = omp_get_thread_num();
        long lo  = start + (long)tid * chunk;
        long hi  = lo + chunk < end ? lo + chunk : end;

        if (lo < end) {
            out[lo] = in[lo];
            for (long i = lo + 1; i < hi; i++)
                out[i] = out[i-1] + in[i];
            partial[tid] = out[hi - 1];
        } else {
            partial[tid] = 0.0f;
        }
    }

    // ── Pass 2 (serial): exclusive prefix scan of per-thread totals ──
    // partial[t] becomes the carry-in offset for thread t's chunk
    float running = offset;
    for (int t = 0; t < T; t++) {
        float tmp  = partial[t];
        partial[t] = running;
        running   += tmp;
    }

    // ── Pass 3: each thread adds its carry-in offset to its chunk ──
    #pragma omp parallel num_threads(T)
    {
        int  tid = omp_get_thread_num();
        long lo  = start + (long)tid * chunk;
        long hi  = lo + chunk < end ? lo + chunk : end;
        float off = partial[tid];
        for (long i = lo; i < hi; i++)
            out[i] += off;
    }

    free(partial);
}

float prefix_sum_chunk_total(const float *in, long start, long end) {
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long i = start; i < end; i++)
        sum += in[i];
    return sum;
}