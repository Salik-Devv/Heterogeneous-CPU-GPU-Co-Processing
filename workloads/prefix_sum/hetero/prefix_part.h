// prefix_part.h
#pragma once
#include "../init_array.h"

// Computes inclusive prefix sum on a sub-array
// in     — full input array
// out    — full output array
// start  — first index this chunk owns
// end    — one past last index this chunk owns
// offset — value to add to every element (from previous chunk's total)
void prefix_sum_cpu_part(const float *in, float *out,
                          long start, long end,
                          float offset);

// Returns the total sum of in[start..end)
// Used to compute the offset for the next chunk
float prefix_sum_chunk_total(const float *in, long start, long end);