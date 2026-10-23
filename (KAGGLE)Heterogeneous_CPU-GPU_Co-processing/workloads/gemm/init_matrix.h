#pragma once
#include <cstdlib>

inline void init_matrix(float *M, int N) {
    for (int i = 0; i < N*N; i++)
        M[i] = static_cast<float>(rand()) / RAND_MAX;
}