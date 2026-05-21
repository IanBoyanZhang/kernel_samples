#pragma once
#include <stdio.h>
#include <cuda_runtime.h>

#define cudaCheck(err) _cudaCheck(err, __FILE__, __LINE__)
void _cudaCheck(cudaError_t error, const char *file, int line) {
    if (error != cudaSuccess) {
        printf("[CUDA ERROR] at file %s(line %d):\n%s\n", file, line, cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
    return;
};

template<typename T>
constexpr T div_ceil(T a, T b) {
    return (a / b) + ( (a % b != 0) && ((a ^ b) >= 0) );
}

template<typename T>
constexpr T ceil(T a, T b) {
    return ((a) + (b) - 1) / (b);
}

// RAII Wrapper for CUDA Events
struct CudaEvent {
    cudaEvent_t event;
    CudaEvent() { cudaEventCreate(&event); }
    ~CudaEvent() { cudaEventDestroy(event); }
    operator cudaEvent_t() const { return event; }
};

// Modern Time Recorder
template <typename Func>
float time_record(int n, Func&& func) {
    if (n <= 0) return 0.0f;

    CudaEvent start, stop;
    float total_time = 0.0f;

    // Warmup (crucial for modern GPUs)
    func();

    for (int i = 0; i < n; ++i) {
        cudaEventRecord(start);
        func();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float elapsed;
        cudaEventElapsedTime(&elapsed, start, stop);
        total_time += elapsed;
    }
    return total_time / n; // Return average
}

// #define TIME_RECORD(N, func)                                                                    \
//     [&] {                                                                                       \
//         float total_time = 0;                                                                   \
//         for (int repeat = 0; repeat <= N; ++repeat) {                                           \
//             cudaEvent_t start, stop;                                                            \
//             cudaCheck(cudaEventCreate(&start));                                                 \
//             cudaCheck(cudaEventCreate(&stop));                                                  \
//             cudaCheck(cudaEventRecord(start));                                                  \
//             cudaEventQuery(start);                                                              \
//             func();                                                                             \
//             cudaCheck(cudaEventRecord(stop));                                                   \
//             cudaCheck(cudaEventSynchronize(stop));                                              \
//             float elapsed_time;                                                                 \
//             cudaCheck(cudaEventElapsedTime(&elapsed_time, start, stop));                        \
//             if (repeat > 0) total_time += elapsed_time;                                         \
//             cudaCheck(cudaEventDestroy(start));                                                 \
//             cudaCheck(cudaEventDestroy(stop));                                                  \
//         }                                                                                       \
//         if (N == 0) return (float)0.0;                                                          \
//         return total_time;                                                                      \
//     }()

void randomize_matrix(float *mat, int N);
void print_matrix(float* a, int M, int N);
bool verify_matrix(float *mat1, float *mat2, size_t N);
