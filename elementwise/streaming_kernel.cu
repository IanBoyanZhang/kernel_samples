#include <cuda_runtime.h>
#include <stdio.h>

// cudaGetErrorString does not allocate memory
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

__global__ void sigmoid(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx <= N) {
        y[idx] = 1.0f / (1.0f + expf(-x[idx]));
    }
}

// float4
__global__ void sigmoid_float4(float* x, float* y, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4; 
    if (idx >= N) {
	return;
    }
    float4 tmp_x = *(reinterpret_cast<float4*>(&x[idx]));
    float4 tmp_y;
    tmp_y.x = 1.0f / (1.0f + expf(-tmp_x.x));
    tmp_y.y = 1.0f / (1.0f + expf(-tmp_x.y));
    tmp_y.z = 1.0f / (1.0f + expf(-tmp_x.z));
    tmp_y.w = 1.0f / (1.0f + expf(-tmp_x.w));
    *reinterpret_cast<float4*>(&y[idx]) = tmp_y;
}

__global__ void relu(float* x, float* y, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}

int main() {
    //constexpr int N = 7;
    constexpr int N = 1e6;

    // Apply padding
    constexpr size_t bytes_needed = N * sizeof(float);
    // Round up to nearest multiple of 16
    constexpr size_t padded_bytes = (bytes_needed + 15) & ~15;

    // 1. Allocate Host Memory
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));

    // Initialize data
    for (int i = 0; i < N; i++) {
	h_x[i] = (float)i / N;
    }

    float *d_x, *d_y;

    cudaCheck(cudaMalloc((void**)&d_x, padded_bytes));
    cudaCheck(cudaMalloc((void**)&d_y, padded_bytes));
    cudaCheck(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size{1024};
    // int grid_size{div_ceil(div_ceil(N, 4), 1024)};
    int grid_size{div_ceil(div_ceil(N, 4), 1024)};

    printf("grid_size %d, block_size: %d\n", grid_size, block_size);

    sigmoid_float4<<<grid_size, block_size>>>(d_x, d_y, N);

    cudaCheck(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_y));

    return 0;
}
