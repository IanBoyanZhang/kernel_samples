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

__global__ void elementwise_add_float4(float* a, float* b, float* c, int N) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx >= N) {
      return;
    }

    float4 tmp_a = *(reinterpret_cast<float4*>(&a[idx]));
    float4 tmp_b = *(reinterpret_cast<float4*>(&b[idx]));
    float4 tmp_c;
    tmp_c.x = tmp_a.x + tmp_b.x;
    tmp_c.y = tmp_a.y + tmp_b.y;
    tmp_c.z = tmp_a.z + tmp_b.z;
    tmp_c.w = tmp_a.w + tmp_b.w;
    *reinterpret_cast<float4*>(&c[idx]) = tmp_c;
}

int main() {
    constexpr int N = 1e6;
    float* a_h = (float*)malloc(N * sizeof(float));
    float* b_h = (float*)malloc(N * sizeof(float));
    float* c_h = (float*)malloc(N * sizeof(float));
   
    for (int i = 0; i < N; i++) {
        a_h[i] = i;
	b_h[i] = N - 1 - i;
    }

    float* a_d{nullptr};
    float* b_d{nullptr};
    float* c_d{nullptr};

    // Apply padding
    constexpr size_t bytes_needed = N * sizeof(float);
    // Round up to nearest multiple of 16
    constexpr size_t padded_bytes = (bytes_needed + 15) & ~15;

    cudaCheck(cudaMalloc((void**)&a_d, padded_bytes));
    cudaCheck(cudaMalloc((void**)&b_d, padded_bytes));
    cudaCheck(cudaMalloc((void**)&c_d, padded_bytes));
    cudaCheck(cudaMemcpy(a_d, a_h, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(b_d, b_h, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_size{1024};
    int grid_size{div_ceil(div_ceil(N, 4), 1024)};

    printf("grid_size %d, block_size: %d\n", grid_size, block_size);

    elementwise_add_float4<<<grid_size, block_size>>>(a_d, b_d, c_d, N);

    cudaCheck(cudaMemcpy(c_h, c_d, N * sizeof(float), cudaMemcpyDeviceToHost));

    cudaCheck(cudaFree(a_d));
    cudaCheck(cudaFree(b_d));
    cudaCheck(cudaFree(c_d));

    free(a_h);
    free(b_h);
    free(c_h);

    for (int i = 0; i < N; i++ ) {
        if (i == N-1) { 
	    printf("%f\n", a_h[i]);
	}
        else {
	    printf("%f ", a_h[i]);
	}
    }
    printf("b_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) { 
	    printf("%f\n", b_h[i]);
	}
        else {
	    printf("%f ", b_h[i]);
	}
    }
    printf("c_h:\n");
    for (int i = 0; i < N; i++ ) {
        if (i == N-1) { 
	    printf("%f\n", c_h[i]);
	}
        else { 
	    printf("%f ", c_h[i]);
	}
    }

    return 0;
}
