#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
//#include <cfloat>
#include <float.h>
#include "utils.cuh"

constexpr size_t N = 2048;
constexpr size_t BLOCK_SIZE = 256;
constexpr int repeated_times = 10;

__global__ void setToNegativeMax(float* d_value) {
    *d_value = -FLT_MAX;
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

// From host can only call __global__ function not device
__global__ void maxKernel(float* input, float* output, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    // mask out or ignore threads/elements that fall outside the valid boundaries of your data
    // Find max
    float val = (idx < N) ? input[idx] : (-FLT_MAX);
    //  calling a constexpr __host__ function("lowest") from a __global__ function("maxKernel") is not allowed. The experimental flag '--expt-relaxed-constexpr' can be used to allow this.
    // float val = (idx < N) ? input[idx] : std::numeric_limits<float>::lowest();
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    } 
    if (laneId == 0) {
        s_mem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
        // val = (laneId < warpNum) ? s_mem[laneId] : std::numeric_limits<float>::lowest();
        val = (laneId < warpNum) ? s_mem[laneId] : (-FLT_MAX);
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
        }
        if (laneId == 0) {
            atomicMax(output, val);
        }
    }
}

__global__ void sumKernel(float* input, float* sum, float* max_val, int N) {
    __shared__ float s_mem[32];
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    
    if (laneId == 0) {
        s_mem[warpId] = val;
    }
    __syncthreads();

    if (warpId == 0) {
        int warpNum = blockDim.x / warpSize;
	val = (laneId < warpNum) ? s_mem[laneId] : 0.0f;
	for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
	    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
	}
	if (laneId == 0) {
	    atomicAdd(sum, val);
	}
    }
}

__global__ void softmaxKernel(float* input, float* output, float* sum, float* max_val, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        output[idx] = expf(input[idx] - *max_val) / (*sum);
    }
}

// CPU
void softmax(float* input, float* output, int N, float* M, float* sum) {
    *M = *(std::max_element(input, input + N));
    *sum = 0;
    for (int i = 0; i < N; i++) {
        output[i] = std::exp(input[i] - *M);
	*sum += output[i];
    }
    for (int i = 0; i < N; i++) {
        output[i] /= *sum;
    }
}

void call_softmax_v1(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    // int grid_size = div_ceil(N, BLOCK_SIZE); 
    int grid_size = 8; 

    // 1. initialization
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));
    cudaCheck(cudaMemset(total_max_device, 0, sizeof(float)));

    // 2. summation
    sumKernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 3. softmax (without minus max)
    softmaxKernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}

void call_softmax_v2(float* output, float* input_device, float* output_device, float* total_device, float* total_max_device, int N) {
    int block_size = BLOCK_SIZE;
    //int grid_size = div_ceil(N, BLOCK_SIZE); 
    int grid_size = 8; 

    // 1. initialization
    cudaCheck(cudaMemset(total_device, 0, sizeof(float)));
    cudaCheck(cudaMemset(total_max_device, 0, sizeof(float)));

    // 2. max values
    maxKernel<<<grid_size, block_size>>>(input_device, total_max_device, N);

    // 3. summation
    sumKernel<<<grid_size, block_size>>>(input_device, total_device, total_max_device, N);

    // 4. softmax (minus max values to prevent overflow)
    softmaxKernel<<<grid_size, block_size>>>(input_device, output_device, total_device, total_max_device, N);
}



int main() {
    float* input = (float*)malloc(sizeof(float) * N);
    float* output_ref = (float*)malloc(sizeof(float) * N);
    float* M = (float*)malloc(sizeof(float));
    float* sum = (float*)malloc(sizeof(float));

    for (int i = 0; i < N; i++) {
        input[i] = i / (float)N;
    }

    float total_time_h = time_record(repeated_times, ([&]{softmax(input, output_ref, N, M, sum);}));
    printf("[softmax_cpu]: total_time_h = %f ms\n", total_time_h / repeated_times);

    float* input_device  = nullptr;
    float* output_device = nullptr;
    float* total_device = nullptr;
    float* total_max_device = nullptr;
    cudaCheck(cudaMalloc(&input_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&output_device, N * sizeof(float)));
    cudaCheck(cudaMalloc(&total_device, 1 * sizeof(float)));
    cudaCheck(cudaMalloc(&total_max_device, 1 * sizeof(float)));

    cudaCheck(cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice));
    float* output = (float*)malloc(sizeof(float) * N);

    // softmax_v1
    float total_time_1 = time_record(repeated_times, ([&]{call_softmax_v1(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel1]: total_time_1 = %f ms\n", total_time_1 / repeated_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize(); 
    verify_matrix(output, output_ref, N);

    // softmax_v2
    float total_time_2 = time_record(repeated_times, ([&]{call_softmax_v2(output, input_device, output_device, total_device, total_max_device, N);}));
    printf("[softmax_kernel2]: total_time_2 = %f ms\n", total_time_2 / repeated_times);
    cudaCheck(cudaMemcpy(output, output_device, N * sizeof(float), cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    verify_matrix(output, output_ref, N);

    float* total_host = (float*)malloc(sizeof(float));
    float* total_max_host = (float*)malloc(sizeof(float));
    cudaCheck(cudaMemcpy(total_host, total_device, sizeof(float), cudaMemcpyDeviceToHost));
    cudaCheck(cudaMemcpy(total_max_host, total_max_device, sizeof(float), cudaMemcpyDeviceToHost));

    free(input);
    free(output);
    free(M);
    free(sum);
    free(output_ref);
    cudaCheck(cudaFree(input_device));
    cudaCheck(cudaFree(output_device));
    cudaCheck(cudaFree(total_device));
    cudaCheck(cudaFree(total_max_device));
    return 0;
}
