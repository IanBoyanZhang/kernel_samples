#include <algorithm>
#include <limits>
#include <cfloat>
#include "utils.cuh"

void max_cpu(float* input, float* output, int N) {
    *output = *(std::max_element(input, input + N));
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

int main() {
    size_t const N = 12800;
    const int repeat_times = 10;

    float* input = (float*)malloc(sizeof(float) * N);
    for (int i = N; i > 0; i--) {
        input[i] = i;
    }

    float* output_ref = (float*)malloc(1 * sizeof(float));
    float total_time_h = time_record(repeat_times, ([&]{max_cpu(input, output_ref, N);}));
    printf("[max_cpu]: total_time_h = %f ms\n", total_time_h / repeat_times);

    float* output = (float*)malloc(1 * sizeof(float));
    float* input_device  = nullptr;
    float* output_device = nullptr;

    cudaMalloc(&input_device, N * sizeof(float));
    cudaMalloc(&output_device, 1 * sizeof(float));
    cudaMemcpy(input_device, input, N * sizeof(float), cudaMemcpyHostToDevice);


    constexpr size_t BLOCK_SIZE = 128;
    int block_size = BLOCK_SIZE;
    // int grid_size  = div_ceil(N, BLOCK_SIZE);
    int grid_size  = ceil(N, BLOCK_SIZE);
    printf("grid_size: %d, block_size: %d", grid_size, block_size);
    float total_time_1 = time_record(repeat_times,  ([&]{maxKernel<<<grid_size, block_size>>>(input_device, output_device, N);}));
    printf("[max_kernel]: total_time_1 = %f ms\n", total_time_1 / repeat_times);
    cudaMemcpy(output, output_device, 1 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("output = %f, output_ref = %f\n", *output, *output_ref);

    free(input);
    free(output);
    free(output_ref);

    cudaFree(input_device);
    cudaFree(output_device);

    return 0;
}
