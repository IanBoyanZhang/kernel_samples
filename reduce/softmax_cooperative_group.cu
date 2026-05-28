#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <float.h>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Configuration Constants
const int N = 1048576; // 1M elements to demonstrate grid-scale scaling
constexpr size_t BLOCK_SIZE = 256;
const int repeat_times = 10;

// Error checking macro
#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Verification function
void verify_matrix(const float* gpu_res, const float* cpu_res, int size) {
    float max_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = std::abs(gpu_res[i] - cpu_res[i]);
        if (err > max_err) max_err = err;
    }
    printf("[Verification]: Maximum absolute error = %e\n", max_err);
    if (max_err < 1e-5) {
        printf("VERIFICATION SUCCESSFUL\n");
    } else {
        printf("VERIFICATION FAILED\n");
    }
}

// CPU Reference Softmax implementation
void softmax_cpu(const float* input, float* output, int size) {
    float M = *(std::max_element(input, input + size));
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = std::exp(input[i] - M);
        sum += output[i];
    }
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Grid-Scale Fused Softmax with Cooperative Groups
__global__ void softmax_cooperative_kernel(const float* __restrict__ input, 
                                           float* __restrict__ output, 
                                           float* global_scratch_max, 
                                           float* global_scratch_sum, 
                                           int size) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float s_mem[32]; // Up to 32 warps per block capacity

    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // --- STEP 1: Global Maximum Calculation ---
    float local_max = -FLT_MAX;
    for (int i = global_tid; i < size; i += grid_stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Warp Reduction
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }
    if (laneId == 0) s_mem[warpId] = local_max;
    __syncthreads();

    // Block Reduction
    float block_max = (tid < (blockDim.x / warpSize)) ? s_mem[laneId] : -FLT_MAX;
    if (warpId == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_max = fmaxf(block_max, __shfl_down_sync(0xFFFFFFFF, block_max, offset));
        }
        if (laneId == 0) global_scratch_max[blockIdx.x] = block_max;
    }

    // Grid Sync to allow Block 0 to gather local reductions
    grid.sync();

    if (blockIdx.x == 0 && tid == 0) {
        float absolute_max = -FLT_MAX;
        for (int i = 0; i < gridDim.x; ++i) {
            absolute_max = fmaxf(absolute_max, global_scratch_max[i]);
        }
        global_scratch_max[0] = absolute_max; 
    }

    // Grid Sync to broadcast final absolute maximum
    grid.sync();
    float global_max = global_scratch_max[0];

    // --- STEP 2: Global Sum of Exponents Calculation ---
    float local_sum = 0.0f;
    for (int i = global_tid; i < size; i += grid_stride) {
        local_sum += expf(input[i] - global_max);
    }

    // Warp Reduction
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }
    if (laneId == 0) s_mem[warpId] = local_sum;
    __syncthreads();

    // Block Reduction
    float block_sum = (tid < (blockDim.x / warpSize)) ? s_mem[laneId] : 0.0f;
    if (warpId == 0) {
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
        }
        if (laneId == 0) global_scratch_sum[blockIdx.x] = block_sum;
    }

    // Grid Sync to allow Block 0 to gather final sums
    grid.sync();

    if (blockIdx.x == 0 && tid == 0) {
        float absolute_sum = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            absolute_sum += global_scratch_sum[i];
        }
        global_scratch_sum[0] = absolute_sum;
    }

    // Grid Sync to broadcast final sum
    grid.sync();
    float global_sum = global_scratch_sum[0];

    // --- STEP 3: Element-wise Evaluation ---
    for (int i = global_tid; i < size; i += grid_stride) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

// Host Launch Driver Wrapper
void call_softmax_cooperative(const float* input_device, float* output_device, int size) {
    int num_blocks_per_sm = 0;
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    // Find the ideal hardware context parameters to prevent deadlocks
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, 
        softmax_cooperative_kernel, 
        BLOCK_SIZE, 
        0
    ));
    
    int grid_size = deviceProp.multiProcessorCount * num_blocks_per_sm;

    // Allocate intermediate block scratchpad buffers
    float* d_scratch_max;
    float* d_scratch_sum;
    cudaCheck(cudaMalloc(&d_scratch_max, grid_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_scratch_sum, grid_size * sizeof(float)));

    // Packaging execution parameters
    void* kernel_args[] = {
        (void*)&input_device,
        (void*)&output_device,
        (void*)&d_scratch_max,
        (void*)&d_scratch_sum,
        (void*)&size
    };

    // Warm-up execution
    cudaLaunchCooperativeKernel((void*)softmax_cooperative_kernel, grid_size, BLOCK_SIZE, kernel_args, 0, nullptr);
    cudaCheck(cudaDeviceSynchronize());

    // Profile target kernel block
    auto start = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < repeat_times; ++r) {
        cudaLaunchCooperativeKernel((void*)softmax_cooperative_kernel, grid_size, BLOCK_SIZE, kernel_args, 0, nullptr);
    }
    cudaCheck(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("[softmax_cooperative]: average_time = %f ms\n", elapsed.count() / repeat_times);

    cudaCheck(cudaFree(d_scratch_max));
    cudaCheck(cudaFree(d_scratch_sum));
}

int main() {
    // Memory Allocations on Host
    float* h_input  = (float*)malloc(sizeof(float) * N);
    float* h_output_ref = (float*)malloc(sizeof(float) * N);
    float* h_output_gpu = (float*)malloc(sizeof(float) * N);

    // Initialize Host Input Array Data
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)i / (float)N;
    }

    // CPU Evaluation
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < repeat_times; ++r) {
        softmax_cpu(h_input, h_output_ref, N);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    printf("[softmax_cpu]: average_time = %f ms\n", elapsed_cpu.count() / repeat_times);

    // Device Allocations
    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));

    // Memory Migration to Device Array Context
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Execute Fused GPU Kernel via Cooperative Groups
    call_softmax_cooperative(d_input, d_output, N);

    // Migrate Back to Host and Validate Matrix Math Integrity
    cudaCheck(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_matrix(h_output_gpu, h_output_ref, N);

    // Context Deallocations
    free(h_input);
    free(h_output_ref);
    free(h_output_gpu);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));

    return 0;
}
