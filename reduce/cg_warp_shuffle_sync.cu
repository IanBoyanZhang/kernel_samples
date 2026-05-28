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
const int N = 1048576; // 1M elements to demonstrate grid-scale performance
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

// Pure Warp Shuffle Grid-Scale Softmax Kernel
__global__ void softmax_shuffle_kernel(const float* __restrict__ input, 
                                       float* __restrict__ output, 
                                       float* global_scratch_max, 
                                       float* global_scratch_sum, 
                                       int size) {
    cg::grid_group grid = cg::this_grid();
    
    // Single shared float per block used strictly for inter-warp leader broadcasts.
    // Eliminates bank conflicts and saves shared memory space.
    __shared__ float s_block_broadcast; 

    int tid = threadIdx.x;
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    int warpNum = blockDim.x / warpSize;
    
    int global_tid = blockDim.x * blockIdx.x + threadIdx.x;
    int grid_stride = blockDim.x * gridDim.x;

    // --- STEP 1: Global Maximum (Warp Shuffle Reduction) ---
    float local_max = -FLT_MAX;
    for (int i = global_tid; i < size; i += grid_stride) {
        local_max = fmaxf(local_max, input[i]);
    }

    // Lane-to-lane fast tree reduction inside all individual warps concurrently
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        local_max = fmaxf(local_max, __shfl_down_sync(0xFFFFFFFF, local_max, offset));
    }

    // Inter-warp exchange using a single broadcast float
    float block_max = local_max; 
    if (warpNum > 1) {
        if (laneId == 0) s_block_broadcast = block_max; 
        __syncthreads();
        
        if (warpId == 0) {
            block_max = (laneId < warpNum) ? s_block_broadcast : -FLT_MAX;
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                block_max = fmaxf(block_max, __shfl_down_sync(0xFFFFFFFF, block_max, offset));
            }
        }
    }

    // Write the block result out to global scratchpad memory
    if (global_tid == blockIdx.x * blockDim.x) { 
        global_scratch_max[blockIdx.x] = block_max;
    }

    // Hardware grid synchronization across the entire GPU
    grid.sync();

    // Block 0, Thread 0 reduces the scratchpad to find the absolute global max
    if (blockIdx.x == 0 && tid == 0) {
        float absolute_max = -FLT_MAX;
        for (int i = 0; i < gridDim.x; ++i) {
            absolute_max = fmaxf(absolute_max, global_scratch_max[i]);
        }
        global_scratch_max[0] = absolute_max; 
    }

    grid.sync();
    float global_max = global_scratch_max[0];

    // --- STEP 2: Global Sum (Warp Shuffle Reduction) ---
    float local_sum = 0.0f;
    for (int i = global_tid; i < size; i += grid_stride) {
        local_sum += expf(input[i] - global_max);
    }

    // Lane-to-lane warp reduction for exponent sum
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    }

    // Inter-warp exchange for exponent sum
    float block_sum = local_sum;
    if (warpNum > 1) {
        if (laneId == 0) s_block_broadcast = block_sum;
        __syncthreads();
        
        if (warpId == 0) {
            block_sum = (laneId < warpNum) ? s_block_broadcast : 0.0f;
            for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
                block_sum += __shfl_down_sync(0xFFFFFFFF, block_sum, offset);
            }
        }
    }

    if (global_tid == blockIdx.x * blockDim.x) {
        global_scratch_sum[blockIdx.x] = block_sum;
    }

    grid.sync();

    // Block 0, Thread 0 reduces the scratchpad to get the absolute global sum
    if (blockIdx.x == 0 && tid == 0) {
        float absolute_sum = 0.0f;
        for (int i = 0; i < gridDim.x; ++i) {
            absolute_sum += global_scratch_sum[i];
        }
        global_scratch_sum[0] = absolute_sum;
    }

    grid.sync();
    float global_sum = global_scratch_sum[0];

    // --- STEP 3: Element-Wise Output Generation ---
    for (int i = global_tid; i < size; i += grid_stride) {
        output[i] = expf(input[i] - global_max) / global_sum;
    }
}

// Host Launch Driver Wrapper with Automated Occupancy Tuning
void call_softmax_shuffle(const float* input_device, float* output_device, int size) {
    int min_grid_size = 0;
    int optimal_block_size = 0;

    // Dynamically calculate the highest-performing block size for this kernel
    cudaCheck(cudaOccupancyMaxPotentialBlockSize(
        &min_grid_size, 
        &optimal_block_size, 
        softmax_shuffle_kernel, 
        0, 
        0
    ));

    int num_blocks_per_sm = 0;
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, 0));
    
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks_per_sm, 
        softmax_shuffle_kernel, 
        optimal_block_size, 
        0
    ));
    
    int grid_size = deviceProp.multiProcessorCount * num_blocks_per_sm;
    printf("[Auto-Tuning]: Selected Optimal Block Size = %d, Grid Size = %d\n", optimal_block_size, grid_size);

    // Allocate intermediate block scratchpad buffers based on the dynamic grid footprint
    float* d_scratch_max;
    float* d_scratch_sum;
    cudaCheck(cudaMalloc(&d_scratch_max, grid_size * sizeof(float)));
    cudaCheck(cudaMalloc(&d_scratch_sum, grid_size * sizeof(float)));

    void* kernel_args[] = {
        (void*)&input_device,
        (void*)&output_device,
        (void*)&d_scratch_max,
        (void*)&d_scratch_sum,
        (void*)&size
    };

    // Warm-up run
    cudaLaunchCooperativeKernel((void*)softmax_shuffle_kernel, grid_size, optimal_block_size, kernel_args, 0, nullptr);
    cudaCheck(cudaDeviceSynchronize());

    // Performance profiling loop
    auto start = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < repeat_times; ++r) {
        cudaLaunchCooperativeKernel((void*)softmax_shuffle_kernel, grid_size, optimal_block_size, kernel_args, 0, nullptr);
    }
    cudaCheck(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> elapsed = end - start;
    printf("[softmax_shuffle]: average_time = %f ms\n", elapsed.count() / repeat_times);

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

    // CPU Baseline Execution
    auto start_cpu = std::chrono::high_resolution_clock::now();
    for(int r = 0; r < repeat_times; ++r) {
        softmax_cpu(h_input, h_output_ref, N);
    }
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_cpu = end_cpu - start_cpu;
    printf("[softmax_cpu]: average_time = %f ms\n", elapsed_cpu.count() / repeat_times);

    // Device Memory Allocation
    float* d_input  = nullptr;
    float* d_output = nullptr;
    cudaCheck(cudaMalloc(&d_input, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_output, N * sizeof(float)));

    // Copy data from host to device
    cudaCheck(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Execute optimized kernel
    call_softmax_shuffle(d_input, d_output, N);

    // Copy results back to host and verify correctness
    cudaCheck(cudaMemcpy(h_output_gpu, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));
    verify_matrix(h_output_gpu, h_output_ref, N);

    // Free resources
    free(h_input);
    free(h_output_ref);
    free(h_output_gpu);
    cudaCheck(cudaFree(d_input));
    cudaCheck(cudaFree(d_output));

    return 0;
}
