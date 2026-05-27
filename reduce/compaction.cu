#include <iostream>
#include <cuda_runtime.h>
#include <vector>
#include <numeric>

// 1. Define the CUDA Compaction Kernel
// __global__ void filterPositiveNumbers(const int* input, int* output, int* globalCount, int N) {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int laneId = threadIdx.x % 32;

//     // Evaluate condition
//     bool keep = (idx < N) && (input[idx] > 0);
//     int myValue = keep ? input[idx] : 0;

//     // Map out the warp's active threads
//     unsigned int activeMask = __ballot_sync(0xFFFFFFFF, keep);

//     if (keep) {
//         // Find how many valid items are to the left of this thread
//         unsigned int lowerLanesMask = (1U << laneId) - 1;
//         int warpDestLane = __popc(activeMask & lowerLanesMask); 

//         // Total number of valid elements in this warp
//         int totalWarpValid = __popc(activeMask);

//         // First active thread in warp allocates global memory space for the whole warp
//         __shared__ int warpGlobalOffset;
//         if (warpDestLane == 0) {
//             warpGlobalOffset = atomicAdd(globalCount, totalWarpValid);
//         }
//         __syncthreads(); // Sync shared memory offset

//         // Compact data using shuffle down
//         for (int offset = 16; offset > 0; offset /= 2) {
//             int shuffledVal = __shfl_down_sync(activeMask, myValue, offset);
            
//             // Route values down to sequential active lanes
//             if (laneId + offset < 32 && ((activeMask >> (laneId + offset)) & 1)) {
//                 myValue = shuffledVal; 
//             }
//         }

//         // Write compacted data directly to global memory without gaps
//         output[warpGlobalOffset + warpDestLane] = myValue; 
//     }
// }

// Option 1
__global__ void filterPositiveNumbers(const int* input, int* output, int* globalCount, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;

    // 1. Evaluate condition
    bool keep = (idx < N) && (input[idx] > 0);
    int myValue = keep ? input[idx] : 0;

    // 2. Map out the warp's active threads
    unsigned int activeMask = __ballot_sync(0xFFFFFFFF, keep);

    if (keep) {
        // Find how many valid items are to the left of this thread in the warp
        unsigned int lowerLanesMask = (1U << laneId) - 1;
        int warpDestLane = __popc(activeMask & lowerLanesMask); 

        // Allocate shared/global offset on a per-warp basis using an array indexed by warpId
        // Or simply let warpDestLane == 0 dynamically perform the atomic add
        __shared__ int warpGlobalOffsets[32]; // Accommodates up to 32 warps (1024 threads) per block
        
        if (warpDestLane == 0) {
            int totalWarpValid = __popc(activeMask);
            warpGlobalOffsets[warpId] = atomicAdd(globalCount, totalWarpValid);
        }
        
        // Use a warp-level sync instead of a block-level barrier
        __syncwarp(activeMask); 

        // Each thread writes its own value directly to its exact compacted index
        output[warpGlobalOffsets[warpId] + warpDestLane] = myValue; 
    }
}


// 2. Main Host Execution Code
int main() {
    // Problem parameters
    const int N = 1000;
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Host vectors
    std::vector<int> h_input(N);
    std::vector<int> h_output(N, 0);
    int h_globalCount = 0;

    // Initialize input data with an alternating sequence of positive and negative numbers
    for (int i = 0; i < N; ++i) {
        h_input[i] = (i % 2 == 0) ? (i + 1) : -(i + 1); // 1, -2, 3, -4, 5...
    }

    for (int i = 0; i < 30; ++i) {
        std::cout << h_input[i] << " ";
    }

    std::cout << std::endl;

    // Device pointers
    int *d_input = nullptr;
    int *d_output = nullptr;
    int *d_globalCount = nullptr;

    // Allocate Device Memory
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));
    cudaMalloc(&d_globalCount, sizeof(int));

    // Copy data from Host to Device
    cudaMemcpy(d_input, h_input.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_globalCount, &h_globalCount, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_output, 0, N * sizeof(int));

    // Launch the Kernel
    filterPositiveNumbers<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_globalCount, N);

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy results back to Host
    cudaMemcpy(h_output.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_globalCount, d_globalCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Verify results
    std::cout << "Total elements keeping condition: " << h_globalCount << std::endl;
    std::cout << "First 10 compacted elements in output array:" << std::endl;
    for (int i = 0; i < std::min(10, h_globalCount); ++i) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    // Free Device Memory
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_globalCount);

    return 0;
}
