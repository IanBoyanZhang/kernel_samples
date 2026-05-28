## Atomic Numeric Ops


[atomicMax](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html#atomicmax) only takes integer values.



### `atomicCAS`

Example PTX Generation

```
// Atomic CAS on 32-bit global memory
atom.global.cas.b32 %r3, [addr], %r1, %r2;
```

- The PTX `atom.cas` instruction returns the value stored at the address before the operation, which is used to check if the swap was successful (i.e., if the returned value equals the expected `compare` value)

- Memory scope: It supports various memory scopes (e.g., `cta`, `gpu`, `sys`) to define visibility across different threads and thread blocks, which helps in implementating mutexs

- Memory space: `.global` or `.shared`

- SASS: this often maps to a `ATOM` or `RED` instruction, depends on whether the return value is used

[How can I utilize the 'red' and 'atom' PTX instructions in CUDA C++ code](https://stackoverflow.com/questions/36849679/how-can-i-utilize-the-red-and-atom-ptx-instructions-in-cuda-c-code)


### `atomicMax/Min`

CUDA API `atomicMax/Min` only supports integer types, It is probably (wasteful?) to provide IEEE complaint (Handling `nan`, `inf` or etc?)hardware support for float data type. 

But a [software based solution](https://stackoverflow.com/questions/17399119/how-do-i-use-atomicmax-on-floating-point-values-in-cuda/51549250#51549250) is (easily?) achievable.

```c++
__device__ __forceinline__ float atomicMaxFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int *)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int *)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMinFloat(float * addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMin((int *)addr, __float_as_int(value))) : 
    __uint_as_float(atomicMax((unsigned int *)addr, __float_as_uint(value)));

    return old;
}
```

Another answer in the same stackoverflow post, pointed out an edge case with negative zero. 

```c++

__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}
```

By using [`signbit`](https://en.cppreference.com/cpp/numeric/math/signbit), which is a portable way to handle [negative zero](https://stackoverflow.com/questions/13544342/why-do-floating-point-numbers-have-signed-zeros) defined in IEEE754 standard.


A complete test program is also provided by the author

```c++
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

/*
//these versions fail some of the tests involving negative 0
__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = value >= 0 ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = value >= 0 ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}
*/


__device__ __forceinline__ float atomicMinFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMin((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMax((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = !signbit(value) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
        __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

__global__ void testKernel(float* testMaxData, 
                           float* testMinData,
                           const float* testValues, 
                           int numTests)
{
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index >= numTests)
    {
        return;
    }
    float val = testValues[index];
    atomicMaxFloat(testMaxData + index, val);
    atomicMinFloat(testMinData + index, val);
}

void checkCudaErr(cudaError_t cudaStatus)
{
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime error: %s\n", cudaGetErrorString(cudaStatus));
    }
}

int main()
{
    const int numValues = 6;
    const int numTests = numValues * numValues;
    float testData[numValues] = { 0.0f, -0.0f, 1.0f, -1.0f, 200.0f, -200.0f };
    float testValuesMinMaxHost[numTests];
    float testValuesHost[numTests];

    for (int i = 0; i < numValues; ++i)
    {
        for (int j = 0; j < numValues; ++j)
        {
            /*
            We will test the values of min(a,b) and max(a,b) for
            all values of a and b in the testData array.
            */
            testValuesMinMaxHost[numValues * i + j] = testData[i];
            testValuesHost[numValues * i + j] = testData[j];
        }
    }
  
    float* devTestMax = 0;
    float* devTestMin = 0;
    float* devTestValues = 0;

    checkCudaErr(cudaSetDevice(0));
    checkCudaErr(cudaMalloc((void**)&devTestMax, numTests * sizeof(float)));
    checkCudaErr(cudaMalloc((void**)&devTestMin, numTests * sizeof(float)));
    checkCudaErr(cudaMalloc((void**)&devTestValues, numTests * sizeof(float)));

    checkCudaErr(cudaMemcpy(devTestMax, testValuesMinMaxHost, numTests * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devTestMin, testValuesMinMaxHost, numTests * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErr(cudaMemcpy(devTestValues, testValuesHost, numTests * sizeof(float), cudaMemcpyHostToDevice));

    int blockSize = 128;
    testKernel << < (numTests+(blockSize-1))/ blockSize, blockSize >> > (devTestMax, devTestMin, devTestValues, numTests);
    checkCudaErr(cudaGetLastError());
    
    float resultsMin[numTests];
    float resultsMax[numTests];

    checkCudaErr(cudaMemcpy(resultsMin, devTestMin, numTests * sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErr(cudaMemcpy(resultsMax, devTestMax, numTests * sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErr(cudaFree(devTestMax));
    checkCudaErr(cudaFree(devTestMin));
    checkCudaErr(cudaFree(devTestValues));

    int fail = 0;
    for (int i = 0; i < numTests; ++i)
    {
        float expectedMax = fmax(testValuesMinMaxHost[i], testValuesHost[i]);
        if (resultsMax[i] != expectedMax)
        {
            printf("fail, expected %f, got %f from max(%f, %f)\n",
                   expectedMax,
                   resultsMax[i],
                   testValuesMinMaxHost[i],
                   testValuesHost[i]);
            fail = 1;
        }

        float expectedMin = fmin(testValuesMinMaxHost[i], testValuesHost[i]);
        if (resultsMin[i] != expectedMin)
        {
            printf("fail, expected %f, got %f from min(%f, %f)\n",
                   expectedMin,
                   resultsMin[i],
                   testValuesMinMaxHost[i],
                   testValuesHost[i]);
            fail = 1;
        }
    }

    if (fail == 0)
    {
        printf("all tests passed\n");
    }

    return 0;
}
```

It is said, [signbit in CUDA is implemented in a branchless way](https://stackoverflow.com/questions/35433345/does-cuda-signbit-remove-divergence)




## Tree Reduction Idiom 

```cpp
for (int offset = warpSize >> 1; offset > 0; offset >>= 1)
```

This code is the standard CUDA idiom for a `parallel warp reduction`, which calculates a single result (like a sum, minimum, or maximum) across 32 threads.

$\log_{2}{32} = 5$ steps 

```cpp
// 1. Mask out-of-bounds elements (Your previous question)
float val = (idx < N) ? input[idx] : (-FLT_MAX);

// 2. Warp-level reduction to find the maximum value
for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
    val = max(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
}
// Thread 0 now contains the true maximum value of the warp
```

`__shfl_down_sync` bypasses shared memory by allowing threads to read data directly from the `registers` of other threads within the same warp.

The Streaming Multiprocessor (SM) uses an internal hardware crossbar switch or interconnect network that links the specific register file indices of the sending and receiving threads.

**Synchronous Execution:**  Because all 32 threads in a warp execute the exact same instruction at the exact same cycle (SIMT architecture), they can safely swap register data simultaneously without needing a memory buffer to coordinate.


- *0xFFFFFFFF (Mask)*: A 32-bit mask telling the GPU which threads are participating. `0xFFFFFFFF` means all 32 threads in the warp must hit this instruction before the data transfer happens.
- `val (Source)`: The variable in the current thread's register that it wants to share with a neighbor
- `offset (Shift distance)`



## Warp Level Patterns


## Softmax

Fusing could be the improvement for the existing implementation

### Fences

`__threadfence()` is memory fence not a thread barrier

- it forces memory writes made by the calling threads `visible` to all other threads before the calling thread moving past the fence
- it does not stall or pause other blocks (being scheduled)

### Synchronization

To synchronize across an entire grid

- The Kernel Boundary Method

- Cooperative Groups (CUDA 9+): 

The cooperative groups implementation is also commonly used in reduction pattern in new CUDA GPU hardware

### Hardware support

[independent thread scheduling](https://stackoverflow.com/questions/70987051/independent-thread-scheduling-since-volta) introduced since Volta arch is required for using the new features 


## Further reading

[CUDA atomicMax for float](https://forums.developer.nvidia.com/t/cuda-atomicmax-for-float/194207)

[Implementation of atomicMax for float](https://forums.developer.nvidia.com/t/implementation-of-atomicmax-for-float/220388)
