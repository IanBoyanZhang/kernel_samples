```
#define FLOAT4(a) *(float4*)(&(a))
```

type pruning, treating a memory address as a pointer to a different type than it was originally declared

- in standard C++, this can trigger "undefined behavior", unless use `-fno-strict-aliasing`, this might trigger compiler misoptimzation, note: `-fno-strict-aliasing` has other compiler optimization implications

- Vector types like `float4` often require 16-byte alignment


### Details

#### Vectorization

[flaot4 bandwidth advantages over plain float1](https://forums.developer.nvidia.com/t/float4-bandwidth-advantages-over-plain-float1/62799/4)

The advantage might come from reduced instruction count

issuing one large 128-bit load

increase Bytes per Thread

### Memory issues

For an unpadded version

```
cudaCheck(cudaMalloc((void**)&a_d, N * sizeof(float)));
cudaCheck(cudaMalloc((void**)&b_d, N * sizeof(float)));
cudaCheck(cudaMalloc((void**)&c_d, N * sizeof(float)));
```

compute-sanitizer would give below error

```
========= COMPUTE-SANITIZER
========= Invalid __global__ read of size 16 bytes
=========     at elementwise_add_float4(float *, float *, float *, int)+0xa0
=========     by thread (1,0,0) in block (0,0,0)
=========     Access to 0x403800010 is out of bounds
=========     and is inside the nearest allocation at 0x403800000 of size 28 bytes
=========     Saved host backtrace up to driver entry point at kernel launch time
=========         Host Frame: main [0x8f0d] in a.out
========= 
========= Program hit cudaErrorUnknown (error 999) due to "unknown error" on CUDA API call to cudaMemcpy.
=========     Saved host backtrace up to driver entry point at error
=========         Host Frame: main [0x8f27] in a.out
========= 
[CUDA ERROR] at file add.cu(line 61):
unknown error
========= Target application returned an error
========= ERROR SUMMARY: 2 errors
```

Here is a fixed version

```
constexpr size_t bytes_needed = N * sizeof(float);
// Round up to nearest multiple of 16
constexpr size_t padded_bytes = (bytes_needed + 15) & ~15;

cudaCheck(cudaMalloc((void**)&a_d, padded_bytes));
cudaCheck(cudaMalloc((void**)&b_d, padded_bytes));
cudaCheck(cudaMalloc((void**)&c_d, padded_bytes));
```



ncu reports

```
ncu -k elementwise_add_float4  ./a.out
```

```

==PROF== Disconnected from process 799
[799] a.out@127.0.0.1
  elementwise_add_float4(float *, float *, float *, int) (1, 1, 1)x(1024, 1, 1), Context 1, Stream 7, Device 0, CC 7.5
    Section: GPU Speed Of Light Throughput
    ----------------------- ----------- ------------
    Metric Name             Metric Unit Metric Value
    ----------------------- ----------- ------------
    DRAM Frequency                  Ghz        10.03
    SM Frequency                    Ghz         1.03
    Elapsed Cycles                cycle         2461
    Memory Throughput                 %         2.50
    DRAM Throughput                   %         1.01
    Duration                         us         2.40
    L1/TEX Cache Throughput           %         7.95
    L2 Cache Throughput               %         2.50
    SM Active Cycles              cycle        75.44
    Compute (SM) Throughput           %         0.21
    ----------------------- ----------- ------------

    OPT   This kernel grid is too small to fill the available resources on this device, resulting in only 0.1 full      
          waves across all SMs. Look at Launch Statistics for more details.                                             

    Section: Launch Statistics
    -------------------------------- --------------- ---------------
    Metric Name                          Metric Unit    Metric Value
    -------------------------------- --------------- ---------------
    Block Size                                                  1024
    Function Cache Configuration                     CachePreferNone
    Grid Size                                                      1
    Registers Per Thread             register/thread              16
    Shared Memory Configuration Size           Kbyte           32.77
    Driver Shared Memory Per Block        byte/block               0
    Dynamic Shared Memory Per Block       byte/block               0
    Static Shared Memory Per Block        byte/block               0
    # SMs                                         SM              16
    Stack Size                                                  1024
    Threads                                   thread            1024
    # TPCs                                                         8
    Enabled TPC IDs                                              all
    Uses Green Context                                             0
    Waves Per SM                                                0.06
    -------------------------------- --------------- ---------------

    OPT   Est. Speedup: 93.75%                                                                                          
          The grid for this launch is configured to execute only 1 block, which is less than the GPU's 16               
          multiprocessors. This can underutilize some multiprocessors. If you do not intend to execute this kernel      
          concurrently with other workloads, consider reducing the block size to have at least one block per            
          multiprocessor or increase the size of the grid to fully utilize the available hardware resources. See the    
          Hardware Model (https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-hw-model)            
          description for more details on launch configurations.                                                        

    Section: Occupancy
    ------------------------------- ----------- ------------
    Metric Name                     Metric Unit Metric Value
    ------------------------------- ----------- ------------
    Block Limit SM                        block           16
    Block Limit Registers                 block            4
    Block Limit Shared Mem                block           16
    Block Limit Warps                     block            1
    Theoretical Active Warps per SM        warp           32
    Theoretical Occupancy                     %          100
    Achieved Occupancy                        %        60.94
    Achieved Active Warps Per SM           warp        19.50
    ------------------------------- ----------- ------------

    OPT   Est. Local Speedup: 39.06%                                                                                    
          The difference between calculated theoretical (100.0%) and measured achieved occupancy (60.9%) can be the     
          result of warp scheduling overheads or workload imbalances during the kernel execution. Load imbalances can   
          occur between warps within a block as well as across blocks of the same kernel. See the CUDA Best Practices   
          Guide (https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#occupancy) for more details on     
          optimizing occupancy.                                                                                         

    Section: GPU and Memory Workload Distribution
    -------------------------- ----------- ------------
    Metric Name                Metric Unit Metric Value
    -------------------------- ----------- ------------
    Average DRAM Active Cycles       cycle          190
    Total DRAM Elapsed Cycles        cycle        48128
    Average L1 Active Cycles         cycle        75.44
    Total L1 Elapsed Cycles          cycle        38648
    Average L2 Active Cycles         cycle       555.25
    Total L2 Elapsed Cycles          cycle         9424
    Average SM Active Cycles         cycle        75.44
    Total SM Elapsed Cycles          cycle        38648
    Average SMSP Active Cycles       cycle        53.05
    Total SMSP Elapsed Cycles        cycle       154592
    -------------------------- ----------- ------------

    OPT   Est. Speedup: 6.551%                                                                                          
          One or more L2 Slices have a much lower number of active cycles than the average number of active cycles.     
          Maximum instance value is 27.80% above the average, while the minimum instance value is 32.28% below the      
          average.     
```

`N = 7` is too small for GPU as indicated in the `ncu report`

When we change N to 1e6, it would saturate (DRAM) memory bandwidth 

Some insights for building intuition when analyzing the reports 

- DRAM Throughput (92.54%) suggests, it is close to being saturated, the program is entirely limited by how fast data moves from global RAM to the SMs

- L2 Cache Throughput (68.70%): it is lower than DRAM, confirming the data likely too large for L2 or isn't being reused (typical for streaming additions) 

- SM Frequency (1.05 GHz): The clock speed is relatively low, likely because the GPU is waiting on memory and doesn't need to boost clocks for compute


The L2 cache communicates with the SM in 128-byte cache lines (made of four 32-byte sectors).


`L2 Load Access Pattern` metric in Nsight can be a good spot to look at. Each line is divided into four 32-byte sectors.


## Streaming Kernel 

sigmoid and relu kernel can be found in `streaming_kernel.cu`, they achieve similar metrics in ncu reports


## Future improvement

### `Swizzling(block Scheduling)` 

thread blocks are launched in row-major order, by default. If your kernel is part of larger 2D operations, "block swizzling" can improve L2 hit rates by reordering block execution
so that spatially adjacent blocks (which may share data or L2 cache lines) run concurrently


### Kernel Fusion

The current implementation is often called `streaming kernel`

A streaming kernel is a workload where data is read from memory, processed once, and written back immediately without being reused.

This worths an independent note
