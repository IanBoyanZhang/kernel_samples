```
#define FLOAT4(a) *(float4*)(&(a))
```

type pruning, treating a memory address as a pointer to a different type than it was originally declared

- in standard C++, this can trigger "undefined behavior", unless use `-fno-strict-aliasing`, this might trigger compiler misoptimzation, note: `-fno-strict-aliasing` has other compiler optimization implications

- Vector types like `float4` often require 16-byte alignment


### Details

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

compute-sanitizer would gives below error

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

