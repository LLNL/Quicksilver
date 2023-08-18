#ifndef GPUPORTABILITY_HH
#define GPUPORTABILITY_HH

#if defined __CUDACC__ || defined TARGET_NVIDIA
    #define __DO_CUDA
    #define __PREFIX cuda
    #define HAVE_UVM
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
#elif defined __HIPCC__ || defined TARGET_AMD
    #define __DO_HIP
    #define __PREFIX hip
    #define HAVE_UVM
    #define __HIP_PLATFORM_AMD__
    #include <hip/hip_runtime.h>
#else
    #define __PREFIX invalid
#endif

#if defined HAVE_CUDA || defined HAVE_HIP
    #define GPU_NATIVE
#endif


#ifdef __DO_CUDA
#endif

#ifdef __DO_HIP
#endif

#if defined HAVE_UVM
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#else
    #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif  

#define CONCAT_(A, B) A ## B
#define CONCAT(A1, B1) CONCAT_(A1, B1)

#define gpuMallocManaged      CONCAT(__PREFIX, MallocManaged)
#define gpuFree               CONCAT(__PREFIX, Free)
#define gpuDeviceSynchronize  CONCAT(__PREFIX, DeviceSynchronize)
#define gpuGetDeviceCount     CONCAT(__PREFIX, GetDeviceCount)
#define gpuSetDevice          CONCAT(__PREFIX, SetDevice)
#define gpuPeekAtLastError    CONCAT(__PREFIX, PeekAtLastError)


#undef __DO_CUDA
#undef __DO_HIP

#endif // #ifndef GPUPORTABILITY_HH
