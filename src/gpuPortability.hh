#ifndef MALLOCMANAGED_HH
#define MALLOCMANAGED_HH

#if defined __CUDACC__ || defined TARGET_NVIDIA
    #define MM_DO_CUDA
    #define HAVE_UVM
#elif defined __HIPCC__ || defined TARGET_AMD
    #define MM_DO_HIP
    #define HAVE_UVM
    #define __HIP_PLATFORM_AMD__
#endif

#if defined HAVE_UVM
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#else
    #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif  


#ifdef MM_DO_CUDA
#include <cuda.h>
#endif

#ifdef MM_DO_HIP
#include <hip/hip_runtime_api.h>
#endif


inline int mallocManaged(void** ptr, size_t size)
{
  int rv = 1;
#if defined MM_DO_CUDA
  rv = !(cudaSuccess == cudaMallocManaged(ptr, size, cudaMemAttachGlobal));
#elif defined MM_DO_HIP 
  rv = !(hipSuccess == hipMallocManaged(ptr, size, hipMemAttachGlobal));
#endif
return rv;
}

inline int freeManaged(void* ptr)
{
  int rv = 1;
#if defined MM_DO_CUDA
    rv = !(cudaSuccess == cudaFree(ptr));
#elif defined MM_DO_HIP
    rv = !(hipSuccess == hipFree(ptr));
#endif
  return rv;
}


#undef MM_DO_CUDA
#undef MM_DO_HIP

#endif
