#ifndef CUDAUTILS_HH
#define CUDAUTILS_HH

#if defined(HAVE_CUDA) || defined(HAVE_OPENMP_TARGET)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

#if defined(HAVE_SYCL)
#include <CL/sycl.hpp>
extern sycl::queue q; // global variable for device queue
#else
#define SYCL_EXTERNAL
#endif

#ifdef HAVE_OPENMP_TARGET
    #ifdef USE_OPENMP_NO_GPU
        #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
    #else
        #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
        #define HAVE_UVM
    #endif
#elif HAVE_CUDA
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
    #define HAVE_UVM
#elif HAVE_SYCL
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
    #define HAVE_UVM
#else
    #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif

enum ExecutionPolicy{ cpu, gpuWithCUDA, gpuWithOpenMP, SYCL };

inline ExecutionPolicy getExecutionPolicy( int useGPU )
{
    ExecutionPolicy execPolicy = ExecutionPolicy::cpu;

    if( useGPU )
    {
        #if defined (HAVE_CUDA)
        execPolicy = ExecutionPolicy::gpuWithCUDA;
        #elif defined (HAVE_OPENMP_TARGET)
        execPolicy = ExecutionPolicy::gpuWithOpenMP;
        #elif defined (HAVE_SYCL)
        execPolicy = ExecutionPolicy::SYCL;
        #endif
    }
    return execPolicy;
}
#endif
