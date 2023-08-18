#ifndef CUDAUTILS_HH
#define CUDAUTILS_HH

#if defined(HAVE_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#endif

enum ExecutionPolicy{ cpu, gpuNative, gpuWithOpenMP };

inline ExecutionPolicy getExecutionPolicy( int useGPU )
{
    ExecutionPolicy execPolicy = ExecutionPolicy::cpu;

    if( useGPU )
    {
        #if defined HAVE_CUDA || defined HAVE_HIP 
        execPolicy = ExecutionPolicy::gpuNative;
        #elif defined (HAVE_OPENMP_TARGET)
        execPolicy = ExecutionPolicy::gpuWithOpenMP;
        #endif
    }
    return execPolicy;
}
#endif
