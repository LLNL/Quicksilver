/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef HIPUTILS_HH
#define HIPUTILS_HH

#if defined(HAVE_HIP) || defined(HAVE_OPENMP_TARGET) 
#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#endif

#ifdef HAVE_OPENMP_TARGET
    #ifdef USE_OPENMP_NO_GPU
        #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
    #else
        #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
        #define HAVE_UVM
    #endif
#elif HAVE_HIP
    #define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
    #define HAVE_UVM
#else
    #define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif

enum ExecutionPolicy{ cpu, gpuWithHIP, gpuWithOpenMP };

inline ExecutionPolicy getExecutionPolicy( int useGPU )
{
    ExecutionPolicy execPolicy = ExecutionPolicy::cpu;

    if( useGPU )
    {
        #if defined (HAVE_HIP)
        execPolicy = ExecutionPolicy::gpuWithHIP;
        #elif defined (HAVE_OPENMP_TARGET)
        execPolicy = ExecutionPolicy::gpuWithOpenMP;
        #endif
    }
    return execPolicy;
}

template <class T>
inline void gpuMallocManaged(T** ptr, size_t size, unsigned int flags = hipHostMallocDefault)
{
     #if defined (HAVE_CUDA)
          hipMallocManaged(ptr,size);
     #elif defined (HAVE_HIP)
       #ifdef UNIFIED_HOST
          hipHostMalloc(ptr,size,flags);
       #elif defined(UNIFIED_DEVICE)
          hipMalloc(ptr,size);
       #else
          hipMallocManaged(ptr,size);
       #endif
     #endif

}

template <class T>
inline void gpuFree(T* ptr)
{
     #if defined (HAVE_CUDA)
          hipFree(ptr);
     #elif defined (HAVE_HIP)
          hipHostFree(ptr);
     #endif
}
#endif
