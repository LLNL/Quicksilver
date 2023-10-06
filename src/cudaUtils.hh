/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

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

#if defined(HAVE_SYCL) || defined(HAVE_OPENMP_TARGET)
#include <sycl/sycl.hpp>
extern sycl::queue sycl_device_queue; // global variable for device queue
#endif

#ifdef HAVE_OPENMP_TARGET
#ifdef USE_OPENMP_NO_GPU
#define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#else
#define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#define HAVE_UVM
#endif
#elif HAVE_SYCL
#define VAR_MEM MemoryControl::AllocationPolicy::UVM_MEM
#define HAVE_UVM
#else
#define VAR_MEM MemoryControl::AllocationPolicy::HOST_MEM
#endif

enum ExecutionPolicy
{
    cpu,
    gpuWithCUDA,
    gpuWithOpenMP
};

inline ExecutionPolicy getExecutionPolicy(int useGPU)
{
    ExecutionPolicy execPolicy = ExecutionPolicy::cpu;

    if (useGPU)
    {
#if defined(HAVE_SYCL)
        execPolicy = ExecutionPolicy::gpuWithCUDA;
#elif defined(HAVE_OPENMP_TARGET)
        execPolicy = ExecutionPolicy::gpuWithOpenMP;
#endif
    }
    return execPolicy;
}

template <class T>
inline void gpuMallocManaged(T **ptr, size_t size, unsigned int flags = 1 /*cudaMemAttachGlobal*/)
{
#if defined(HAVE_SYCL)
#ifdef UNIFIED_HOST
    *ptr = (T *)sycl::malloc_host(size, sycl_device_queue);
#elif defined(UNIFIED_DEVICE)
    *ptr = (T *)sycl::malloc_device(size, sycl_device_queue);
#else
    *ptr = (T *)sycl::malloc_shared(size, sycl_device_queue);
#endif
#endif
}

template <class T>
inline void gpuFree(T *ptr)
{
#if defined(HAVE_SYCL)
    sycl::free(ptr, sycl_device_queue);
#endif
}
#endif
