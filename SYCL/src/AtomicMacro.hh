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

// Determine which atomics to use based on platform being compiled for
//

#ifndef ATOMICS_HD
#define ATOMICS_HD

#include <sycl/sycl.hpp>
#include <thread>
#include <mutex>
#include <algorithm>

inline double ull2d(const unsigned long long &val)
{
    return *((double *)&val);
}

#if defined(HAVE_SYCL)

// If in a CUDA GPU section use the CUDA atomics
#ifdef __SYCL_DEVICE_ONLY__

/// Atomically add the value operand to the value at the addr and assign the
/// result to the value at addr.
/// \param [in, out] addr The pointer to the data.
/// \param operand The value to add to the value at \p addr.
/// \param memoryOrder The memory ordering used.
/// \returns The value at the \p addr before the call.
template <typename T,
          sycl::access::address_space addressSpace =
              sycl::access::address_space::global_space,
          sycl::memory_order memoryOrder = sycl::memory_order::relaxed,
          sycl::memory_scope memoryScope = sycl::memory_scope::device>
inline T atomic_fetch_add(T *addr, T operand)
{
    auto atm =
        sycl::atomic_ref<T, memoryOrder, memoryScope, addressSpace>(addr[0]);
    return atm.fetch_add(operand);
}

// Currently not atomic here. But its only used when it does not necissarially need to be atomic.
#define ATOMIC_WRITE(x, v) \
    x = v;

#define ATOMIC_ADD(x, v)                                                         \
    {                                                                            \
        using Ty = std::remove_reference<decltype(x)>::type;                     \
        atomic_fetch_add<Ty, sycl::access::address_space::generic_space>(&x, v); \
    }

#define ATOMIC_UPDATE(x)                                                           \
    {                                                                              \
        using Ty = std::remove_reference<decltype(x)>::type;                       \
        Ty inc = 1;                                                                \
        atomic_fetch_add<Ty, sycl::access::address_space::generic_space>(&x, inc); \
    }

#define ATOMIC_CAPTURE(x, v, p)                                                  \
    {                                                                            \
        using Ty = typename std::remove_reference<decltype(x)>::type;            \
        \	
    p = atomic_fetch_add<Ty, sycl::access::address_space::generic_space>(&x, v); \
    }
// If in a CPU OpenMP section use the OpenMP atomics
#elif defined(USE_OPENMP_ATOMICS)
#define ATOMIC_WRITE(x, v)      \
    _Pragma("omp atomic write") \
        x = v;

#define ATOMIC_ADD(x, v)  \
    _Pragma("omp atomic") \
        x += v;

#define ATOMIC_UPDATE(x)         \
    _Pragma("omp atomic update") \
        x++;

#define ATOMIC_CAPTURE(x, v, p)   \
    _Pragma("omp atomic capture") \
    {                             \
        p = x;                    \
        x = x + v;                \
    }

// If in a serial section, no need to use atomics
#else
#define ATOMIC_WRITE(x, v) \
    x = v;

#define ATOMIC_UPDATE(x) \
    x++;

#define ATOMIC_ADD(x, v) \
    x += v;

#define ATOMIC_CAPTURE(x, v, p) \
    {                           \
        p = x;                  \
        x = x + v;              \
    }

#endif

// If in a OpenMP section use the OpenMP atomics
#elif defined(USE_OPENMP_ATOMICS)
#define ATOMIC_WRITE(x, v)      \
    _Pragma("omp atomic write") \
        x = v;

#define ATOMIC_ADD(x, v)  \
    _Pragma("omp atomic") \
        x += v;

#define ATOMIC_UPDATE(x)         \
    _Pragma("omp atomic update") \
        x++;

#define ATOMIC_CAPTURE(x, v, p)   \
    _Pragma("omp atomic capture") \
    {                             \
        p = x;                    \
        x = x + v;                \
    }

// If in a serial section, no need to use atomics
#else
#define ATOMIC_WRITE(x, v) \
    x = v;

#define ATOMIC_UPDATE(x) \
    x++;

#define ATOMIC_ADD(x, v) \
    x += v;

#define ATOMIC_CAPTURE(x, v, p) \
    {                           \
        p = x;                  \
        x = x + v;              \
    }
#endif

#endif
