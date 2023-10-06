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

#include <sycl/sycl.hpp>
#include "cudaFunctions.hh"
#include "cudaUtils.hh"
#include <stdio.h>

namespace testname
{
#if HAVE_SYCL
#include "cudaFunctions.hh"
    void WarmUpKernel(sycl::nd_item<3> item_ct1)
    {
        int global_index = getGlobalThreadID(item_ct1);
        if (global_index == 0)
        {
        }
    }
#endif
}

#if defined(HAVE_SYCL)
void warmup_kernel()
{
    using namespace testname;
    sycl_device_queue.parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, 1), sycl::range<3>(1, 1, 1)),
        [=](sycl::nd_item<3> item_ct1)
        {
            testname::WarmUpKernel(item_ct1);
        });
    sycl_device_queue.wait();
}
#endif

#if defined(HAVE_SYCL)
int ThreadBlockLayout(sycl::range<3> &grid, sycl::range<3> &block,
                      int num_particles)
{
    int run_kernel = 1;
    const uint64_t max_block_size = 2147483647;
    // const uint64_t threads_per_block = 128;
    const uint64_t threads_per_block = 256;

    block[2] = threads_per_block;
    block[1] = 1;
    block[0] = 1;

    uint64_t num_blocks = num_particles / threads_per_block + ((num_particles % threads_per_block == 0) ? 0 : 1);

    if (num_blocks == 0)
    {
        run_kernel = 0;
    }
    else if (num_blocks <= max_block_size)
    {
        grid[2] = num_blocks;
        grid[1] = 1;
        grid[0] = 1;
    }
    else if (num_blocks <= max_block_size * max_block_size)
    {
        grid[2] = max_block_size;
        grid[1] = 1 + (num_blocks / max_block_size);
        grid[0] = 1;
    }
    else if (num_blocks <= max_block_size * max_block_size * max_block_size)
    {
        grid[2] = max_block_size;
        grid[1] = max_block_size;
        grid[0] = 1 + (num_blocks / (max_block_size * max_block_size));
    }
    else
    {
        printf("Error: num_blocks exceeds maximum block specifications. Cannot handle this case yet\n");
        run_kernel = 0;
    }

    return run_kernel;
}
#endif
