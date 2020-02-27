/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#if HAVE_HIP
    #include "hip/hip_runtime.h"
#endif
#include "hipFunctions.hh"
#include "hipUtils.hh"
#include <stdio.h> 

namespace testname
{
#if HAVE_HIP
#include "hipFunctions.hh"
    __global__ void WarmUpKernel(int a)
    {
        int global_index = getGlobalThreadID();
        if( global_index == 0)
        {
        }
    }
#endif
}

#if defined (HAVE_HIP)
void warmup_kernel()
{
        using namespace testname;
        hipLaunchKernelGGL((WarmUpKernel), dim3(1), dim3(1), 0, 0, 1 );
        hipDeviceSynchronize();
}
#endif

#if defined (HAVE_HIP)
int ThreadBlockLayout( dim3 &grid, dim3 &block, int num_particles )
{
    int run_kernel = 1;
    const uint64_t max_block_size = 65535;
    //const uint64_t threads_per_block = 128;
    const uint64_t threads_per_block = 64;
    
    block.x = threads_per_block;
    block.y = 1;
    block.z = 1;

    uint64_t num_blocks = num_particles / threads_per_block + ( ( num_particles%threads_per_block == 0 ) ? 0 : 1 );

    if( num_blocks == 0 )
    {
        run_kernel = 0;
    }
    else if( num_blocks <= max_block_size )
    {
        grid.x = num_blocks;
        grid.y = 1;
        grid.z = 1;
    } 
    else if( num_blocks <= max_block_size*max_block_size )
    {
        grid.x = max_block_size;
        grid.y = 1 + (num_blocks / max_block_size );
        grid.z = 1;
    }
    else if( num_blocks <= max_block_size*max_block_size*max_block_size )
    {
        grid.x = max_block_size;
        grid.y = max_block_size;
        grid.z = 1 + (num_blocks / (max_block_size*max_block_size));
    }
    else
    {
        printf("Error: num_blocks exceeds maximum block specifications. Cannot handle this case yet\n");
        run_kernel = 0;
    }

    return run_kernel;
} 
#endif

