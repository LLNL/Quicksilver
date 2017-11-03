#include "cudaFunctions.hh"
#include "cudaUtils.hh"
#include <stdio.h> 

namespace
{
#if HAVE_CUDA
#include "cudaFunctions.hh"
    __global__ void WarmUpKernel()
    {
        int global_index = getGlobalThreadID();
        if( global_index == 0)
        {
        }
    }
#endif
}

#if defined (HAVE_CUDA)
void warmup_kernel()
{
        WarmUpKernel<<<1, 1>>>();
        cudaDeviceSynchronize();
}
#endif

#if defined (HAVE_CUDA)
int ThreadBlockLayout( dim3 &grid, dim3 &block, int num_particles )
{
    int run_kernel = 1;
    const uint64_t max_block_size = 65535;
    const uint64_t threads_per_block = 128;
    
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

#if defined (HAVE_CUDA)
DEVICE 
int getGlobalThreadID()
{
    int blockID  =  blockIdx.x + 
                    blockIdx.y * gridDim.x + 
                    blockIdx.z * gridDim.x * gridDim.y;

    int threadID =  blockID * (blockDim.x * blockDim.y * blockDim.z) +
                    threadIdx.z * ( blockDim.x * blockDim.y ) + 
                    threadIdx.y * blockDim.x +
                    threadIdx.x;
    return threadID;
}
#endif
