/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef HIPFUNCTIONS_HH
#define HIPFUNCTIONS_HH

#include "hipUtils.hh"
#include "DeclareMacro.hh"

#if defined (HAVE_HIP)
void warmup_kernel();
int ThreadBlockLayout( dim3 &grid, dim3 &block, int num_particles );
DEVICE 
#endif

#if defined (HAVE_HIP)
inline DEVICE
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

inline DEVICE
int getLocalThreadID()
{

    int threadID =  threadIdx.z * ( blockDim.x * blockDim.y ) +
                    threadIdx.y * blockDim.x +
                    threadIdx.x;
    return threadID;
}
#endif


#endif
