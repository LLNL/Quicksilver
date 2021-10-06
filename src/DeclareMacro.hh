/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef DECLAREMACRO_HH
#define DECLAREMACRO_HH

#ifdef HAVE_HIP
    #include "hip/hip_runtime.h"
    #define HOST_DEVICE __host__ __device__
    #define HOST_DEVICE_HIP __host__ __device__
    #define HOST_DEVICE_CLASS 
    #define HOST_DEVICE_END
    #define DEVICE __device__
    #define DEVICE_END 
    //#define HOST __host__
    #define HOST_END 
    #define GLOBAL __global__
#elif HAVE_OPENMP_TARGET
    #define HOST_DEVICE _Pragma( "omp declare target" )
    #define HOST_DEVICE_HIP
    #define HOST_DEVICE_CLASS _Pragma( "omp declare target" )
    #define HOST_DEVICE_END _Pragma("omp end declare target")
    //#define HOST_DEVICE #pragma omp declare target
    //#define HOST_DEVICE_END #pragma omp end declare target
    //#define DEVICE #pragma omp declare target 
    //#define DEVICE_END #pragma omp end declare target
    //#define HOST 
    #define HOST_END 
    #define GLOBAL
#else
    #define HOST_DEVICE
    #define HOST_DEVICE_HIP
    #define HOST_DEVICE_CLASS
    #define HOST_DEVICE_END
    #define DEVICE
    #define DEVICE_END 
    //#define HOST
    #define HOST_END 
    #define GLOBAL
#endif

#define NUM_TALLIES 8
#endif
