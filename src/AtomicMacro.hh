/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/



//Determine which atomics to use based on platform being compiled for
//

#ifndef ATOMICS_HD
#define ATOMICS_HD

//If compiling with HIP
#include <thread>
#include <mutex>
#include <algorithm>
#ifdef HAVE_HIP
   #include <hip/hcc_detail/hip_runtime.h>
#endif


inline __device__ double ull2d(const unsigned long long& val){
  return *((double*) &val);
}


#ifdef HAVE_CUDA
  #include <cuda.h>
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

  #else
      inline __device__ double atomicAdd(double* pointer, double val)
      {
          //A workaround dealing with the fact that atomic doubles don't work with all versions of CUDA. 
          unsigned long long int* pointer_as_p2ull = (unsigned long long int*)pointer;
          unsigned long long int old = *pointer_as_p2ull, check_value;
          do {
              check_value = old;
              old = atomicCAS(pointer_as_p2ull, check_value, __double_as_longlong(val + ull2d(check_value)));
          } while (check_value != old);
          return ull2d(old);
      };
  #endif

#endif


#ifdef HAVE_OPENMP
    #define USE_OPENMP_ATOMICS
#elif HAVE_OPENMP_TARGET
    #define USE_OPENMP_ATOMICS
#endif


#if defined (HAVE_HIP)


    //If in a HIP GPU section use the HIP atomics
    #ifdef  __HIP_DEVICE_COMPILE__

        //Currently not atomic here. But its only used when it does not necissarially need to be atomic.
        #define ATOMIC_WRITE( x, v ) \
            x = v;          

        #define ATOMIC_ADD( x, v ) \
            atomicAdd( &x, v );
        
        #define ATOMIC_UPDATE( x ) \
            atomicAdd( &x, 1 );

        #define ATOMIC_CAPTURE( x, v, p ) \
            p = atomicAdd( &x, v );
    //If in a CPU OpenMP section use the OpenMP atomics
    #elif defined (USE_OPENMP_ATOMICS)
        #define ATOMIC_WRITE( x, v ) \
            _Pragma("omp atomic write") \
            x = v;

        #define ATOMIC_ADD( x, v ) \
            _Pragma("omp atomic") \
            x += v;

        #define ATOMIC_UPDATE( x ) \
            _Pragma("omp atomic update") \
            x++;

        #define ATOMIC_CAPTURE( x, v, p ) \
            _Pragma("omp atomic capture") \
            {p = x; x = x + v;}

    //If in a serial section, no need to use atomics
    #else
        #define ATOMIC_WRITE( x, v ) \
            x = v;

        #define ATOMIC_UPDATE( x ) \
            x++;

        #define ATOMIC_ADD( x, v ) \
            x += v;

        #define ATOMIC_CAPTURE( x, v, p ) \
            {p = x; x = x + v;}

    #endif

//If in a OpenMP section use the OpenMP atomics
#elif defined (USE_OPENMP_ATOMICS)
    #define ATOMIC_WRITE( x, v ) \
        _Pragma("omp atomic write") \
        x = v;

    #define ATOMIC_ADD( x, v ) \
        _Pragma("omp atomic") \
        x += v;

    #define ATOMIC_UPDATE( x ) \
        _Pragma("omp atomic update") \
        x++;

        #define ATOMIC_CAPTURE( x, v, p ) \
            _Pragma("omp atomic capture") \
            {p = x; x = x + v;}

//If in a serial section, no need to use atomics
#else
    #define ATOMIC_WRITE( x, v ) \
        x = v;

    #define ATOMIC_UPDATE( x ) \
        x++;

    #define ATOMIC_ADD( x, v ) \
        x += v;

    #define ATOMIC_CAPTURE( x, v, p ) \
        {p = x; x = x + v;}
#endif

#endif
