#ifndef DECLAREMACRO_HH
#define DECLAREMACRO_HH

#ifdef HAVE_CUDA
    #define HOST_DEVICE __host__ __device__
    #define HOST_DEVICE_CUDA __host__ __device__
    #define HOST_DEVICE_CLASS 
    #define HOST_DEVICE_END
    #define DEVICE __device__
    #define DEVICE_END 
    //#define HOST __host__
    #define HOST_END 
    #define GLOBAL __global__
#elif HAVE_OPENMP_TARGET
    #define HOST_DEVICE _Pragma( "omp declare target" )
    #define HOST_DEVICE_CUDA
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
    #define HOST_DEVICE_CUDA
    #define HOST_DEVICE_CLASS
    #define HOST_DEVICE_END
    #define DEVICE
    #define DEVICE_END 
    //#define HOST
    #define HOST_END 
    #define GLOBAL
#endif

#endif
