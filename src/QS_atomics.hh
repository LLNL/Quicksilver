#ifndef QS_ATOMICS_HH
#define QS_ATOMICS_HH

#include "gpuPortability.hh"

// Provides the following atomic functions:
// * QS::atomicWrite(a,b)          a=b
// * QS::atomicAdd(a,b)            a+=b
// * QS::atomicIncrement(a,b)      a++
// * QS::atomicCaptureAdd(a,b,c)   c=a; a+=b
// These all function correctly on hip(AMD), cuda, openMP, and openMP offload.
//
// There is one significant complication that we need to worry about
// when trying to provide device native implementations of atomics on
// hip and cuda.  Cuda doesn't allow function overloading based on
// __host__ or __device__ attributes.  If you have two functions with
// the same signature, one with __host__ (or undecorated, since
// functions are __host by default) and another with __device__, nvcc
// will produce an error that the function is multiply defined.  The
// solution to this problem is to wrap the overloaded functions in a
// check for the __CUDA_ARCH__ macro, which is defined only when
// compiling for the device.  See
// https://forums.developer.nvidia.com/t/overloading-host-and-device-function/29601
//
// On the other hand, hip seems to have no such problem managing
// functions that are overloaded on __host__ or __device__ attributes.
// Hence, we don't have to worry about checking for the device pass on
// a hip build.




// First, we need to provide some "built-in" atomic signatures that
// the CUDA API doesn't provide.  These should only be available in
// the device pass of a CUDA build.  HIP provides these signatures.
#if defined HAVE_CUDA && defined __CUDA_ARCH__

// atomicAdd for uint64_t:
// It is common that unsigned long and unsigned long long are both
// 64-bit integers.  In such cases, uint64_t may be defined as
// unsigned long.  Unfortunately, nvidia doesn't supply a version of
// atomicAdd that takes unsigned long arguments.  As long as unsigned
// long and unsigned long long are the same size, we can get away with
// this kind of nonsense.
static inline __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long),
		"type size mismatch");
  return ::atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}

// atomicExch for double:
// nvidia doesn't supply a version of atomicExch that takes doubles.
// So, we will roll our own with this somewhat evil hack.
static inline __device__ double atomicExch(double* address, double val)
{
  static_assert(sizeof(double) == sizeof(unsigned long long),
		"type size mismatch");
  return __longlong_as_double
    (
     ::atomicExch(reinterpret_cast<unsigned long long*>(address),
		  __double_as_longlong(val))
    );
}

#endif //#if defined HAVE_CUDA && defined __CUDA_ARCH__


namespace QS
{
  // First, the versions defined in terms of the native atomic
  // functions provided by CUDA and HIP.  

  // These get built when building for HIP (which QS assumes means AMD),
  // or the device pass of a CUDA build
  #if defined HAVE_HIP  || (defined HAVE_CUDA && defined __CUDA_ARCH__)

  template <typename T> static inline __device__
  void atomicWrite(T& aa, T bb)
  {
    atomicExch(&aa, bb);
  }

  template <typename T> static inline __device__
  void atomicAdd(T& aa, T bb)
  {
    ::atomicAdd(&aa, bb);
  }

  template <typename T> static inline __device__
  void atomicIncrement(T& aa)
  {
    ::atomicAdd(&aa, 1);
  }

  template <typename T> static inline __device__
  void atomicCaptureAdd(T& aa, T bb, T& cc)
  {
    cc = ::atomicAdd(&aa, bb);
  }

  #endif // #if defined HAVE_HIP  || (defined HAVE_CUDA && defined __CUDA_ARCH__)
  

  // Now the version defined in terms of omp atomic directives.  Note
  // that these apply to both CPU and GPU (i.e., target) code.  These
  // also supply implementations for CPU builds without openMP.
  // Obviously, these functions aren't actually atomic without
  // openMP. That's OK since without openMP quicksilver can't need
  // atomics on the CPU since it has no way run multiple threads in
  // the same address space.

  // These get build for everything *except* the device pass of a CUDA
  // build.
  #if ! (defined HAVE_CUDA && defined __CUDA_ARCH__) 

  template <typename T> static inline
  void atomicWrite(T& aa, T bb)
  {
    #pragma omp atomic write
    aa = bb;
  }

  template <typename T> static inline
  void atomicAdd(T& aa, T bb)
  {
    #pragma omp atomic
    aa += bb;
  }

  template <typename T> static inline
  void atomicIncrement(T& aa)
  {
    #pragma omp atomic update
    aa++;
  }
  
  template <typename T> static inline
  void atomicCaptureAdd(T& aa, T bb, T& cc)
  {
    #pragma omp atomic capture
    {cc = aa; aa += bb;}
  }

  #endif // #if ! (defined HAVE_CUDA && defined __CUDA_ARCH__) 
  
} // namespace QS

#endif // #ifndef QS_ATOMICS_HH
