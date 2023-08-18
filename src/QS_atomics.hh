#ifndef QS_ATOMICS_HH
#define QS_ATOMICS_HH

#include "gpuPortability.hh"

// Atomic write (a=b), add (a+=b), increment (a++), and captureAdd (c=a; a+=b)


// CUDA doesn't allow function overloading based on __host__ or
// __device__ attributes.  If you have two functions with the same
// signature, one with __host__ (or undecorated, since functions are
// __host by default) and another with __device__, nvcc will produce
// an error that the function is multiply defined.  The solution to
// this problem is to wrap the device code in a check for the
// __CUDA_ARCH__ macro, which is defined only when compiling for the
// device.  See
// https://forums.developer.nvidia.com/t/overloading-host-and-device-function/29601


#if defined HAVE_CUDA && defined __CUDA_ARCH__

static inline __device__ uint64_t atomicAdd(uint64_t* address, uint64_t val)
{
  static_assert(sizeof(uint64_t) == sizeof(unsigned long long),
		"type size mismatch");
  return ::atomicAdd(reinterpret_cast<unsigned long long*>(address), val);
}

static inline __device__ double atomicExch(double* address, double val)
{
  static_assert(sizeof(double) == sizeof(unsigned long long),
		"type size mismatch");
  return ::atomicExch(reinterpret_cast<unsigned long long*>(address),
		      __double_as_longlong(val));
}

#endif


namespace QS
{
  // These are the versions defined in terms of the native atomic
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
  

  // These are the version defined in terms of omp atomic directives.
  // Note that these apply to both CPU and GPU (i.e., target) code.

  // These get build for everything *except* the device pass of a CUDA
  // build
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
