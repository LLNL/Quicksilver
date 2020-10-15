#ifndef AtomicMacro_HH_
#define AtomicMacro_HH_

#define USE_MACRO_FUNCTIONS 1

//Determine which atomics to use based on platform being compiled for
//
//If compiling with CUDA

#ifdef HAVE_OPENMP
    #define USE_OPENMP_ATOMICS
#elif HAVE_OPENMP_TARGET
    #define USE_OPENMP_ATOMICS
#endif

#ifdef HAVE_SYCL
#include <CL/sycl.hpp>
#include <cstdint>
#endif

// --------------------------------------------------
// Original Names            -> Inline function names
// --------------------------------------------------
// ATOMIC_WRITE( x, v )      -> ATOMIC_WRITE
// ATOMIC_UPDATE( x )        -> ATOMIC_INCREMENT
// ATOMIC_ADD( x, v )        -> ATOMIC_ADD
// ATOMIC_CAPTURE( x, v, p ) -> ATOMIC_FETCH_ADD
// --------------------------------------------------

#if defined (USE_MACRO_FUNCTIONS)

#define ATOMIC_CAPTURE( x, v, p )  ATOMIC_FETCH_ADD((x),(v),(p))
#define ATOMIC_UPDATE( x )         ATOMIC_INCREMENT((x))

#if defined(HAVE_SYCL)

static const cl::sycl::memory_order          order = cl::sycl::memory_order::relaxed;
static const cl::sycl::access::address_space space = cl::sycl::access::address_space::global_space;

template <typename T>
inline void ATOMIC_WRITE(T & x, T v) {
    //x = v;
}

template <typename T>
inline void ATOMIC_INCREMENT(T& x) {
    //atomicAdd( &x, 1 );
    T one{1};
    cl::sycl::atomic<T, space> y( (cl::sycl::multi_ptr<T, space>(&x)));
    cl::sycl::atomic_fetch_add(y, one, order);
}

template <typename T>
inline void ATOMIC_ADD(T& x, T v) {
    //atomicAdd( &x, v );
    cl::sycl::atomic<T, space> y( (cl::sycl::multi_ptr<T, space>(&x)));
    cl::sycl::atomic_fetch_add(y, v, order);
}

template <>
inline void ATOMIC_ADD(double& x, double v) {
    static_assert(sizeof(double) == sizeof(uint64_t), "Unsafe: double is not 64-bits");
    //atomicAdd( &x, v );
    cl::sycl::atomic<uint64_t, space> t( (cl::sycl::multi_ptr<uint64_t, space>( reinterpret_cast<uint64_t*>(&x)) ));
    uint64_t old_i = t.load(order);
    double   old_d;
    do {
      old_d = *reinterpret_cast<const double*>(&old_i);
      const double   new_d = old_d + v;
      const uint64_t new_i = *reinterpret_cast<const uint64_t *>(&new_d);
      if (t.compare_exchange_strong(old_i, new_i, order)) break;
    } while (true);
    // p = old_d;
}

template <typename T1, typename T2>
inline void ATOMIC_ADD(T1& x, T2 v) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    //atomicAdd( &x, v );
    T1 val = static_cast<T2>(v);
    cl::sycl::atomic<T1, space> y( (cl::sycl::multi_ptr<T1, space>(&x)));
    cl::sycl::atomic_fetch_add(y, val, order);
}

template <typename T>
inline void ATOMIC_FETCH_ADD(T& x, T v, T& p) {
    //p = atomicAdd( &x, v );
    cl::sycl::atomic<T, space> y( (cl::sycl::multi_ptr<T, space>(&x)));
    p = cl::sycl::atomic_fetch_add(y, v, order);
}

template <typename T1, typename T2>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T1& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    //p = atomicAdd( &x, v );
    cl::sycl::atomic<T1, space> y( (cl::sycl::multi_ptr<T1, space>(&x)));
    T1 val = static_cast<T2>(v);
    p = cl::sycl::atomic_fetch_add(y, val, order);
}

template <typename T1, typename T2, typename T3>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T3& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    static_assert( sizeof(T3) >= sizeof(T1), "Unsafe: small := large");
    //p = atomicAdd( &x, v );
    T1 val = static_cast<T2>(v);
    cl::sycl::atomic<T1, space> y( (cl::sycl::multi_ptr<T1, space>(&x)));
    p = cl::sycl::atomic_fetch_add(y, val, order);
}

#elif defined(HAVE_CUDA) && defined(__CUDA_ARCH__)

template <typename T>
inline void ATOMIC_WRITE(T & x, T v) {
    x = v;
}

template <typename T>
inline void ATOMIC_INCREMENT(T& x) {
    atomicAdd( &x, 1 );
}

template <typename T>
inline void ATOMIC_ADD(T& x, T v) {
    atomicAdd( &x, v );
}

template <typename T1, typename T2>
inline void ATOMIC_ADD(T1& x, T2 v) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    atomicAdd( &x, v );
}

template <typename T>
inline void ATOMIC_FETCH_ADD(T& x, T v, T& p) {
    p = atomicAdd( &x, v );
}

template <typename T1, typename T2>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T1& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    p = atomicAdd( &x, v );
}

template <typename T1, typename T2, typename T3>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T3& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    static_assert( sizeof(T3) >= sizeof(T1), "Unsafe: small := large");
    p = atomicAdd( &x, v );
}

#elif defined(USE_OPENMP_ATOMICS)

template <typename T>
inline void ATOMIC_WRITE(T & x, T v) {
    _Pragma("omp atomic write")
    x = v;
}

template <typename T>
inline void ATOMIC_INCREMENT(T& x) {
    _Pragma("omp atomic update")
    x++;
}

template <typename T>
inline void ATOMIC_ADD(T& x, T v) {
    _Pragma("omp atomic")
    x += v;
}

template <typename T1, typename T2>
inline void ATOMIC_ADD(T1& x, T2 v) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    _Pragma("omp atomic")
    x += v;
}

template <typename T>
inline void ATOMIC_FETCH_ADD(T& x, T v, T& p) {
    _Pragma("omp atomic capture")
    {p = x; x = x + v;}
}

template <typename T1, typename T2>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T1& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    _Pragma("omp atomic capture")
    {p = x; x = x + v;}
}

template <typename T1, typename T2, typename T3>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T3& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    static_assert( sizeof(T3) >= sizeof(T1), "Unsafe: small := large");
    _Pragma("omp atomic capture")
    {p = x; x = x + v;}
}

#else // SEQUENTIAL

template <typename T>
inline void ATOMIC_WRITE(T & x, T v) {
    x = v;
}

template <typename T>
inline void ATOMIC_INCREMENT(T& x) {
    x++;
}

template <typename T>
inline void ATOMIC_ADD(T& x, T v) {
    x += v;
}

template <typename T1, typename T2>
inline void ATOMIC_ADD(T1& x, T2 v) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    x += v;
}

template <typename T>
inline void ATOMIC_FETCH_ADD(T& x, T v, T& p) {
    {p = x; x = x + v;}
}

template <typename T1, typename T2>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T1& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    {p = x; x = x + v;}
}

template <typename T1, typename T2, typename T3>
inline void ATOMIC_FETCH_ADD(T1& x, T2 v, T3& p) {
    static_assert( sizeof(T1) >= sizeof(T2), "Unsafe: small += large");
    static_assert( sizeof(T3) >= sizeof(T1), "Unsafe: small := large");
    {p = x; x = x + v;}
}

#endif // BACKENDS

#else // ! USE_MACRO_FUNCTIONS

#if defined (HAVE_SYCL)

#error You must define USE_MACRO_FUNCTIONS with SYCL!

#elif defined (HAVE_CUDA)

    //If in a CUDA GPU section use the CUDA atomics
    #ifdef  __CUDA_ARCH__

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

#endif // BACKENDS

#endif // USE_MACRO_FUNCTIONS

#endif // AtomicMacro_HH_
