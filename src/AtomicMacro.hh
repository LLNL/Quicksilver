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

#if defined(HAVE_CUDA) && defined(__CUDA_ARCH__)

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

#if defined (HAVE_CUDA)

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
