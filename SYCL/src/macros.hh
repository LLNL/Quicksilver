#ifndef MACROS_HH
#define MACROS_HH

#include "qs_assert.hh"
#include <algorithm>

#define MC_CALLOC(A, N1, TYPE) if ( N1 ) { A = (TYPE*) calloc((N1), sizeof(TYPE)); } else { A = NULL; }
#define MC_MALLOC(A, N1, TYPE) if ( N1 ) { A = (TYPE*) malloc((N1)*sizeof(TYPE)); } else { A = NULL; }
#define MC_NEW_ARRAY(A,N1,TYPE)  if ( N1 ) { A = new TYPE[N1]; } else { A = NULL; }
#define MC_REALLOC(a, b, c) {qs_assert(false); }
#define MC_FREE(A)          if (A != NULL) { free(A) ; A = NULL ; }
#define MC_DELETE(A)        if (A != NULL) { delete A ; A = NULL ; }
#define MC_DELETE_ARRAY(A)  if (A != NULL) { delete [] A ; A = NULL ; }
#define MC_MEMCPY(a, b, c)  {qs_assert(false); }
#define MC_FABS(x) ( (x) < 0 ? -(x) : (x) )


#define MC_Fatal_Jump(...) {qs_assert(false); }

//#define MC_MIN(a, b)       {std::min(a,b)}
#define MC_MIN(a, b)       { ((a < b) ? a : b) } 

// If not compiled with OpenMP, define stub OpenMP
// function that will work for the code.
#ifdef HAVE_OPENMP
    #include <omp.h>
#else
    #include <iostream>
    #include <cstdlib>
    #define omp_get_thread_num()   0
    #define omp_get_max_threads()  1
    #define omp_get_num_procs()    1
#endif
#else
#endif

#if defined(HAVE_OPENMP) && defined(HAVE_DEBUG)
#define MC_VERIFY_THREAD_ZERO MC_Verify_Thread_Zero(__FILE__, __LINE__);
#else
#define MC_VERIFY_THREAD_ZERO 
#endif

#ifdef USE_PRINT_DEBUG
#define PRINT_DEBUG printf("FILE: %s\tLINE: %d\n", __FILE__, __LINE__ )
#else
#define PRINT_DEBUG
#endif
