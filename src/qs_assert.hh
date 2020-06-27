#include <cstdio>
#include "cudaUtils.hh"

#ifdef __CUDA_ARCH__

#define qs_assert( cond) \
   do \
   { \
      if (!(cond)) \
      { \
        printf("ERROR\n"); \
      } \
   } while(0)

#elif defined(HAVE_SYCL)

#define qs_assert( cond) \
   do \
   { \
      if (!(cond)) \
      { \
          static const OPENCL_CONSTANT char format[] = "file=%s: line=%d ERROR\n"; \
          sycl::intel::experimental::printf(format,__FILE__,__LINE__); \
      } \
   } while(0)

#else

#define qs_assert( cond)                        \
   do \
   { \
      if (!(cond)) \
      { \
        printf("file=%s: line=%d ERROR\n",__FILE__,__LINE__); \
      } \
   } while(0)

#endif
