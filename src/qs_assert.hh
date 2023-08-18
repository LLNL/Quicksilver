#include <cstdio>

#if defined HAVE_HIP
#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>
#include <hip/hip_runtime.h>
#endif

#if defined __CUDA_ARCH__ || defined __HIP_DEVICE_COMPILE__
#define qs_assert( cond) \
   do \
   { \
      if (!(cond)) \
      { \
        printf("ERROR\n"); \
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
