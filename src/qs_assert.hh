#include <cstdio>

#ifdef __CUDA_ARCH__
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
