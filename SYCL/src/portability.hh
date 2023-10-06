#ifndef PORTABILITY_HH
#define PORTABILITY_HH

#ifdef CSTDINT_MISSING
#include <stdint.h>
#else
#include <cstdint>
#endif

#endif
