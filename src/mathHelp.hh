#ifndef MATH_HELP_HH
#define MATH_HELP_HH

#ifdef HAVE_SYCL
#include <CL/sycl.hpp>
#define SIN  sycl::sin
#define COS  sycl::cos
#define SQRT sycl::sqrt
#define LOG  sycl::log
#else
#include <cmath>
#define SIN  sin
#define COS  cos
#define SQRT sqrt
#define LOG  log
#endif

#endif // MATH_HELP_HH
