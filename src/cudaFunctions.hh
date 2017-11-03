#ifndef CUDAFUNCTIONS_HH
#define CUDAFUNCTIONS_HH

#include "cudaUtils.hh"
#include "DeclareMacro.hh"

#if defined (HAVE_CUDA)
void warmup_kernel();
int ThreadBlockLayout( dim3 &grid, dim3 &block, int num_particles );
DEVICE 
int getGlobalThreadID();
#endif

#endif
