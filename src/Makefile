# Makefile for Quicksilver

# Quicksilver is a relatively easy to build code with no external
# dependencies (except MPI and OpenMP).  You should be able to build
# Quicksilver on nearly any system by customizing the values of only
# four variables:
#
#    CXX        The name of the C++ compiler (with path if necessary)
#               Quicksilver uses C++11 features, so a C++11 compliant 
#               compiler should be used.
#
#    CXXFLAGS   Command line switches to pass to the C++ compiler
#               when compiling objects *and* when linking the executable.
#
#    CPPFLAGS   Command line switches to pass to the compiler *only*
#               when compiling objects
#
#    LDFLAGS    Command line switches to pass to the compiler *only*
#               when linking the executable
#
# Any other variable you may see in this Makefile (such as OPTFLAGS,
# OPENMP_FLAGS, CUDA_PATH, etc) are defined for convienience and clarity
# only.  They do not appear in the build recipes.  Only the four
# variables above are used in the build recipes.
#
#
# 
#
# Quicksilver recognizes a number of pre-processor macros that
# enable or disable various code features such as MPI, OpenMP, etc.
# The following pre-processor DEFINES are recognized:
#
# -DHAVE_MPI        
#                   Define HAVE_MPI to enable MPI feartures in
#                   Quicksilver.  If this is not defined, the MPI
#                   functions will be replaced with stub implmentations
#                   and the code will run on a single "rank".
#
# -DHAVE_ASYNC_MPI  
#                   Define this if your MPI has support for non-blocking
#                   collectives.  (Quicksilver will use MPI_Iallreduce 
#                   in the test-for-done algorithm.)
#
# -DHAVE_OPENMP     
#                   Use this define to generate a code which uses  OpenMP
#                   threads.  It will also be necessary to add the appropriate
#                   compiler flags to CXXFLAGS and LDFLAGS, which vary
#                   by compiler, such as '-qopenmp -pthread' for Intel, or
#                   '-fopenmp' for Gnu.  Defining HAVE_OPENMP will use
#                   only features in OpenMP 3.x.
#
# -DHAVE_OPENMP_TARGET
#                   Use this define to generate OpenMP 4.5 code,
#                   including code targeting GPUs
#
# -DDISABLE_TIMERS  Quicksilver uses built-in high resolution timers to
#                   track performance of important high level functions.
#                   To disable the internal timers (and the timing reports)
#                   define DISABLE_TIMERS.  This can be useful when using a
#                   profiler or other external timing mechanism.
#
# -DCSTDINT_MISSING 
#                   Define this if <cstdint> is not available.
#                   In this case the include file <stdint.h> will be used 
#                   as an alternative.  This was found to be necessary with 
#                   PGI and some Clang compilers.
#
# -DCHRONO_MISSING  
#                   Define this if <chrono>  is not available.
#                   Normally we use the C++11 high resolution timers for 
#                   internal timing.  This requires the include of <chrono>
#                   However, if this is not available, then one may specify 
#                   this -D option, and the MPI high resolution timer will 
#                   be used as an alternative.  This was found to be necessary
#                   with some Clang compilers, some older Gnu compilers on BG/Q
#                   and older Intel compilers.
#         
# -DUSE_NVTX        Define this for some extra NVProf profiling information.
#                   It will create regions that can be visualized in NVVP.
#
# -DUSE_OPENMP_NO_GPU 
#                   Define this for runs with OPENMP_TARGET but when you do not
#                   want to target GPU devices only the CPU
#
# -DEXPONENTIAL_TALLY
#                   Define this to run Cycle Tracking with an exponential
#                   cell-based tally, in order to partially mimic photon
#                   transport problems.
#  
# ------------------------------------------------------------------------------

SHELL = /bin/bash

# Set your desired C++ compiler and any necessary flags.
# The interpretation of each of these four variables is described above.
# A number of examples for machines we regularly run on are given below.
CXX = 
CXXFLAGS =
CPPFLAGS = 
LDFLAGS =

###############################################################################
# Very simple GCC build with OpenMP but without MPI.  
# This works on a Macbook (if gcc is installed)
###############################################################################
#
#OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp 
#OPENMP_LDFLAGS = -fopenmp
#OPTFLAGS = -g -O2
#
#CXX=g++
#CXXFLAGS = -std=c++11  $(OPTFLAGS) -Wpedantic
#CPPFLAGS = $(OPENMP_FLAGS)
#LDFLAGS = $(OPENMP_LDFLAGS) 


###############################################################################
### GCC -- with MPI and OpenMP 
###############################################################################
#OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp
#OPENMP_LDFLAGS = -fopenmp
#MPI_FLAGS = -DHAVE_MPI
#OPTFLAGS = -g -O2
#
#CXX=mpicxx
#CXXFLAGS = -std=c++11 $(OPTFLAGS) -Wpedantic
#CPPFLAGS = $(MPI_FLAGS) $(OPENMP_FLAGS)
#LDFLAGS = $(OPENMP_LDFLAGS) 


###############################################################################
# LLNL LC BG/Q Comilers                                                       #
###############################################################################
### BGQ GNU
#OPTFLAGS = -g -O2
##
#CXX=/usr/local/tools/compilers/ibm/mpicxx-4.8.4
#CXXFLAGS = -std=c++11 $(OPTFLAGS) 
#CPPFLAGS = -DCHRONO_MISSING -DHAVE_MPI -DHAVE_OPENMP -fopenmp
#LDFLAGS  = -fopenmp


###############################################################################
# OpenMP 4.5 on LLNL CORAL EA nodes
###############################################################################
## Choose one Cuda path
##CUDA_PATH = /usr/local/cuda-8.0
#CUDA_PATH = /usr/tcetmp/packages/cuda-9.0.176

## Choose one of these compilers
#CXX=mpiclang++-gpu
#CXX=mpixlC-gpu

## Uncomment next line to run on CPU only (disable GPUs)
#OPENMP_OFFLOAD_FLAGS = -DUSE_OPENMP_NO_GPU 

#OPENMP_OFFLOAD_FLAGS += -DHAVE_OPENMP_TARGET -mcpu=power8  
#OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp ${OPENMP_OFFLOAD_FLAGS}
#OPTFLAGS = -O2

#CUDA_FLAGS = -I${CUDA_PATH}/include/
#CUDA_LDFLAGS = -L${CUDA_PATH}/lib64/ -lcuda -lcudart

#CXXFLAGS = -std=c++11 $(OPTFLAGS)
#CPPFLAGS = -DHAVE_MPI $(OPENMP_FLAGS) $(CUDA_FLAGS) 
#LDFLAGS  = $(CUDA_LDFLAGS)


###############################################################################
# Cuda on LLNL CORAL EA nodes
###############################################################################
## Choose one Cuda path
##CUDA_PATH = /usr/local/cuda-8.0
#CUDA_PATH = /usr/tcetmp/packages/cuda-9.0.176

#HOST_COMPILER = /usr/tce/packages/spectrum-mpi/spectrum-mpi-2017.04.03-xl-beta-2017.09.13/bin/mpixlC

#OPTFLAGS = -O2 
## Version below for debugging
##OPTFLAGS = -DUSE_NVTX -g -G -lineinfo -O0

#CUDA_FLAGS = -I${CUDA_PATH}/include/
#CUDA_LDFLAGS = -L${CUDA_PATH}/lib64/ -lcuda -lcudart
#
#CXX=$(CUDA_PATH)/bin/nvcc
#CXXFLAGS = -DHAVE_CUDA -std=c++11 $(OPTFLAGS) -Xptxas -v 
#CXXFLAGS += -gencode=arch=compute_60,code=\"sm_60,compute_60\" 
#CXXFLAGS += --compiler-bindir=$(HOST_COMPILER)
#CPPFLAGS = -x cu -dc -DHAVE_MPI -DHAVE_ASYNC_MPI
#LDFLAGS  = $(CUDA_LDFLAGS) 
##LDFLAGS += ${CUDA_PATH}/lib64/libnvToolsExt.so




###############################################################################
# LLNL TOSS GCC + OpenMP (mvapich 2 - version 1.7) [cab]
###############################################################################
#OPTFLAGS = -g -O2 
#OPENMP_FLAGS = -DHAVE_OPENMP -fopenmp 
#OPENMP_LDFLAGS = -fopenmp
#
#CXX = /usr/apps/gnu/4.9.3/bin/mpig++
#CXXFLAGS = -std=c++0x $(OPTFLAGS) -mpi=mvapich2-gnu-1.7 
#CPPFLAGS = -DHAVE_MPI $(OPENMP_FLAGS)
#LDFLAGS = $(OPENMP_LDFLAGS)

###############################################################################
# LLNL TOSS Intel + OpenMP (mvapich 2 - version 2.1) [quartz] 
###############################################################################
#OPENMP_FLAGS = -DHAVE_OPENMP -qopenmp 
#OPENMP_LDFLAGS = -qopenmp 
#OPTFLAGS = -g -O2
#
#CXX=/usr/local/bin/mpiicpc-17.0.174
#CXXFLAGS = -std=c++11 -mpi=mvapich2-intel-2.1 -DHAVE_MPI $(OPENMP_FLAGS)
#CXXFLAGS += -wd1128 -wd64 -wd21
#LDFLAGS  = $(OPENMP_LDFLAGS)


###############################################################################
# LLNL TOSS Clang (cab)
###############################################################################
#CLANGPATH = /usr/global/tools/clang/chaos_5_x86_64_ib/clang-omp-3.5.0
#OPTFLAGS = -g -O2 
#
#CXX=${CLANGPATH}/bin/mpiclang++
#CXXFLAGS = -std=c++11 $(OPTFLAGS)
#CPPFLAGS = -DHAVE_MPI
#LDFLAGS  = -Wl,-rpath,${CLANGPATH}/lib


###############################################################################
# Trinity Compilers                                                           #
#                                                                             #
# One must 'swap'  modules on this machine to access different compilers.     #
###############################################################################

### Defaults to Intel.
#OPTFLAGS = -g -O2 -xmic-avx512 -ipo
#OPENMP_FLAGS = -DHAVE_OPENMP -qopenmp -pthread -DUSE_OPENMP_NO_GPU
#OPENMP_LDFLAGS = -qopenmp -pthread
#
#CXX=CC
#CXXFLAGS = -std=c++11 $(OPTFLAGS)
#CPPFLAGS = -DHAVE_MPI -DCHRONO_MISSING  $(OPENMP_FLAGS)
#LDFLAGS  = $(OPENMP_LDFLAGS) 



################################################################################
### Below here, it is pitch black.                                           ###
### You are likely to be eaten by a grue.                                    ###
################################################################################

#GITVERS := -D'GIT_VERS="$(shell git log -n 1 |  grep Date   | awk -F " " '{print $$6 "-" $$3 "-" $$4 "-" $$5}')"'
#GITHASH := -D'GIT_HASH="$(shell git log -n 1 |  grep commit | awk -F " " '{print $$2}')"'
GITVERS := "$(shell git log -n 1 |  grep Date   | awk -F " " '{print $$6 "-" $$3 "-" $$4 "-" $$5}')"
GITHASH := "$(shell git log -n 1 |  grep commit | awk -F " " '{print $$2}')"

Quicksilver_EXE=qs

# clear all suffixes
.SUFFIXES:
# list only those that we use 
.SUFFIXES: .cc .o

.PHONY: DEFAULT clean distclean depend

# For development purposes, what is working now.
SOURCES= \
    CollisionEvent.cc \
    CoralBenchmark.cc \
    CycleTracking.cc \
    DecompositionObject.cc \
    DirectionCosine.cc \
	EnergySpectrum.cc \
    GlobalFccGrid.cc \
    GridAssignmentObject.cc \
    InputBlock.cc \
    MCT.cc \
    MC_Adjacent_Facet.cc \
    MC_Base_Particle.cc \
    MC_Domain.cc \
    MC_Facet_Crossing_Event.cc \
    MC_Fast_Timer.cc \
    MC_Load_Particle.cc \
    MC_Location.cc \
    MC_Particle_Buffer.cc \
    MC_RNG_State.cc \
    MC_Segment_Outcome.cc \
    MC_SourceNow.cc \
    MacroscopicCrossSection.cc \
    MeshPartition.cc \
    MonteCarlo.cc \
    MpiCommObject.cc \
    NuclearData.cc \
    Parameters.cc \
    ParticleVault.cc \
    ParticleVaultContainer.cc \
    PopulationControl.cc \
    SendQueue.cc \
    SharedMemoryCommObject.cc \
    Tallies.cc \
    cmdLineParser.cc \
    cudaFunctions.cc \
    initMC.cc \
    main.cc \
    parseUtils.cc \
    utils.cc \
    utilsMpi.cc 

CC_OBJECTS=$(SOURCES:.cc=.o)

DEFAULT: ${Quicksilver_EXE}

git_hash.hh:
	echo "#define GIT_HASH \"$(GITHASH)\" "> git_hash.hh

git_vers.hh:
	echo "#define GIT_VERS \"$(GITVERS)\" "> git_vers.hh

%.o: %.cc
	${CXX} ${CPPFLAGS} ${CXXFLAGS} -c $< -o $@

${Quicksilver_EXE}: git_hash.hh git_vers.hh ${CC_OBJECTS}  
	${CXX} ${CXXFLAGS} ${LDFLAGS} -o ${Quicksilver_EXE} ${CC_OBJECTS}

clean:
	rm -f *.o git_hash.hh git_vers.hh .depend load.map *.core *.optrpt

distclean: clean
	rm -f ${Quicksilver_EXE}   .depend.bak
	rm -rf html latex vtune*

.depend: $(SOURCES)
	@touch .depend
	@$(MAKE) --no-print-directory depend

depend:
	@echo "Rebuilding dependencies..."
	@makedepend -f .depend -Y. --$(CXXFLAGS) $(CPPFLAGS)-- $(SOURCES) 2> /dev/null

-include .depend
