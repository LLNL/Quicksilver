cmake_minimum_required(VERSION 3.21)

project (QuickSilver VERSION 2.0.0)

## Fail if someone tries to config an in-source build.
if("${CMAKE_SOURCE_DIR}" STREQUAL "${CMAKE_BINARY_DIR}")
   message(FATAL_ERROR "In-source builds are not supported. Please remove "
                       "CMakeCache.txt from the 'src' dir and configure an "
                       "out-of-source build in another directory.")
endif()

# Set the build type to Release by default if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfoO3")
endif()

if(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfoO3")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
    set(O3_OPT_LEVEL "-O3")
    string(REPLACE "-O2" ${O3_OPT_LEVEL} CMAKE_CXX_FLAGS_RELWITHDEBINFO ${CMAKE_CXX_FLAGS_RELWITHDEBINFO})
    string(REPLACE "-O2" ${O3_OPT_LEVEL} CMAKE_C_FLAGS_RELWITHDEBINFO ${CMAKE_C_FLAGS_RELWITHDEBINFO})
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(ENABLE_MPI "Use MPI" TRUE) 
option(USE_ASYNC_MPI "Use Async MPI" TRUE) 
option(USE_NVTX "Use Nvidia NVTX" FALSE) 
option(ENABLE_CUDA "Use CUDA" FALSE)
option(ENABLE_HIP "Use HIP" FALSE)
option(ENABLE_CALIPER "Enable Caliper" FALSE)
option(ENABLE_OMP "Enable OpenMP" FALSE)
option(DISABLE_TIMERS "Disable high resolution timers" FALSE)
option(EXPONENTIAL_TALLY "Exponential cell-based tally" FALSE)
option(CMAKE_EXTRA_CXXFLAGS "Any extra CXX flags" "")

set(CMAKE_VERBOSE_MAKEFILE ON)

execute_process(
    COMMAND git log -1 --format="Date:   %cd-%h"
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GITVERS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GITHASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

set(SOURCES 
    src/CollisionEvent.cc
    src/CoralBenchmark.cc
    src/CycleTracking.cc
    src/DecompositionObject.cc
    src/DirectionCosine.cc
	  src/EnergySpectrum.cc
    src/GlobalFccGrid.cc
    src/GridAssignmentObject.cc
    src/InputBlock.cc
    src/MCT.cc
    src/MC_Adjacent_Facet.cc
    src/MC_Base_Particle.cc
    src/MC_Domain.cc
    src/MC_Facet_Crossing_Event.cc
    src/MC_Fast_Timer.cc
    src/MC_Load_Particle.cc
    src/MC_Location.cc
    src/MC_Particle_Buffer.cc
    src/MC_RNG_State.cc
    src/MC_Segment_Outcome.cc
    src/MC_SourceNow.cc
    src/MacroscopicCrossSection.cc
    src/MeshPartition.cc
    src/MonteCarlo.cc
    src/MpiCommObject.cc
    src/NuclearData.cc
    src/Parameters.cc
    src/ParticleVault.cc
    src/ParticleVaultContainer.cc
    src/PopulationControl.cc
    src/SendQueue.cc
    src/SharedMemoryCommObject.cc
    src/Tallies.cc
    src/cmdLineParser.cc
    src/cudaFunctions.cc
    src/initMC.cc
    src/main.cc
    src/parseUtils.cc
    src/utils.cc
    src/utilsMpi.cc)

set (CMAKE_QS_CXXFLAGS "-Wall")
if (DISABLE_TIMERS)
  list(APPEND CMAKE_QS_CXXFLAGS "-DDISABLE_TIMERS")
endif()

if (EXPONENTIAL_TALLY)
  list(APPEND CMAKE_QS_CXXFLAGS "-DEXPONENTIAL_TALLY")
endif()

if (ENABLE_CUDA)
  enable_language(CXX)

  find_package(CUDAToolkit REQUIRED)
  list(APPEND QS_DEPENDENCIES
    CUDA::cudart)

  include(CheckLanguage)
  check_language(CUDA)

  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

  if (NOT CMAKE_CUDA_COMPILER)
    message(FATAL_ERROR "Unable to find the nvcc compiler. Please use"
                        "CMAKE_CUDA_COMPILER to provide the nvcc compiler.")
  endif()

  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "${CUDA_ARCH}")
  set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CUDA)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")
  list(APPEND CMAKE_QS_CXXFLAGS "-DHAVE_CUDA")

  if(O3_OPT_LEVEL)
    string(REPLACE "-O2" ${O3_OPT_LEVEL} CMAKE_CUDA_FLAGS_RELWITHDEBINFO ${CMAKE_CUDA_FLAGS_RELWITHDEBINFO})
  endif()

  if(USE_NVTX)
    find_library(CUDA_NVTX_LIBRARY
      NAMES nvToolsExt nvTools nvtoolsext nvtools nvtx NVTX
      PATHS "${CUDA_CUDART_LIBRARY_DIR}" "${CUDA_TOOLKIT_ROOT_DIR}" ENV LD_LIBRARY_PATH
      PATH_SUFFIXES "lib64" "common/lib64" "common/lib" "lib"
      DOC "Find the location of the CUDA Toolkit Extension (NVTX) library"
      NO_DEFAULT_PATH
    )
    if(NOT CUDA_NVTX_LIBRARY)
      message(FATAL_ERROR "NVTX library not found.")
    endif()
    list(APPEND QS_DEPENDENCIES
      ${CUDA_NVTX_LIBRARY})
    list(APPEND CMAKE_QS_CXXFLAGS "-DUSE_NVTX")
  endif()
endif()

if (ENABLE_HIP)
  if(NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
  endif()

  #set(HIP_PATH "${ROCM_PATH}/hip")
  set(CMAKE_HIP_COMPILER "${ROCM_PATH}/llvm/bin/clang++" ${CMAKE_MODULE_PATH})
  set(CMAKE_C_COMPILER "${ROCM_PATH}/bin/hipcc")
  set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")
  #set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

  #if(NOT CMAKE_C_COMPILER)
  #  set(CMAKE_C_COMPILER ${ROCM_PATH}/bin/hipcc)
  #endif()

  #if(NOT CMAKE_CXX_COMPILER)
  #  set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
  #endif()

  enable_language(HIP)

  #find_package(HIP REQUIRED)
  list(APPEND QS_DEPENDENCIES
    "-fgpu-rdc"
    "--offload-arch=${HIP_ARCH}"
    "--hip-link")
    #HIP::host
    #HIP::device)
  list(APPEND CMAKE_QS_CXXFLAGS "-DHAVE_HIP")
  list(APPEND CMAKE_QS_CXXFLAGS -x hip)
  list(APPEND CMAKE_QS_CXXFLAGS "--offload-arch=${HIP_ARCH}")
  list(APPEND CMAKE_QS_CXXFLAGS "-fgpu-rdc")
  list(APPEND CMAKE_QS_CXXFLAGS "-Wno-unused-result")
endif()

if (ENABLE_OMP)
  find_package(OpenMP REQUIRED)
  list(APPEND QS_DEPENDENCIES
    OpenMP::OpenMP_CXX)
  list(APPEND CMAKE_QS_CXXFLAGS "-DHAVE_OPENMP")
endif()

if (ENABLE_MPI)
  find_package(MPI REQUIRED)
  list(APPEND QS_DEPENDENCIES
    MPI::MPI_CXX)
  list(APPEND CMAKE_QS_CXXFLAGS "-DHAVE_MPI")
  if (USE_ASYNC_MPI)
    list(APPEND CMAKE_QS_CXXFLAGS "-DHAVE_ASYNC_MPI")
  endif()
endif()

if (ENABLE_CALIPER)
  #find_package(caliper REQUIRED)
  #list(APPEND QS_DEPENDENCIES 
  #  caliper)
  #set(USE_CALIPER TRUE)
  #find_package(adiak REQUIRED)
  #list(APPEND QS_DEPENDENCIES
  #  adiak::adiak)
endif()

if (CMAKE_EXTRA_CXXFLAGS)
  string(REPLACE " " ";" CMAKE_LIST_QS_EXTRA_CXXFLAGS "${CMAKE_EXTRA_CXXFLAGS}")
  list(APPEND CMAKE_QS_CXXFLAGS ${CMAKE_LIST_QS_EXTRA_CXXFLAGS})
endif()

if (ENABLE_CUDA)
  string(JOIN " " CMAKE_QS_CUDAFLAGS ${CMAKE_QS_CXXFLAGS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler ${CMAKE_QS_CUDAFLAGS}")
else()
  add_compile_options(${CMAKE_QS_CXXFLAGS})
endif()

configure_file(
  "${CMAKE_SOURCE_DIR}/config.hh.in"
  "${CMAKE_BINARY_DIR}/config.hh"
)

add_executable(qs ${SOURCES})
target_link_libraries(qs PUBLIC ${QS_DEPENDENCIES})
target_include_directories(qs PRIVATE ${CMAKE_BINARY_DIR} ${HIP_INCLUDE_DIRS})

if (ENABLE_CUDA)
  set_target_properties(qs
    PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

if (ENABLE_HIP)
  set_target_properties(qs
    PROPERTIES
      HIP_SEPARABLE_COMPILATION ON
      HIP_RESOLVE_DEVICE_SYMBOLS ON)
endif()

install(TARGETS qs DESTINATION ${CMAKE_INSTALL_BINDIR})
