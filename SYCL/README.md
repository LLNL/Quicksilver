Quicksilver
===========

Introduction
------------

Quicksilver is a proxy application that represents some elements of
the Mercury workload by solving a simpliﬁed dynamic monte carlo
particle transport problem.  Quicksilver attempts to replicate the
memory access patterns, communication patterns, and the branching or
divergence of Mercury for problems using multigroup cross sections.
OpenMP and MPI are used for parallelization.  A GPU version is
available.  Unified memory is assumed.

Performance of Quicksilver is likely to be dominated by latency bound
table look-ups, a highly branchy/divergent code path, and poor
vectorization potential.

For more information, visit the
[LLNL co-design pages.](https://codesign.llnl.gov/quicksilver.php)

**To build sycl version**

source /path/to/oneAPI package

mkdir build && cd build

CXX=icpx cmake ../ -DGPU_AOT=PVC

make -sj

**To build sycl version on nvidia backend**

source /path/to/clang/

mkdir build && cd build

//For A100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=80

//For H100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_NVIDIA_BACKEND=YES -DUSE_SM=90

make -sj

**To build sycl version on amd backend**

source /path/to/clang/

mkdir build && cd build

//For MI-100 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx908

//For MI-250 Machine

CC=clang CXX=clang++ cmake ../ -DUSE_AMDHIP_BACKEND=gfx90a

make -sj

Running Quicksilver
-------------------

Quicksilver’s behavior is controlled by a combination of command line
options and an input file.  All of the parameters that can be set on
the command line can also be set in the input file.  The input file
values will override the command line.  Run `$ qs –h` to see
documentation on the available command line switches.  Documentation
of the input file parameters is in preparation.

Quicksilver also has the property that the output of every run is a
valid input file.  Hence you can repeat any run for which you have the
output file by using that output as an input file.

For benchmarking run the example "Examples/CORAL2_Benchmark/Problem1/Coral2_P1_1.inp"

**To run sycl version**

export QS_DEVICE=GPU

./qs -i ../Examples/AllScattering/scatteringOnly.inp

**To run sycl version on nvidia backend**

export QS_DEVICE=GPU

./qs -i ../Examples/AllScattering/scatteringOnly.inp

**To run sycl version on amd backend**

export QS_DEVICE=GPU

ONEAPI_DEVICE_SELECTOR=hip:* ./qs -i ../Examples/AllScattering/scatteringOnly.inp

License and Distribution Information
------------------------------------

Quicksilver is available [on github](https://github.com/LLNL/Quicksilver)