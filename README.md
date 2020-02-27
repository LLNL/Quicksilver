Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=================================
A hipified version of Quicksilver
=================================
NOTE:
=====
This will only work with ROCM 1.7 and later. 


Compiling:
==========
Please see the Makefile in "src/Makefile". This file contains sample compile flags for compiling just HIP, HIP with MPI, and HIP with nvcc.



Changes from DOE provided Quicksilver:
======================================
I have made several changes to get this to work with hip:

1. hip doesn't have an equivalent to CUDA's mallocManaged, so significant work had to be done to copy most data between the host and device manually. The main changes to do this are summarized below:

    a. Several new objects were created to store necessary data on the device. These objects are typically named the same as their host side counterparts, except with a "_d." For instance there is an object "domain_d," that stores the MC_Domain information contained in the object "domain" but on the device. This object "domain_d" is of type "MC_Domain_d" which differs from "MC_Domain" primarily in that "MC_Domain_d" stores it's member arrays on the device rather than the host. We also introduced the classes Material_d, NuclearData_d, and Tallies_d for similar reasons.
    
    b. Functions were created to unroll the MC_Domain, Material, and NuclearData classes and copy them onto the device.

    c. Tallies like numSegments are stored in buffers on the device and then copied back to the host. Further tallies for each block are actually stored in LDS and then added to the global memory buffer. This was found to be slightly more efficient. 

    d. Handling all memory transfers manually is still somewhat of a work in progress. At this time ParticleVault's are not yet copied between device and host manually. Instead the ParticleVault's arrays are allocated using the command "hipHostMalloc(&ptr, size*sizeof(T),hipHostMallocNonCoherent)". This command allocates memory on the host that is pinned. Further the "hipHostMallocNonCoherent" argument says that the memory can be cached by the device. Given the fact that particles are only operated on by one particle at a time this is fine and coherence does not need to be maintained. A version that manually handles particle movement between host and device is in development.

2. The CycleTracking kernel has been modified so that a wave will only iterate through a limited number of sub-cycles before exiting, rather than waiting for every particle in the wave to reach census. This has been found to improve performance when running on the GPU because it allows kernels to more fully utilize GPU resources. Users can specify how many cycles the kernel should allow at compile time using "-DMaxIt=N" where N is the number of iterations. "N=10" has been found to be a good value for the CORAL2 problems, but this will vary for different problems. By default MaxIt is set to the largest value an int type can take. Thus by default MaxIt is set to a number large enough it should never be reached in any one timestep. If MaxIt is not set by the user then the loop in the CycleTracking kernel will execute like the while loop in the version of Quicksilver on the public github repo: https://github.com/LLNL/Quicksilver

Compiling and running:
======================
To compile you must set several environment variables in the Makefile in the src/ directory. For example, in order to compile using HIP without MPI, and up to 15 sub-iterations for a particle per kernel launch,  you could set the environment variables as follows (this is how they are set by default, you just need to point to where HIP is installed):

CXX = $(HIP)/bin/hipcc<br/>
CXXFLAGS = -I$(HIP)/include/<br/>
CPPFLAGS = -DHAVE_HIP=1 -DMaxIt=15<br/>
LDFLAGS = -L$(HIP)/lib -L$(HIP)/lib

To compile using HIP and MPI and up to 15 sub-iterations for a particle per kernel launch, you could set the following environment variables:

CXX = $(HIP)/bin/hipcc<br/>
CXXFLAGS1 = -I$(HIP)/include/<br/>
CXXFLAGS2 = $(CXXFLAGS1) -I$(MPIPATH)/include<br/>
CXXFLAGS = $(CXXFLAGS2) -pthread<br/>
CPPFLAGS = -DHAVE_HIP=1 -DHAVE_MPI -DMaxIt=15<br/>
LDFLAGS = -L$(HIP)/lib -L$(MPIPATH)/lib -lmpicxx -lmpi

To run look in src/READ.ME.HOW.TO.RUN. You can also look at the examples in the .sh files found in the directories 'Examples/CORAL2_Benchmark/Problem1' or 'Examples/CORAL2_Benchmark/Problem2'.

A Note on ROCm Versions:
========================
This hip port of Quicksilver was tested and found to work with every ROCm release up through ROCm 2.10. Please report any issues you find with the hip port to scott.moe@amd.com. 

