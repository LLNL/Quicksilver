/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include <sycl/sycl.hpp>
#include "MonteCarlo.hh"
#include "NuclearData.hh"
#include "MaterialDatabase.hh"
#include "ParticleVaultContainer.hh"
#include "MC_RNG_State.hh"
#include "Tallies.hh"
#include "MC_Processor_Info.hh"
#include "MC_Time_Info.hh"
#include "MC_Particle_Buffer.hh"
#include "MC_Fast_Timer.hh"
#include <cmath>

#include "macros.hh" // current location of openMP wrappers.
#include "cudaUtils.hh"

//----------------------------------------------------------------------------------------------------------------------
// Construct a MonteCarlo object.
//----------------------------------------------------------------------------------------------------------------------
MonteCarlo::MonteCarlo(const Parameters &params)
    : domain_d(NULL),
      domainSize(0),
      _params(params),
      _nuclearData(NULL),
      _material_d(NULL),
      _nuclearData_d(NULL)
{
    _nuclearData = 0;
    _materialDatabase = 0;

#if defined(HAVE_UVM)
    void *ptr1, *ptr2, *ptr3, *ptr4;

    gpuMallocManaged(&ptr1, sizeof(Tallies), 1 /*cudaMemAttachGlobal*/);
    gpuMallocManaged(&ptr2, sizeof(MC_Processor_Info), 1 /*cudaMemAttachGlobal*/);
    gpuMallocManaged(&ptr3, sizeof(MC_Time_Info), 1 /*cudaMemAttachGlobal*/);
    gpuMallocManaged(&ptr4, sizeof(MC_Fast_Timer_Container), 1 /*cudaMemAttachGlobal*/);

    _tallies = new (ptr1) Tallies(params.simulationParams.balanceTallyReplications,
                                  params.simulationParams.fluxTallyReplications,
                                  params.simulationParams.cellTallyReplications,
                                  params.simulationParams.energySpectrum,
                                  params.simulationParams.nGroups);
    processor_info = new (ptr2) MC_Processor_Info();
    time_info = new (ptr3) MC_Time_Info();
    fast_timer = new (ptr4) MC_Fast_Timer_Container();

#else
    _tallies = new Tallies(params.simulationParams.balanceTallyReplications,
                           params.simulationParams.fluxTallyReplications,
                           params.simulationParams.cellTallyReplications,
                           params.simulationParams.energySpectrum,
                           params.simulationParams.nGroups);
    processor_info = new MC_Processor_Info();
    time_info = new MC_Time_Info();
    fast_timer = new MC_Fast_Timer_Container();
#endif

    source_particle_weight = 0.0;

    size_t num_processors = processor_info->num_processors;
    size_t num_particles = params.simulationParams.nParticles;
    size_t batch_size = params.simulationParams.batchSize;
    size_t num_batches = params.simulationParams.nBatches;

    size_t num_particles_on_process = num_particles / num_processors;

    if (num_particles_on_process <= 0)
    {
        MC_Fatal_Jump("Not enough particles for each process ( Ranks: %d Num Particles: %d ) \n", num_processors, num_particles);
        num_particles_on_process = 1;
    }

    if (batch_size == 0) // batch size unset - use num_batches to get batch_size
    {
        batch_size = (num_particles_on_process / num_batches) + ((num_particles_on_process % num_batches == 0) ? 0 : 1);
    }
    else // batch size explicatly set - use to find num_batches
    {
        num_batches = num_particles_on_process / batch_size + ((num_particles_on_process % batch_size == 0) ? 0 : 1);
    }

    size_t vector_size = 0;

    for (auto matIter = params.materialParams.begin();
         matIter != params.materialParams.end();
         matIter++)
    {
        const MaterialParameters &mp = matIter->second;
        double nuBar = params.crossSectionParams.at(mp.fissionCrossSection).nuBar;
        size_t nb = sycl::ceil(nuBar);
        size_t test_size = nb * (batch_size);

        if (test_size > vector_size)
            vector_size = test_size;
    }
    if (vector_size == 0)
        vector_size = 2 * batch_size;

    int num_extra_vaults = (vector_size / batch_size) + 1;
    // Previous definition was not enough extra space for some reason? need to determine why still

#if defined(HAVE_UVM)
    void *ptr5, *ptr6;
    gpuMallocManaged(&ptr5, sizeof(MC_Particle_Buffer), 1 /*cudaMemAttachGlobal*/);
    gpuMallocManaged(&ptr6, sizeof(ParticleVaultContainer), 1 /*cudaMemAttachGlobal*/);
    particle_buffer = new (ptr5) MC_Particle_Buffer(this, batch_size);
    _particleVaultContainer = new (ptr6) ParticleVaultContainer(batch_size, num_batches, num_extra_vaults);
#else
    particle_buffer = new MC_Particle_Buffer(this, batch_size);
    _particleVaultContainer = new ParticleVaultContainer(batch_size, num_batches, num_extra_vaults);
#endif
}

//----------------------------------------------------------------------------------------------------------------------
// Destruct a MonteCarlo object.
//----------------------------------------------------------------------------------------------------------------------
MonteCarlo::~MonteCarlo()
{
#if defined(HAVE_UVM)

    _nuclearData->~NuclearData();
    _particleVaultContainer->~ParticleVaultContainer();
    _materialDatabase->~MaterialDatabase();
    _tallies->~Tallies();
    processor_info->~MC_Processor_Info();
    time_info->~MC_Time_Info();
    fast_timer->~MC_Fast_Timer_Container();
    particle_buffer->~MC_Particle_Buffer();

    // sycl::free(_nuclearData, sycl_device_queue);
    gpuFree(_particleVaultContainer);
    // sycl::free(_materialDatabase, sycl_device_queue);
    gpuFree(_tallies);
    gpuFree(processor_info);
    gpuFree(time_info);
    gpuFree(fast_timer);
    gpuFree(particle_buffer);

    sycl::free(domain_d, sycl_device_queue);
    sycl::free(_material_d, sycl_device_queue);
    sycl::free(_nuclearData_d, sycl_device_queue);

#else
    delete _nuclearData;
    delete _particleVaultContainer;
    delete _materialDatabase;
    delete _tallies;
    delete processor_info;
    delete time_info;
    delete fast_timer;
    delete particle_buffer;
#endif
}

void MonteCarlo::clearCrossSectionCache()
{
    int numEnergyGroups = _nuclearData->_numEnergyGroups;
    for (unsigned ii = 0; ii < domain.size(); ++ii)
        domain[ii].clearCrossSectionCache(numEnergyGroups);
}
