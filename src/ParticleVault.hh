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

#ifndef PARTICLEVAULT_HH
#define PARTICLEVAULT_HH

#include "MC_Base_Particle.hh"
#include "ParticleVault.hh"
#include "MC_Particle.hh"
#include "MC_Time_Info.hh"
#include "DeclareMacro.hh"
#include "QS_Vector.hh"

#include <vector>

class ParticleVault
{
public:

   // Is the vault empty.
   bool empty() const {return _particles.empty();}

   // Get the size of the vault.
   HOST_DEVICE_SYCL
   size_t size() const {return _particles.size();}

   HOST_DEVICE_SYCL
   void setsize(int size) {_particles.setsize(size);}

   // Reserve the size for the container of particles.
   void reserve(size_t n)
   { 
       _particles.reserve(n,VAR_MEM); 
   }

   // Add all particles in a 2nd vault into this vault.
   void append (ParticleVault & vault2)
        { _particles.appendList( vault2._particles.size(), &vault2._particles[0] ); }

   void collapse( size_t fill_size, ParticleVault* vault2 );

   // Clear all particles from the vault
   void clear() { _particles.clear(); } 

   // Access particle at a given index.
   MC_Base_Particle& operator[](size_t n) {return _particles[n];}

   // Access a particle at a given index.
   const MC_Base_Particle& operator[](size_t n) const {return _particles[n];}

   // Put a particle into the vault, down casting its class.
   HOST_DEVICE_SYCL
   void pushParticle(MC_Particle &particle);

   // Put a base particle into the vault.
   HOST_DEVICE_SYCL
   void pushBaseParticle(MC_Base_Particle &base_particle);

   // Get a base particle from the vault.
   bool popBaseParticle(MC_Base_Particle &base_particle);

   // Get a particle from the vault.
   bool popParticle(MC_Particle &particle);

   // Get a particle from the vault 
   bool getBaseParticleComm(MC_Base_Particle &particle, int index);
   HOST_DEVICE_SYCL
   bool getParticle(MC_Particle &particle, int index);
   // Copy a particle back into the vault
   HOST_DEVICE_SYCL
   bool putParticle(MC_Particle particle, int index);

   // invalidates the particle in the vault at an index
   HOST_DEVICE_SYCL
   void invalidateParticle(int index);

#if 0
   // Remove all of the invalid particles form the _particles list
   void cleanVault(int end_index);
#endif

   // Swap vaults.
   void swapVaults(ParticleVault &vault);

   // Swaps this particle at index with last particle and resizes to delete it
   void eraseSwapParticle(int index);

private:

   // The container of particles.
   qs_vector<MC_Base_Particle> _particles;
};

// -----------------------------------------------------------------------
HOST_DEVICE_SYCL
inline void ParticleVault::
pushParticle(MC_Particle &particle)
{
    MC_Base_Particle base_particle(particle);
    size_t indx = _particles.atomic_Index_Inc(1);
    _particles[indx] = base_particle;
}


// -----------------------------------------------------------------------
HOST_DEVICE_SYCL
inline void ParticleVault::
pushBaseParticle(MC_Base_Particle &base_particle)
{
    int indx = _particles.atomic_Index_Inc(1);
    _particles[indx] = base_particle;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
popBaseParticle(MC_Base_Particle &base_particle)
{
   bool notEmpty = false;

#include "mc_omp_critical.hh"
{
   if (!empty())
   {
      base_particle = _particles.back();
      _particles.pop_back();
      notEmpty = true;
   }
}
    return notEmpty;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
popParticle(MC_Particle &particle)
{
   bool notEmpty = false;

#include "mc_omp_critical.hh"
{
   if (!empty())
   {
      MC_Base_Particle base_particle(_particles.back());
      _particles.pop_back();
      particle = MC_Particle(base_particle);
      notEmpty = true;
   }
}
   return notEmpty;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
getBaseParticleComm( MC_Base_Particle &particle, int index )
{
    if(size() > index)
    {
            particle = _particles[index];
            _particles[index].species = -1;
            return true;
    }
    else
    {
        qs_assert(false);
    }
    return false;
}

// -----------------------------------------------------------------------
   HOST_DEVICE_SYCL
inline bool ParticleVault::
getParticle(MC_Particle &particle, int index)
{
    qs_assert(size() > index);
    if(size() > index)
    {
            MC_Base_Particle base_particle(_particles[index]);
            particle = MC_Particle(base_particle);

            return true;
    }
    return false;
}

// -----------------------------------------------------------------------
HOST_DEVICE_SYCL
inline bool ParticleVault::
putParticle(MC_Particle particle, int index)
{
    qs_assert(size() > index);
    if(size() > index)
    {
        MC_Base_Particle base_particle(particle);
        _particles[index] = base_particle;
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------
HOST_DEVICE_SYCL
inline void ParticleVault::
invalidateParticle(int index)
{
    qs_assert(index >= 0);
    qs_assert(index < _particles.size());
    _particles[index].species = -1;
}

// -----------------------------------------------------------------------
inline void ParticleVault::
eraseSwapParticle(int index)
{
    #include "mc_omp_critical.hh"
    {
        _particles[index] = _particles.back();
        _particles.pop_back();
    }
}

// -----------------------------------------------------------------------
inline HOST_DEVICE
void MC_Load_Particle(MonteCarlo *monteCarlo, MC_Particle &mc_particle, ParticleVault *particleVault, int particle_index)
{
    //particleVault.popParticle(mc_particle);
    particleVault->getParticle(mc_particle, particle_index);

    // Time to Census
    if (mc_particle.time_to_census <= 0.0)
    {
        mc_particle.time_to_census += monteCarlo->time_info->time_step;
    }

    // Age
    if (mc_particle.age < 0.0) { mc_particle.age = 0.0; }

//    Energy Group
#ifdef  __SYCL_DEVICE_ONLY__
    mc_particle.energy_group = monteCarlo->_nuclearData_d->getEnergyGroup(mc_particle.kinetic_energy);
#else
    mc_particle.energy_group = monteCarlo->_nuclearData->getEnergyGroup(mc_particle.kinetic_energy);
#endif
//                    printf("file=%s line=%d\n",__FILE__,__LINE__);
}
HOST_DEVICE_END

#endif
