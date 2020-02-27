/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef MC_FACET_CROSSING_EVENT_HH
#define MC_FACET_CROSSING_EVENT_HH

#include "Tallies.hh"
#include "DeclareMacro.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Domain.hh"
#include "MC_Particle.hh"
#include "MC_Facet_Adjacency.hh"
#include "Globals.hh"
#include "MCT.hh"
#include "MC_Particle_Buffer.hh"
#include "DeclareMacro.hh"
#include "macros.hh"
#include "SendQueue.hh"

class ParticleVault;
class MC_Particle;

//----------------------------------------------------------------------------------------------------------------------
//  Determines whether the particle has been tracked to a facet such that it:
//    (i) enters into an adjacent cell
//   (ii) escapes across the system boundary (Vacuum BC), or
//  (iii) reflects off of the system boundary (Reflection BC).
//
//----------------------------------------------------------------------------------------------------------------------


inline HOST_DEVICE

MC_Tally_Event::Enum MC_Facet_Crossing_Event(MC_Particle &mc_particle, MonteCarlo* monteCarlo, int particle_index, ParticleVault* processingVault)
{
    MC_Location location = mc_particle.Get_Location();

    Subfacet_Adjacency &facet_adjacency = MCT_Adjacent_Facet(location, mc_particle, monteCarlo);

    if ( facet_adjacency.event == MC_Subfacet_Adjacency_Event::Transit_On_Processor )
    {
        // The particle will enter into an adjacent cell.
        mc_particle.domain     = facet_adjacency.adjacent.domain;
        mc_particle.cell       = facet_adjacency.adjacent.cell;
        mc_particle.facet      = facet_adjacency.adjacent.facet;
        mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Transit_Exit;
    }
    else if ( facet_adjacency.event == MC_Subfacet_Adjacency_Event::Boundary_Escape )
    {
        // The particle will escape across the system boundary.
        mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Escape;
    }
    else if ( facet_adjacency.event == MC_Subfacet_Adjacency_Event::Boundary_Reflection )
    {
        // The particle will reflect off of the system boundary.
        mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Reflection;
    }
    else if ( facet_adjacency.event == MC_Subfacet_Adjacency_Event::Transit_Off_Processor )
    {
        // The particle will enter into an adjacent cell on a spatial neighbor.
        // The neighboring domain is on another processor. Set domain local domain on neighbor proc

        mc_particle.domain     = facet_adjacency.adjacent.domain;
        mc_particle.cell       = facet_adjacency.adjacent.cell;
        mc_particle.facet      = facet_adjacency.adjacent.facet;
        mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Communication;

        #ifdef  __HIP_DEVICE_COMPILE__
        int neighbor_rank = monteCarlo->domain_d[facet_adjacency.current.domain].mesh._nbrRank[facet_adjacency.neighbor_index];
        #else
        // Select particle buffer
        int neighbor_rank = monteCarlo->domain[facet_adjacency.current.domain].mesh._nbrRank[facet_adjacency.neighbor_index];
        #endif

        processingVault->putParticle( mc_particle, particle_index );

        //Push neighbor rank and mc_particle onto the send queue
        monteCarlo->_particleVaultContainer->getSendQueue()->push( neighbor_rank, particle_index );

    }

    return mc_particle.last_event;
}

HOST_DEVICE_END


HOST_DEVICE
MC_Tally_Event::Enum MC_Facet_Crossing_Event(MC_Particle &mc_particle, MonteCarlo* monteCarlo, int particle_index, ParticleVault* processingVault);
HOST_DEVICE_END

#endif

