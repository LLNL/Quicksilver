#include "MC_Facet_Crossing_Event.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Domain.hh"
#include "Tallies.hh"
#include "MC_Particle.hh"
#include "MC_Facet_Adjacency.hh"
#include "Globals.hh"
#include "MCT.hh"
#include "MC_Particle_Buffer.hh"
#include "DeclareMacro.hh"
#include "macros.hh"
#include "SendQueue.hh"

//----------------------------------------------------------------------------------------------------------------------
//  Determines whether the particle has been tracked to a facet such that it:
//    (i) enters into an adjacent cell
//   (ii) escapes across the system boundary (Vacuum BC), or
//  (iii) reflects off of the system boundary (Reflection BC).
//
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE

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

        // Select particle buffer
        int neighbor_rank = monteCarlo->domain[facet_adjacency.current.domain].mesh._nbrRank[facet_adjacency.neighbor_index];

        processingVault->putParticle( mc_particle, particle_index );

        //Push neighbor rank and mc_particle onto the send queue
        monteCarlo->_particleVaultContainer->getSendQueue()->push( neighbor_rank, particle_index );

    }

    return mc_particle.last_event;
}

HOST_DEVICE_END
