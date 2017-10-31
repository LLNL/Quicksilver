#include "MCT.hh"
#include "MC_Domain.hh"
#include "Globals.hh"
#include "MonteCarlo.hh"
#include "DeclareMacro.hh"

class MC_Particle;

HOST_DEVICE

Subfacet_Adjacency &MCT_Adjacent_Facet(const MC_Location &location, MC_Particle &mc_particle, MonteCarlo* monteCarlo)

{
   MC_Domain &domain = monteCarlo->domain[location.domain];

   Subfacet_Adjacency &adjacency =domain.mesh._cellConnectivity[location.cell]._facet[location.facet].subfacet;

   return adjacency;
}

HOST_DEVICE_END
