#ifndef MCT_HH
#define MCT_HH

#include "portability.hh"
#include "DeclareMacro.hh"

class MC_Particle;
class MC_Domain;
class MC_Location;
class MC_Vector;
class DirectionCosine;
class MC_Nearest_Facet;
class Subfacet_Adjacency;
class MonteCarlo;


HOST_DEVICE
MC_Nearest_Facet MCT_Nearest_Facet(
   MC_Particle *mc_particle,
   MC_Location &location,
   MC_Vector &coordinate,
   const DirectionCosine *direction_cosine,
   double distance_threshold,
   double current_best_distance,
   bool new_segment, 
   MonteCarlo* monteCarlo);
HOST_DEVICE_END


HOST_DEVICE
void MCT_Generate_Coordinate_3D_G(
   uint64_t *random_number_seed,
   int domain_num,
   int cell,
   MC_Vector &coordinate,
   MonteCarlo* monteCarlo);
HOST_DEVICE_END

HOST_DEVICE
MC_Vector MCT_Cell_Position_3D_G(
   const MC_Domain   &domain,
   int cell_index);
HOST_DEVICE_END

HOST_DEVICE
Subfacet_Adjacency &MCT_Adjacent_Facet(const MC_Location &location, MC_Particle &mc_particle, MonteCarlo* monteCarlo);
HOST_DEVICE_END

HOST_DEVICE
void MCT_Reflect_Particle(MonteCarlo *mcco, MC_Particle &particle);
HOST_DEVICE_END

#endif
