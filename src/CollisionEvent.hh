#ifndef COLLISION_EVENT_HH
#define COLLISION_EVENT_HH

class MonteCarlo;
class MC_Particle;

#include "DeclareMacro.hh"
HOST_DEVICE
bool CollisionEvent(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int tally_index );
HOST_DEVICE_END


#endif

