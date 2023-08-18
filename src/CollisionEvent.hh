#ifndef COLLISION_EVENT_HH
#define COLLISION_EVENT_HH

#include "DeclareMacro.hh"

class MonteCarlo;
class MC_Particle;

HOST_DEVICE
bool CollisionEvent(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int tally_index );
HOST_DEVICE_END


#endif

