#ifndef MC_FACET_CROSSING_EVENT_HH
#define MC_FACET_CROSSING_EVENT_HH

#include "Tallies.hh"
#include "DeclareMacro.hh"

class ParticleVault;
class MC_Particle;

HOST_DEVICE
MC_Tally_Event::Enum MC_Facet_Crossing_Event(MC_Particle &mc_particle, MonteCarlo* monteCarlo, int particle_index, ParticleVault* processingVault);
HOST_DEVICE_END

#endif

