#include "DeclareMacro.hh"

// Forward Declaration
class ParticleVault;
class MonteCarlo;
class MC_Particle;

HOST_DEVICE
void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault );
HOST_DEVICE_END

HOST_DEVICE
void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault);
HOST_DEVICE_END
