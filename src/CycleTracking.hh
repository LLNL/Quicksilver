#include "DeclareMacro.hh"

// Forward Declaration
class ParticleVault;
class MonteCarlo;
class MC_Particle;

HOST_DEVICE SYCL_EXTERNAL
void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault );
HOST_DEVICE_END

HOST_DEVICE SYCL_EXTERNAL
void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault);
HOST_DEVICE_END
