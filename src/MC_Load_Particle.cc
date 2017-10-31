#include "ParticleVault.hh"
#include "MC_Particle.hh"
#include "MC_Time_Info.hh"
#include "DeclareMacro.hh"

//----------------------------------------------------------------------------------------------------------------------
//  Copies a single particle from the particle-vault data structure into the active-particle data structure.
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE
void MC_Load_Particle(MonteCarlo *monteCarlo, MC_Particle &mc_particle, ParticleVault *particleVault, int particle_index)
{
    //particleVault.popParticle(mc_particle);
    particleVault->getParticle(mc_particle, particle_index);

    // Time to Census
    if ( mc_particle.time_to_census <= 0.0 )
    {
        mc_particle.time_to_census += monteCarlo->time_info->time_step;
    }

    // Age
    if (mc_particle.age < 0.0) { mc_particle.age = 0.0; }

    //    Energy Group
    mc_particle.energy_group = monteCarlo->_nuclearData->getEnergyGroup(mc_particle.kinetic_energy);
//                    printf("file=%s line=%d\n",__FILE__,__LINE__);

}
HOST_DEVICE_END

