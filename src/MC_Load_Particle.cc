#include "ParticleVault.hh"
#include "MC_Base_Particle.hh"
#include "MC_Time_Info.hh"
#include "DeclareMacro.hh"

//----------------------------------------------------------------------------------------------------------------------
//  Copies a single particle from the particle-vault data structure into the active-particle data structure.
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE
//void MC_Load_Particle(MonteCarlo *monteCarlo, MC_Base_Particle &mc_particle, ParticleVault *particleVault, int particle_index)
MC_Base_Particle& MC_Load_Particle(MonteCarlo *monteCarlo, ParticleVault *particleVault, int particle_index)
{
    //particleVault.popParticle(mc_particle);
   //particleVault->getParticle(mc_particle, particle_index);

   MC_Base_Particle& mc_particle = (*particleVault)[particle_index];
     double speed = mc_particle.velocity.Length();

     if ( speed > 0 )
     {
         double factor = 1.0/speed;
         mc_particle.direction_cosine.alpha = factor * mc_particle.velocity.x;
         mc_particle.direction_cosine.beta  = factor * mc_particle.velocity.y;
         mc_particle.direction_cosine.gamma = factor * mc_particle.velocity.z;
     }
     else
     {
         qs_assert(false);
     }

     mc_particle.totalCrossSection = -1.;
     mc_particle.mean_free_path = 0;
     mc_particle.segment_path_length = 0;
     mc_particle.normal_dot = 0;
     mc_particle.species = 0;
     mc_particle.task = 0;

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

    return mc_particle;
}
HOST_DEVICE_END

