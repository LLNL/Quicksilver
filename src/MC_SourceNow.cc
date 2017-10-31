#include "MC_SourceNow.hh"
#include "QS_Vector.hh"
#include <iostream>
#include "utils.hh"
#include "utilsMpi.hh"
#include "MonteCarlo.hh"
#include "MaterialDatabase.hh"
#include "initMC.hh"
#include "Tallies.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Processor_Info.hh"
#include "MC_Cell_State.hh"
#include "MC_Time_Info.hh"
#include "MCT.hh"
#include "PhysicalConstants.hh"
#include "macros.hh"
#include "AtomicMacro.hh"
#include "NVTX_Range.hh"
#include <vector>

namespace
{
   double Get_Speed_From_Energy(double energy);
}


void MC_SourceNow(MonteCarlo *monteCarlo)
{
    NVTX_Range range("MC_Source_Now");
  
    std::vector<double> source_rate(monteCarlo->_materialDatabase->_mat.size());  // Get this from user input

    for ( int material_index = 0; material_index < monteCarlo->_materialDatabase->_mat.size(); material_index++ )
    {
        std::string name = monteCarlo->_materialDatabase->_mat[material_index]._name;
        double sourceRate = monteCarlo->_params.materialParams[name].sourceRate;
        source_rate[material_index] = sourceRate;
    }

    double local_weight_particles = 0;

    for ( int domain_index = 0; domain_index < monteCarlo->domain.size(); domain_index++ )
    {
        MC_Domain &domain = monteCarlo->domain[domain_index];

        for ( int cell_index = 0; cell_index < domain.cell_state.size(); cell_index++ )
        {
            MC_Cell_State &cell = domain.cell_state[cell_index];
            double cell_weight_particles = cell._volume * source_rate[cell._material] * monteCarlo->time_info->time_step;
            local_weight_particles += cell_weight_particles;
        }
    }

    double total_weight_particles = 0;

    mpiAllreduce(&local_weight_particles, &total_weight_particles, 1, MPI_DOUBLE, MPI_SUM, monteCarlo->processor_info->comm_mc_world);

    uint64_t num_particles = monteCarlo->_params.simulationParams.nParticles;
    double source_fraction = 0.1;
    double source_particle_weight = total_weight_particles/(source_fraction * num_particles);
    // Store the source particle weight for later use.
    monteCarlo->source_particle_weight = source_particle_weight;

    uint64_t vault_size       = monteCarlo->_particleVaultContainer->getVaultSize();
    uint64_t processing_index = monteCarlo->_particleVaultContainer->sizeProcessing() / vault_size;

    uint64_t task_index = 0;
    uint64_t particle_count = 0;

    // Compute the partial sums on each mpi process.
    // uint64_t local_num_particles = (int)(local_weight_particles / source_particle_weight);

    for ( int domain_index = 0; domain_index < monteCarlo->domain.size(); domain_index++ )
    {
        MC_Domain &domain = monteCarlo->domain[domain_index];

        for ( int cell_index = 0; cell_index < domain.cell_state.size(); cell_index++ )
        {
            MC_Cell_State &cell = domain.cell_state[cell_index];
            double cell_weight_particles = cell._volume * source_rate[cell._material] * monteCarlo->time_info->time_step;
            double cell_num_particles_float = cell_weight_particles / source_particle_weight;
            int cell_num_particles = (int)cell_num_particles_float;

            //Can Make this parallel - have an optimization from Leopold to add still
            for ( int particle_index = 0; particle_index < cell_num_particles; particle_index++ )
            {
                MC_Particle particle;

                uint64_t random_number_seed;

                ATOMIC_CAPTURE( cell._sourceTally, 1, random_number_seed );

                random_number_seed += cell._id;

                particle.random_number_seed = rngSpawn_Random_Number_Seed(&random_number_seed);
                particle.identifier = random_number_seed;

                MCT_Generate_Coordinate_3D_G(&particle.random_number_seed, domain_index, cell_index, particle.coordinate, monteCarlo);

                particle.direction_cosine.Sample_Isotropic(&particle.random_number_seed);

                // sample energy uniformly from [eMin, eMax] MeV
                particle.kinetic_energy = (monteCarlo->_params.simulationParams.eMax - monteCarlo->_params.simulationParams.eMin)*
                                rngSample(&particle.random_number_seed) + monteCarlo->_params.simulationParams.eMin;

                double speed = Get_Speed_From_Energy(particle.kinetic_energy);

                particle.velocity.x = speed * particle.direction_cosine.alpha;
                particle.velocity.y = speed * particle.direction_cosine.beta;
                particle.velocity.z = speed * particle.direction_cosine.gamma;

                particle.domain = domain_index;
                particle.cell   = cell_index;
                particle.task   = task_index;
                particle.weight = source_particle_weight;

                double randomNumber = rngSample(&particle.random_number_seed);
                particle.num_mean_free_paths = -1.0*log(randomNumber);

                randomNumber = rngSample(&particle.random_number_seed);
                particle.time_to_census = monteCarlo->time_info->time_step * randomNumber;

                MC_Base_Particle base_particle( particle );

                monteCarlo->_particleVaultContainer->addProcessingParticle( base_particle, processing_index );

                particle_count++;

                ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[particle.task]._source);
            }
        }
    }

#if 0 
    // Check for duplicate particle random number seeds.
    std::vector<uint64_t> particle_seeds;
    int task_index = 0;
    //for ( int task_index = 0; task_index < num_threads; task_index++ )
    {
       ParticleVault& particleVault = monteCarlo->_particleVaultContainer->getTaskProcessingVault(task_index);
       
       uint64_t currentNumParticles = particleVault.size();
       for (int particleIndex = 0; particleIndex < currentNumParticles; particleIndex++)
       {
	  MC_Base_Particle &currentParticle = particleVault[particleIndex];
	  particle_seeds.push_back(currentParticle.random_number_seed);
       }
    }

    std::sort(particle_seeds.begin(), particle_seeds.end());
    uint64_t num_dupl = 0;
    for (size_t pi_index = 0; pi_index<particle_seeds.size()-1; pi_index++)
    {
      if (particle_seeds[pi_index] == particle_seeds[pi_index+1])
      {
 	 num_dupl++;
 	 printf("*** found duplicate particle random number seed= %ull (%ull) at index pi_index= %d\n", 
		particle_seeds[pi_index], particle_seeds[pi_index+1], pi_index);
      }
    }
    printf("Number of duplicate random number seeds= %ull \n", num_dupl);
#endif
}


namespace
{
   double Get_Speed_From_Energy(double energy)
   {
      static const double rest_mass_energy = PhysicalConstants::_neutronRestMassEnergy;
      static const double speed_of_light  = PhysicalConstants::_speedOfLight;


      return speed_of_light * sqrt(energy * (energy + 2.0*(rest_mass_energy)) /
                                   ((energy + rest_mass_energy) * (energy + rest_mass_energy)));
   }
}
