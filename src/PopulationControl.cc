#include "PopulationControl.hh"
#include "MC_Processor_Info.hh"
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Particle.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "utilsMpi.hh"
#include "NVTX_Range.hh"
#include <vector>

namespace
{
   void PopulationControlGuts(const double splitRRFactor, 
                              uint64_t currentNumParticles,
                              ParticleVaultContainer* my_particle_vault,
                              Balance& taskBalance);
}

void PopulationControl(MonteCarlo* monteCarlo, bool loadBalance)
{
    NVTX_Range range("PopulationControl");

    uint64_t targetNumParticles = monteCarlo->_params.simulationParams.nParticles;
    uint64_t globalNumParticles = 0;
    uint64_t localNumParticles = monteCarlo->_particleVaultContainer->sizeProcessing();
   
    if (loadBalance)
    {
        // If we are parallel, we will have one domain per mpi processs.  The targetNumParticles is across
        // all MPI processes, so we need to divide by the number or ranks to get the per-mpi-process number targetNumParticles
        targetNumParticles = ceil((double)targetNumParticles / (double)mcco->processor_info->num_processors );

        //NO LONGER SPLITING VAULTS BY THREADS
//        // If we are threaded, targetNumParticles should be divided by the number of threads (tasks) to balance
//        // the particles across the thread level vaults.
//        targetNumParticles = ceil((double)targetNumParticles / (double)mcco->processor_info->num_tasks);
    }
    else
    {
        mpiAllreduce(&localNumParticles, &globalNumParticles, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    }
     
    Balance & taskBalance = monteCarlo->_tallies->_balanceTask[0];

    double splitRRFactor = 1.0;
    if (loadBalance)
    {
        int currentNumParticles = localNumParticles;
        if (currentNumParticles != 0)
            splitRRFactor = (double)targetNumParticles / (double)currentNumParticles;
        else
            splitRRFactor = 1.0;
    }
    else
    {
        splitRRFactor = (double)targetNumParticles / (double)globalNumParticles;
    }

    if (splitRRFactor != 1.0)  // no need to split if population is already correct.
        PopulationControlGuts(splitRRFactor, localNumParticles, monteCarlo->_particleVaultContainer, taskBalance);

    monteCarlo->_particleVaultContainer->collapseProcessing();

    return;
}


namespace
{
void PopulationControlGuts(const double splitRRFactor, uint64_t currentNumParticles, ParticleVaultContainer* my_particle_vault, Balance& taskBalance)
{
    uint64_t vault_size = my_particle_vault->getVaultSize();
    uint64_t fill_vault_index = currentNumParticles / vault_size;

    // March backwards through the vault so killed particles doesn't mess up the indexing
    for (int particleIndex = currentNumParticles-1; particleIndex >= 0; particleIndex--)
    {
        uint64_t vault_index = particleIndex / vault_size; 

        ParticleVault& taskProcessingVault = *( my_particle_vault->getTaskProcessingVault(vault_index) );

        uint64_t taskParticleIndex = particleIndex%vault_size;

        MC_Base_Particle &currentParticle = taskProcessingVault[taskParticleIndex];
        double randomNumber = rngSample(&currentParticle.random_number_seed);
        if (splitRRFactor < 1)
        {
            if (randomNumber > splitRRFactor)
            {
                // Kill
	            taskProcessingVault.eraseSwapParticle(taskParticleIndex);
	            taskBalance._rr++; 
	        }
	        else
	        {
	            currentParticle.weight /= splitRRFactor;
	        }
        }
        else if (splitRRFactor > 1)
        {
            // Split
	        int splitFactor = (int)floor(splitRRFactor);
	        if (randomNumber > (splitRRFactor - splitFactor)) { splitFactor--; }
	  
	        currentParticle.weight /= splitRRFactor;
	        MC_Base_Particle splitParticle = currentParticle;
	  
	        for (int splitFactorIndex = 0; splitFactorIndex < splitFactor; splitFactorIndex++)
	        {
	            taskBalance._split++;
	     
	            splitParticle.random_number_seed = rngSpawn_Random_Number_Seed(
			        &currentParticle.random_number_seed);
	            splitParticle.identifier = splitParticle.random_number_seed;

                my_particle_vault->addProcessingParticle( splitParticle, fill_vault_index );

	        }
        }
    }
}
} // anonymous namespace


// Roulette low-weight particles relative to the source particle weight.
void RouletteLowWeightParticles(MonteCarlo* monteCarlo)
{
    NVTX_Range range("RouletteLowWeightParticles");

    const double lowWeightCutoff = monteCarlo->_params.simulationParams.lowWeightCutoff;

    if (lowWeightCutoff > 0.0)
    {

        uint64_t currentNumParticles = monteCarlo->_particleVaultContainer->sizeProcessing();
        uint64_t vault_size          = monteCarlo->_particleVaultContainer->getVaultSize();

        Balance& taskBalance = monteCarlo->_tallies->_balanceTask[0];

	    // March backwards through the vault so killed particles don't mess up the indexing
	    const double source_particle_weight = monteCarlo->source_particle_weight;
	    const double weightCutoff = lowWeightCutoff*source_particle_weight;

	    for ( int64_t particleIndex = currentNumParticles-1; particleIndex >= 0; particleIndex--)
	    {
            uint64_t vault_index = particleIndex / vault_size; 

            ParticleVault& taskProcessingVault = *(monteCarlo->_particleVaultContainer->getTaskProcessingVault(vault_index));
            uint64_t taskParticleIndex = particleIndex%vault_size;
	        MC_Base_Particle &currentParticle = taskProcessingVault[taskParticleIndex];

	        if (currentParticle.weight <= weightCutoff)
	        {
	            double randomNumber = rngSample(&currentParticle.random_number_seed);
	            if (randomNumber <= lowWeightCutoff)
	            {
		            // The particle history continues with an increased weight.
		            currentParticle.weight /= lowWeightCutoff;
	            }
	            else
	            {
		            // Kill
		            taskProcessingVault.eraseSwapParticle(taskParticleIndex);
		            taskBalance._rr++;
	            } 
	        }
	    }
        monteCarlo->_particleVaultContainer->collapseProcessing();
    }
}
