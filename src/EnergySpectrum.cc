#include "EnergySpectrum.hh"
#include "MonteCarlo.hh"
#include "ParticleVault.hh"
#include "ParticleVaultContainer.hh"
#include "NuclearData.hh"
#include "utilsMpi.hh"
#include "MC_Processor_Info.hh"
#include "Parameters.hh"
#include <string>

using std::string;

void updateSpectrum( MonteCarlo* monteCarlo, uint64_t *hist )
{
    if( monteCarlo->_params.simulationParams.energySpectrum == "" ) return;

    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processingSize(); ii++)
    {
        ParticleVault* processing = monteCarlo->_particleVaultContainer->getTaskProcessingVault( ii );
        for( uint64_t jj = 0; jj < processing->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processing, jj);
            hist[mc_particle.energy_group]++;
        }
    }
    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processedSize(); ii++)
    {
        ParticleVault* processed = monteCarlo->_particleVaultContainer->getTaskProcessedVault( ii );
        for( uint64_t jj = 0; jj < processed->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processed, jj);
            hist[mc_particle.energy_group]++;
        }
    }
}

void printSpectrum( uint64_t *hist, MonteCarlo* monteCarlo)
{
    if( monteCarlo->_params.simulationParams.energySpectrum == "" ) return;

    const int count = monteCarlo->_nuclearData->_energies.size();
    uint64_t *sumHist = new uint64_t[ count ]();

    mpiAllreduce( hist, sumHist, count, MPI_INT64_T, MPI_SUM, monteCarlo->processor_info->comm_mc_world );

    if( monteCarlo->processor_info->rank == 0 )
    {
        string fileName = monteCarlo->_params.simulationParams.energySpectrum + ".dat";
        FILE* spectrumFile;
        spectrumFile = fopen( fileName.c_str(), "w" );

        for( int ii = 0; ii < 230; ii++ )
        {
            fprintf( spectrumFile, "%d\t%g\t%lu\n", ii, monteCarlo->_nuclearData->_energies[ii], sumHist[ii] );
        }

        fclose( spectrumFile );
    }
}
