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

void EnergySpectrum::Allocate(string name, uint64_t size)
{
    fileName = name;

    if( fileName == "" ) return;

    CensusEnergySpectrum = new uint64_t [size];
}

void EnergySpectrum::UpdateSpectrum(MonteCarlo* monteCarlo)
{
    if( fileName == "" ) return;

    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processingSize(); ii++)
    {
        ParticleVault* processing = monteCarlo->_particleVaultContainer->getTaskProcessingVault( ii );
        for( uint64_t jj = 0; jj < processing->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processing, jj);
            CensusEnergySpectrum[mc_particle.energy_group]++;
        }
    }
    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processedSize(); ii++)
    {
        ParticleVault* processed = monteCarlo->_particleVaultContainer->getTaskProcessedVault( ii );
        for( uint64_t jj = 0; jj < processed->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processed, jj);
            CensusEnergySpectrum[mc_particle.energy_group]++;
        }
    }
}

void EnergySpectrum::PrintSpectrum(MonteCarlo* monteCarlo)
{
    if( fileName == "" ) return;

    const int count = monteCarlo->_nuclearData->_energies.size();
    uint64_t *sumHist = new uint64_t[ count ]();

    mpiAllreduce( CensusEnergySpectrum, sumHist, count, MPI_INT64_T, MPI_SUM, monteCarlo->processor_info->comm_mc_world );

    if( monteCarlo->processor_info->rank == 0 )
    {
        fileName += ".dat";
        FILE* spectrumFile;
        spectrumFile = fopen( fileName.c_str(), "w" );

        for( int ii = 0; ii < 230; ii++ )
        {
            fprintf( spectrumFile, "%d\t%g\t%lu\n", ii, monteCarlo->_nuclearData->_energies[ii], sumHist[ii] );
        }

        fclose( spectrumFile );
    }
}
