/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "EnergySpectrum.hh"
#include "MonteCarlo.hh"
#include "ParticleVault.hh"
#include "ParticleVaultContainer.hh"
#include "utilsMpi.hh"
#include "MC_Processor_Info.hh"
#include "Parameters.hh"
#include <string>

using std::string;

void EnergySpectrum::UpdateSpectrum(MonteCarlo* monteCarlo)
{
    if( _fileName == "" ) return;

    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processingSize(); ii++)
    {
        ParticleVault* processing = monteCarlo->_particleVaultContainer->getTaskProcessingVault( ii );
        for( uint64_t jj = 0; jj < processing->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processing, jj);
            _censusEnergySpectrum[mc_particle.energy_group]++;
        }
    }
    for( uint64_t ii = 0; ii < monteCarlo->_particleVaultContainer->processedSize(); ii++)
    {
        ParticleVault* processed = monteCarlo->_particleVaultContainer->getTaskProcessedVault( ii );
        for( uint64_t jj = 0; jj < processed->size(); jj++ )
        {
            MC_Particle mc_particle;
            MC_Load_Particle(monteCarlo, mc_particle, processed, jj);
            _censusEnergySpectrum[mc_particle.energy_group]++;
        }
    }
}

void EnergySpectrum::PrintSpectrum(MonteCarlo* monteCarlo)
{
    if( _fileName == "" ) return;

    const int count = monteCarlo->_nuclearData->_energies.size();
    uint64_t *sumHist = new uint64_t[ count ]();

    mpiAllreduce( _censusEnergySpectrum.data(), sumHist, count, MPI_INT64_T, MPI_SUM, monteCarlo->processor_info->comm_mc_world );

    if( monteCarlo->processor_info->rank == 0 )
    {
        _fileName += ".dat";
        FILE* spectrumFile;
        spectrumFile = fopen( _fileName.c_str(), "w" );

        for( int ii = 0; ii < 230; ii++ )
        {
            fprintf( spectrumFile, "%d\t%g\t%lu\n", ii, monteCarlo->_nuclearData->_energies[ii], sumHist[ii] );
        }

        fclose( spectrumFile );
    }
    delete []sumHist;
}
