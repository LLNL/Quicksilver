/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef PARTICLEVAULTCONTAINER_HH
#define PARTICLEVAULTCONTAINER_HH

#include "DeclareMacro.hh"
#include "ParticleVault.hh"
#include "SendQueue.hh"
#include "MemoryControl.hh"
#include "qs_assert.hh"
#include "portability.hh"
#include "QS_Vector.hh"
#include <vector>

//---------------------------------------------------------------
// ParticleVaultContainer is a container of ParticleVaults. 
// These Vaults are broken down into user defined chunks that can 
// be used to overlap asynchronous MPI with the tracking kernel.
//
// Facilities for storing Processing, Processed, and Extra vaults 
// are controled by the ParticleVaultContainer. As well as the 
// sendQueue, which lists the particles that must be send to 
// another process via MPI
//--------------------------------------------------------------

class MC_Base_Particle;
class MC_Particle;
class ParticleVault;
class SendQueue;

typedef unsigned long long int uint64_cu;

class ParticleVaultContainer
{
  public:
    
    //Constructor
    ParticleVaultContainer( uint64_t vault_size, 
        uint64_t num_vaults, uint64_t num_extra_vaults );

    //Destructor
    ~ParticleVaultContainer();

    //Basic Getters
    uint64_t getVaultSize(){      return _vaultSize; }
    
    HOST_DEVICE
    uint64_t getNumExtraVaults(){ return _numExtraVaults; }
    HOST_DEVICE_END

    uint64_t processingSize(){ return _processingVault.size(); }
    uint64_t processedSize(){ return _processedVault.size(); }

    //Returns the ParticleVault that is currently pointed too 
    //by index listed
    ParticleVault* getTaskProcessingVault(uint64_t tallyArray);
    ParticleVault* getTaskProcessedVault( uint64_t tallyArray);

    //Returns the index to the first empty Processed Vault
    uint64_t getFirstEmptyProcessedVault();

    //Returns a pointer to the Send Queue
    HOST_DEVICE
    SendQueue* getSendQueue();
    HOST_DEVICE_END

    //Counts Particles in all vaults
    uint64_t sizeProcessing();
    uint64_t sizeProcessed();
    uint64_t sizeExtra();

    //Collapses Particles down into lowest amount of vaults as 
    //needed to hold them removes all but the last parially 
    //filled vault
    void collapseProcessing();
    void collapseProcessed();

    //Swaps the particles in Processed for the empty vaults in 
    //Processing
    void swapProcessingProcessedVaults();

    //Adds a particle to the processing particle vault
    void addProcessingParticle( MC_Base_Particle &particle, uint64_t &fill_vault_index );
    //Adds a particle to the extra particle vault
    HOST_DEVICE
    void addExtraParticle( MC_Particle &particle );
    HOST_DEVICE_END
 
    HOST_DEVICE
    void addExtraParticle( MC_Particle &particle, int * tallyArray,int * particleindex);
    HOST_DEVICE_END

    uint64_t getextraVaultIndex();

    ParticleVault * getExtraVault(int index);


    //Pushes particles from Extra Vaults onto the Processing 
    //Vault list
    void cleanExtraVaults();

  private:
    
    //The Size of the ParticleVaults (fixed at runtime for 
    //each run)
    uint64_t _vaultSize;

    //The number of Extra Vaults needed based on hueristics 
    //(fixed at runtime for each run)
    uint64_t _numExtraVaults;

    //A running index for the number of particles int the extra 
    //particle vaults
    uint64_cu _extraVaultIndex;

    //The send queue - stores particle index and neighbor index 
    //for any particles that hit (TRANSIT_OFF_PROCESSOR) 
    SendQueue *_sendQueue;

    //The list of active particle vaults (size - grow-able)
    std::vector<ParticleVault*> _processingVault;

    //The list of censused particle vaults (size - grow-able)
    std::vector<ParticleVault*> _processedVault;

    //The list of extra particle vaults (size - fixed)
    qs_vector<ParticleVault*>   _extraVault;
     
};

//--------------------------------------------------------------
//------------getSendQueue--------------------------------------
//Returns a pointer to the Send Queue
//--------------------------------------------------------------
inline HOST_DEVICE
SendQueue* ParticleVaultContainer::
getSendQueue()
{
    return this->_sendQueue;
}
HOST_DEVICE_END



inline HOST_DEVICE
void ParticleVaultContainer::
addExtraParticle( MC_Particle &particle)
{
    uint64_cu index = 0;
    ATOMIC_CAPTURE( this->_extraVaultIndex, 1, index );
    uint64_t vault = index / this->_vaultSize;
    _extraVault[vault]->pushParticle( particle );
}
HOST_DEVICE_END



#endif
