#ifndef PARTICLEVAULTCONTAINER_HH
#define PARTICLEVAULTCONTAINER_HH

#include "DeclareMacro.hh"

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
    uint64_t getNumExtraVaults(){ return _numExtraVaults; }

    uint64_t processingSize(){ return _processingVault.size(); }
    uint64_t processedSize(){ return _processedVault.size(); }

    //Returns the ParticleVault that is currently pointed too 
    //by index listed
    ParticleVault* getTaskProcessingVault(uint64_t vaultIndex);
    ParticleVault* getTaskProcessedVault( uint64_t vaultIndex);

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

#endif
