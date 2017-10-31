#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "SendQueue.hh"
#include "MemoryControl.hh"
#include "qs_assert.hh"

//--------------------------------------------------------------
//------------ParticleVaultContainer Constructor----------------
//Sets up the fixed sized data and pre-allocates the minimum 
//needed for processing and processed vaults
//--------------------------------------------------------------

ParticleVaultContainer::
ParticleVaultContainer( uint64_t vault_size, 
                        uint64_t num_vaults, 
                        uint64_t num_extra_vaults )
: _vaultSize      ( vault_size       ),
  _numExtraVaults ( num_extra_vaults ),
  _extraVaultIndex( 0                ),
  _processingVault( num_vaults       ),
  _processedVault ( num_vaults       ),
  _extraVault     ( num_extra_vaults, VAR_MEM )
{

    //Allocate and reserve space for particles for each vault
    for( uint64_t vault = 0; vault < num_vaults; vault++ )
    {
        //Allocate Processing Vault
        _processingVault[vault] = 
            MemoryControl::allocate<ParticleVault>(1 ,VAR_MEM);
        _processingVault[vault]->reserve( vault_size );

        //Allocate Processed Vault
        _processedVault[vault]  = 
            MemoryControl::allocate<ParticleVault>(1 ,VAR_MEM);
        _processedVault[vault]->reserve( vault_size );
    }

    //Allocate and reserve space for particles for each extra vault
    for( uint64_t e_vault = 0; 
                  e_vault < num_extra_vaults; 
                  e_vault++ )
    {
        //Allocate Extra Vault
        _extraVault[e_vault] = 
            MemoryControl::allocate<ParticleVault>(1 ,VAR_MEM);
        _extraVault[e_vault]->reserve( vault_size );
    }

    _sendQueue = MemoryControl::allocate<SendQueue>(1 ,VAR_MEM);
    _sendQueue->reserve( vault_size );
}

//--------------------------------------------------------------
//------------ParticleVaultContainer Destructor-----------------
//Deletes memory allocaetd using the Memory Control class
//--------------------------------------------------------------

ParticleVaultContainer::
~ParticleVaultContainer()
{
    for( int64_t ii = _processingVault.size()-1; ii >= 0; ii-- )
    {
        MemoryControl::deallocate(_processingVault[ii], 1, VAR_MEM);
    }
    for( int64_t jj = _processedVault.size()-1; jj >= 0; jj-- )
    {
        MemoryControl::deallocate(_processedVault[jj], 1, VAR_MEM);
    }
    for( int64_t ii = _extraVault.size()-1; ii >= 0; ii-- )
    {
        MemoryControl::deallocate(_extraVault[ii], 1, VAR_MEM);
    }
    MemoryControl::deallocate( _sendQueue, 1, VAR_MEM );
}

//--------------------------------------------------------------
//------------getTaskProcessingVault----------------------------
//Returns a pointer to the Particle Vault in the processing list
//at the index provided
//--------------------------------------------------------------

ParticleVault* ParticleVaultContainer::
getTaskProcessingVault(uint64_t vaultIndex)
{
//   qs_assert(vaultIndex >= 0);
//   qs_assert(vaultIndex < _processingVault.size());
   return _processingVault[vaultIndex];
}

//--------------------------------------------------------------
//------------getTaskProcessedVault-----------------------------
//Returns a pointer to the Particle Vault in the processed list
//at the index provided
//--------------------------------------------------------------

ParticleVault* ParticleVaultContainer::
getTaskProcessedVault(uint64_t vaultIndex)
{
//   qs_assert(vaultIndex >= 0);
//   qs_assert(vaultIndex < _processedVault.size());
   return _processedVault[vaultIndex];
}

//--------------------------------------------------------------
//------------getFirstEmptyProcessedVault-----------------------
//Returns a pointer to the first empty Particle Vault in the 
//processed list
//--------------------------------------------------------------

uint64_t ParticleVaultContainer::
getFirstEmptyProcessedVault()
{
    uint64_t index = 0;

    while( _processedVault[index]->size() != 0 )
    {
        index++;
        if( index == _processedVault.size() )
        {
            ParticleVault* vault = MemoryControl::allocate<ParticleVault>(1,VAR_MEM);
            vault->reserve( _vaultSize );
            this->_processedVault.push_back(vault);
        }
    }

    return index;
}

//--------------------------------------------------------------
//------------getSendQueue--------------------------------------
//Returns a pointer to the Send Queue
//--------------------------------------------------------------
HOST_DEVICE
SendQueue* ParticleVaultContainer::
getSendQueue()
{
    return this->_sendQueue;
}
HOST_DEVICE_END

//--------------------------------------------------------------
//------------sizeProcessing------------------------------------
//returns the total number of particles in the processing vault
//--------------------------------------------------------------

uint64_t ParticleVaultContainer::
sizeProcessing()
{
    uint64_t sum_size = 0;
    for( uint64_t vault = 0; vault < _processingVault.size(); vault++ )
    {
        sum_size += _processingVault[vault]->size();
    }
    return sum_size;
}

//--------------------------------------------------------------
//------------sizeProcessed-------------------------------------
//returns the total number of particles in the processed vault
//--------------------------------------------------------------

uint64_t ParticleVaultContainer::
sizeProcessed()
{
    uint64_t sum_size = 0;
    for( uint64_t vault = 0; vault < _processedVault.size(); vault++ )
    {
        sum_size += _processedVault[vault]->size();
    }
    return sum_size;
}


//--------------------------------------------------------------
//------------sizeExtra-----------------------------------------
//returns the total number of particles in the extra vault
//--------------------------------------------------------------

uint64_t ParticleVaultContainer::
sizeExtra()
{
    uint64_t sum_size = 0;
    for( uint64_t vault = 0; vault < _extraVault.size(); vault++ )
    {
        sum_size += _extraVault[vault]->size();
    }
    return sum_size;
}

//--------------------------------------------------------------
//------------collapseProcessing--------------------------------
//Collapses the particles in the processing vault down to the
//first vaults needed to hold that number of particles
//--------------------------------------------------------------

void ParticleVaultContainer::
collapseProcessing()
{
    uint64_t size_processing = this->sizeProcessing(); 

    uint64_t num_vaults = this->_processingVault.size();

    uint64_t fill_vault_index = 0;
    uint64_t from_vault_index = num_vaults-1;

    while( fill_vault_index < from_vault_index )
    {
        if( _processingVault[fill_vault_index]->size() == this->_vaultSize )
        {
            fill_vault_index++;
        }
        else
        {
            if( this->_processingVault[from_vault_index]->size() == 0 )
            {
                from_vault_index--;
            }
            else
            {
                uint64_t fill_size = this->_vaultSize - this->_processingVault[fill_vault_index]->size();

                this->_processingVault[fill_vault_index]->collapse( fill_size, this->_processingVault[from_vault_index] );
            }
        }
    }
}

//--------------------------------------------------------------
//------------collapseProcessed---------------------------------
//Collapses the particles in the processed vault down to the
//first vaults needed to hold that number of particles
//--------------------------------------------------------------
    
void ParticleVaultContainer::
collapseProcessed()
{
    uint64_t size_processed = this->sizeProcessed(); 

    uint64_t num_vaults = this->_processedVault.size();

    uint64_t fill_vault_index = 0;
    uint64_t from_vault_index = num_vaults-1;

    while( fill_vault_index < from_vault_index )
    {
        if( _processedVault[fill_vault_index]->size() == this->_vaultSize )
        {
            fill_vault_index++;
        }
        else
        {
            if( this->_processedVault[from_vault_index]->size() == 0 )
            {
                from_vault_index--;
            }
            else
            {
                uint64_t fill_size = this->_vaultSize - this->_processedVault[fill_vault_index]->size();

                this->_processedVault[fill_vault_index]->collapse( fill_size, this->_processedVault[from_vault_index] );
            }
        }
    }
}

//--------------------------------------------------------------
//------------swapProcessingProcessedVaults---------------------
//Swaps the vaults from Processed that have particles in them
//with empty vaults from processing to prepair for the next
//cycle
//
//ASSUMPTIONS:: 
//  2) _processingVault is always empty of particles when this is
//      called
//--------------------------------------------------------------

void ParticleVaultContainer::
swapProcessingProcessedVaults()
{
    //Collapse Processed Vault to insure the particles are all
    //in the front of the list
    this->collapseProcessed();

    //start swapping from the beginning
    uint64_t processed_vault = 0;

    bool need_to_swap = (this->_processedVault[processed_vault]->size() > 0);

    while( need_to_swap )
    {
        std::swap( this->_processedVault[processed_vault], this->_processingVault[processed_vault] );
        processed_vault++;

        if( processed_vault == this->_processingVault.size() )
        {
            ParticleVault* vault = MemoryControl::allocate<ParticleVault>(1,VAR_MEM);
            vault->reserve( _vaultSize );
            this->_processingVault.push_back(vault);
        }

        if( processed_vault < this->_processedVault.size() )
        {
            need_to_swap = (this->_processedVault[processed_vault]->size() > 0);
        }
        else
        {
            need_to_swap = false;
        }
    }
}

//--------------------------------------------------------------
//------------addProcessingParticle-----------------------------
//Adds a particle to the processing particle vault
//--------------------------------------------------------------

void ParticleVaultContainer::
addProcessingParticle( MC_Base_Particle &particle, uint64_t &fill_vault_index )
{
    bool space = ( _processingVault[fill_vault_index]->size() < this->_vaultSize );
    while( !space )
    {
        fill_vault_index++;
        if( !(fill_vault_index < _processingVault.size()) )
        {
           ParticleVault* vault = MemoryControl::allocate<ParticleVault>(1,VAR_MEM);
           vault->reserve( this->_vaultSize );
           _processingVault.push_back(vault);
        }
        space = ( _processingVault[fill_vault_index]->size() < this->_vaultSize );
    }
    _processingVault[fill_vault_index]->pushBaseParticle(particle);
}

//--------------------------------------------------------------
//------------addExtraParticle----------------------------------
//adds a particle to the extra particle vaults (used in kernel)
//--------------------------------------------------------------
HOST_DEVICE
void ParticleVaultContainer::
addExtraParticle( MC_Particle &particle)
{
    uint64_cu index = 0;
    ATOMIC_CAPTURE( this->_extraVaultIndex, 1, index ); 
    uint64_t vault = index / this->_vaultSize;
    _extraVault[vault]->pushParticle( particle );
}
HOST_DEVICE_END

//--------------------------------------------------------------
//------------cleanExtraVaults----------------------------------
//Moves the particles from the _extraVault into the 
//_processedVault
//--------------------------------------------------------------

void ParticleVaultContainer::
cleanExtraVaults()
{
    uint64_t size_extra = this->sizeExtra();
    if( size_extra > 0 )
    {
        uint64_t num_extra  = (size_extra / this->_vaultSize) + ((size_extra%this->_vaultSize == 0) ? 0 : 1);

        uint64_t extra_index      = 0;
        uint64_t processing_index = 0;

        while( extra_index < num_extra )
        {

            if( this->_extraVault[extra_index]->size() == 0 )
            {
                extra_index++;
            }
            else
            {
                if( processing_index == this->_processingVault.size() )
                {
                    ParticleVault* vault = MemoryControl::allocate<ParticleVault>(1,VAR_MEM);
                    vault->reserve( _vaultSize );
                    this->_processingVault.push_back(vault);
                }
                else
                {
                    if( this->_processingVault[processing_index]->size() == this->_vaultSize )
                    {
                        processing_index++;
                    }
                    else
                    {
                        uint64_t fill_size = this->_vaultSize - this->_processingVault[processing_index]->size();

                        this->_processingVault[processing_index]->collapse( fill_size, this->_extraVault[extra_index] );
                    }
                }
            }
        }
    }
    _extraVaultIndex = 0;
}

