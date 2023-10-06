#include "ParticleVault.hh"
#include "MC_Processor_Info.hh"
#include "Globals.hh"


void ParticleVault::
collapse( size_t fill_size, ParticleVault* vault2 )
{
    //The entirety of vault 2 fits in the space available in this vault 
    if( vault2->size() < fill_size )
    {
        this->append( *vault2 );
        vault2->clear();
    }
    else //Fill in what we can until either vault2 is empty or we have filled this vault
    {
        bool notEmpty = false;
        uint64_t fill = 0;
        do
        {
            MC_Base_Particle base_particle;
            notEmpty = vault2->popBaseParticle( base_particle );
            if( notEmpty )
            {
                this->pushBaseParticle( base_particle );
                fill++;
            }
        }while( notEmpty && fill < fill_size);
    }
}
