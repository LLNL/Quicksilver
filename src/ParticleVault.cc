#include "ParticleVault.hh"
#include "MC_Processor_Info.hh"
#include "Globals.hh"

#if 0
void ParticleVault::
cleanVault( int end_index )
{
    int s1 = end_index;
    int s2 = _particles.size();

    int starting_point = s2 - ( ( s1<(s2-s1)) ? s1 : (s2-s1));

#if defined HAVE_OPENMP_TARGET
    int USE_GPU = mcco->processor_info->use_gpu;
    #pragma omp target teams distribute parallel for thread_limit(64) if(target:USE_GPU) 
#endif
    for( int ii = starting_point; ii < s2; ii++ )
    {
        qs_assert( _particles[ii-starting_point].species == -1 );
        _particles[ii-starting_point] = _particles[ii];
        _particles[ii].species = -1;
    }

    _particles.eraseEnd( _particles.size() - end_index );
}
#endif

void ParticleVault::
collapse( size_t fill_size, ParticleVault* vault2 )
{
    //The entirety of vault 2 fits in the space available in this vault 
    if( vault2->size() < fill_size )
    {
        this->append( *vault2 );
        vault2->clear();
    }
    else //Fill in what we can untill either vault2 is empty or we have filled this vault
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
