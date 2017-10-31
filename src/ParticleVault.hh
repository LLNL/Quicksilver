#ifndef PARTICLEVAULT_HH
#define PARTICLEVAULT_HH

#include "MC_Base_Particle.hh"
#include "QS_Vector.hh"
#include "DeclareMacro.hh"

#include <vector>

class ParticleVault
{
public:

   // Is the vault empty.
   bool empty() const {return _particles.empty();}

   // Get the size of the vault.
   HOST_DEVICE_CUDA
   size_t size() const {return _particles.size();}

   // Reserve the size for the container of particles.
   void reserve(size_t n)
   { 
       _particles.reserve(n,VAR_MEM); 
   }

   // Add all particles in a 2nd vault into this vault.
   void append (ParticleVault & vault2)
        { _particles.appendList( vault2._particles.size(), &vault2._particles[0] ); }

   void collapse( size_t fill_size, ParticleVault* vault2 );

   // Clear all particles from the vault
   void clear() { _particles.clear(); } 

   // Access particle at a given index.
   MC_Base_Particle& operator[](size_t n) {return _particles[n];}

   // Access a particle at a given index.
   const MC_Base_Particle& operator[](size_t n) const {return _particles[n];}

   // Put a particle into the vault, down casting its class.
   HOST_DEVICE_CUDA
   void pushParticle(MC_Particle &particle);

   // Put a base particle into the vault.
   HOST_DEVICE_CUDA
   void pushBaseParticle(MC_Base_Particle &base_particle);

   // Get a base particle from the vault.
   bool popBaseParticle(MC_Base_Particle &base_particle);

   // Get a particle from the vault.
   bool popParticle(MC_Particle &particle);

   // Get a particle from the vault 
   bool getBaseParticleComm(MC_Base_Particle &particle, int index);
   HOST_DEVICE_CUDA
   bool getParticle(MC_Particle &particle, int index);
   // Copy a particle back into the vault
   HOST_DEVICE_CUDA
   bool putParticle(MC_Particle particle, int index);

   // invalidates the particle in the vault at an index
   HOST_DEVICE_CUDA
   void invalidateParticle( int index );

#if 0
   // Remove all of the invalid particles form the _particles list
   void cleanVault(int end_index);
#endif

   // Swap vaults.
   void swapVaults(ParticleVault &vault);

   // Swaps this particle at index with last particle and resizes to delete it
   void eraseSwapParticle(int index);

private:

   // The container of particles.
   qs_vector<MC_Base_Particle> _particles;
};

// -----------------------------------------------------------------------
HOST_DEVICE_CUDA
inline void ParticleVault::
pushParticle(MC_Particle &particle)
{
    MC_Base_Particle base_particle(particle);
    size_t indx = _particles.atomic_Index_Inc(1);
    _particles[indx] = base_particle;
}

// -----------------------------------------------------------------------
HOST_DEVICE_CUDA
inline void ParticleVault::
pushBaseParticle(MC_Base_Particle &base_particle)
{
    int indx = _particles.atomic_Index_Inc(1);
    _particles[indx] = base_particle;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
popBaseParticle(MC_Base_Particle &base_particle)
{
   bool notEmpty = false;

#include "mc_omp_critical.hh"
{
   if (!empty())
   {
      base_particle = _particles.back();
      _particles.pop_back();
      notEmpty = true;
   }
}
    return notEmpty;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
popParticle(MC_Particle &particle)
{
   bool notEmpty = false;

#include "mc_omp_critical.hh"
{
   if (!empty())
   {
      MC_Base_Particle base_particle(_particles.back());
      _particles.pop_back();
      particle = MC_Particle(base_particle);
      notEmpty = true;
   }
}
   return notEmpty;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
getBaseParticleComm( MC_Base_Particle &particle, int index )
{
    if( size() > index )
    {
            particle = _particles[index];
            _particles[index].species = -1;
            return true;
    }
    else
    {
        qs_assert(false);
    }
    return false;
}

// -----------------------------------------------------------------------
   HOST_DEVICE_CUDA
inline bool ParticleVault::
getParticle( MC_Particle &particle, int index )
{
    qs_assert( size() > index );
    if( size() > index )
    {
            MC_Base_Particle base_particle( _particles[index] );
            particle = MC_Particle( base_particle );
            return true;
    }
    return false;
}

// -----------------------------------------------------------------------
inline bool ParticleVault::
putParticle(MC_Particle particle, int index)
{
    qs_assert( size() > index );
    if( size() > index )
    {
        MC_Base_Particle base_particle( particle );
        _particles[index] = base_particle;
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------
inline void ParticleVault::
invalidateParticle( int index )
{
    qs_assert( index >= 0 );
    qs_assert( index < _particles.size() );
    _particles[index].species = -1;
}

// -----------------------------------------------------------------------
inline void ParticleVault::
eraseSwapParticle(int index)
{
    #include "mc_omp_critical.hh"
    {
        _particles[index] = _particles.back();
        _particles.pop_back();
    }
}

// -----------------------------------------------------------------------
HOST_DEVICE
void MC_Load_Particle(MonteCarlo *mcco, MC_Particle &mc_particle, ParticleVault *particleVault, int particle_index);
HOST_DEVICE_END

#endif
