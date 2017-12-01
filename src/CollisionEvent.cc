#include "CollisionEvent.hh"
#include "MC_Particle.hh"
#include "NuclearData.hh"
#include "DirectionCosine.hh"
#include "MonteCarlo.hh"
#include "MC_Cell_State.hh"
#include "MaterialDatabase.hh"
#include "MacroscopicCrossSection.hh"
#include "MC_Base_Particle.hh"
#include "ParticleVaultContainer.hh"
#include "PhysicalConstants.hh"
#include "DeclareMacro.hh"
#include "AtomicMacro.hh"

#define MAX_PRODUCTION_SIZE 4

//----------------------------------------------------------------------------------------------------------------------
//  Routine MC_Collision_Event determines the isotope, reaction and secondary (projectile)
//  particle characteristics for a collision event.
//
//  Return true if the particle will continue.
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE
void updateTrajectory( double energy, double angle, MC_Particle& particle )
{
    particle.kinetic_energy = energy;
    double cosTheta = angle;
    double randomNumber = rngSample(&particle.random_number_seed);
    double phi = 2 * 3.14159265 * randomNumber;
    double sinPhi = sin(phi);
    double cosPhi = cos(phi);
    double sinTheta = sqrt((1.0 - (cosTheta*cosTheta)));
    particle.direction_cosine.Rotate3DVector(sinTheta, cosTheta, sinPhi, cosPhi);
    double speed = (PhysicalConstants::_speedOfLight *
            sqrt((1.0 - ((PhysicalConstants::_neutronRestMassEnergy *
            PhysicalConstants::_neutronRestMassEnergy) /
            ((energy + PhysicalConstants::_neutronRestMassEnergy) *
            (energy + PhysicalConstants::_neutronRestMassEnergy))))));
    particle.velocity.x = speed * particle.direction_cosine.alpha;
    particle.velocity.y = speed * particle.direction_cosine.beta;
    particle.velocity.z = speed * particle.direction_cosine.gamma;
    randomNumber = rngSample(&particle.random_number_seed);
    particle.num_mean_free_paths = -1.0*log(randomNumber);
}
HOST_DEVICE_END

HOST_DEVICE

bool CollisionEvent(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int tally_index)
{
   const MC_Cell_State &cell = monteCarlo->domain[mc_particle.domain].cell_state[mc_particle.cell];

   int globalMatIndex = cell._material;

   //------------------------------------------------------------------------------------------------------------------
   //    Pick the isotope and reaction.
   //------------------------------------------------------------------------------------------------------------------
   double randomNumber = rngSample(&mc_particle.random_number_seed);
   double totalCrossSection = mc_particle.totalCrossSection;
   double currentCrossSection = totalCrossSection * randomNumber;
   int selectedIso = -1;
   int selectedUniqueNumber = -1;
   int selectedReact = -1;
   int numIsos = (int)monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso.size();
   
   for (int isoIndex = 0; isoIndex < numIsos && currentCrossSection >= 0; isoIndex++)
   {
      int uniqueNumber = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._gid;
      int numReacts = monteCarlo->_nuclearData->getNumberReactions(uniqueNumber);
      for (int reactIndex = 0; reactIndex < numReacts; reactIndex++)
      {
         currentCrossSection -= macroscopicCrossSection(monteCarlo, reactIndex, mc_particle.domain, mc_particle.cell,
                   isoIndex, mc_particle.energy_group);
         if (currentCrossSection < 0)
         {
            selectedIso = isoIndex;
            selectedUniqueNumber = uniqueNumber;
            selectedReact = reactIndex;
            break;
         }
      }
   }
   qs_assert(selectedIso != -1);

   //------------------------------------------------------------------------------------------------------------------
   //    Do the collision.
   //------------------------------------------------------------------------------------------------------------------
   double energyOut[MAX_PRODUCTION_SIZE];
   double angleOut[MAX_PRODUCTION_SIZE];
   int nOut = 0;
   double mat_mass = monteCarlo->_materialDatabase->_mat[globalMatIndex]._mass;

   monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0]._reactions[selectedReact].sampleCollision(
      mc_particle.kinetic_energy, mat_mass, &energyOut[0], &angleOut[0], nOut, &(mc_particle.random_number_seed), MAX_PRODUCTION_SIZE );

   //--------------------------------------------------------------------------------------------------------------
   //  Post-Collision Phase 1:
   //    Tally the collision
   //--------------------------------------------------------------------------------------------------------------

   // Set the reaction for this particle.
   ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._collision );
   NuclearDataReaction::Enum reactionType = monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0].\
           _reactions[selectedReact]._reactionType;
   switch (reactionType)
   {
      case NuclearDataReaction::Scatter:
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._scatter);
         break;
      case NuclearDataReaction::Absorption:
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._absorb);
         break;
      case NuclearDataReaction::Fission:
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._fission);
         ATOMIC_ADD( monteCarlo->_tallies->_balanceTask[tally_index]._produce, nOut);
         break;
      case NuclearDataReaction::Undefined:
         printf("reactionType invalid\n");
         qs_assert(false);
   }

   if( nOut == 0 ) return false;

   for (int secondaryIndex = 1; secondaryIndex < nOut; secondaryIndex++)
   {
        // Newly created particles start as copies of their parent
        MC_Particle secondaryParticle = mc_particle;
        secondaryParticle.random_number_seed = rngSpawn_Random_Number_Seed(&mc_particle.random_number_seed);
        secondaryParticle.identifier = secondaryParticle.random_number_seed;
        updateTrajectory( energyOut[secondaryIndex], angleOut[secondaryIndex], secondaryParticle );
        monteCarlo->_particleVaultContainer->addExtraParticle(secondaryParticle);
   }

   updateTrajectory( energyOut[0], angleOut[0], mc_particle);

   // If a fission reaction produces secondary particles we also add the original
   // particle to the "extras" that we will handle later.  This avoids the 
   // possibility of a particle doing multiple fission reactions in a single
   // kernel invocation and overflowing the extra storage with secondary particles.
   if ( nOut > 1 ) 
       monteCarlo->_particleVaultContainer->addExtraParticle(mc_particle);

   //If we are still tracking this particle the update its energy group
   mc_particle.energy_group = monteCarlo->_nuclearData->getEnergyGroup(mc_particle.kinetic_energy);

   return nOut == 1;
}

HOST_DEVICE_END

