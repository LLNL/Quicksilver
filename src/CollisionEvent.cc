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

//----------------------------------------------------------------------------------------------------------------------
//  Routine MC_Collision_Event determines the isotope, reaction and secondary (projectile)
//  particle characteristics for a collision event.
//
//  Return true if the particle will continue.
//----------------------------------------------------------------------------------------------------------------------

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
   double energyOut[4];
   double angleOut[4];
   int energy_angle_size = 0;
   double mat_mass = monteCarlo->_materialDatabase->_mat[globalMatIndex]._mass;

   monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0]._reactions[selectedReact].sampleCollision(
      mc_particle.kinetic_energy, mat_mass, &energyOut[0], &angleOut[0], &energy_angle_size, &(mc_particle.random_number_seed) );

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
         ATOMIC_ADD( monteCarlo->_tallies->_balanceTask[tally_index]._produce, energy_angle_size);
         break;
      case NuclearDataReaction::Undefined:
         printf("reactionType invalid\n");
         qs_assert(false);
   }

   DirectionCosine referenceDirectionCosine = mc_particle.direction_cosine;
   for (int secondaryIndex = 0; secondaryIndex < energy_angle_size; secondaryIndex++)
   {
      // Copy mc_particle into secondaryParticle buffer
      MC_Particle secondaryParticle = mc_particle;
      // Assign a pointer to this buffer
      MC_Particle *currentParticle = &secondaryParticle;
      // If this is the first particle, just update mc_particle instead and ignore buffer
      if (secondaryIndex == 0) { currentParticle = &mc_particle; }

      currentParticle->kinetic_energy = energyOut[secondaryIndex];
      currentParticle->direction_cosine = referenceDirectionCosine;
      double cosTheta = angleOut[secondaryIndex];
      randomNumber = rngSample(&mc_particle.random_number_seed);
      double phi = 2 * 3.14159265 * randomNumber;
      double sinPhi = sin(phi);
      double cosPhi = cos(phi);
      double sinTheta = sqrt((1.0 - (cosTheta*cosTheta)));
      currentParticle->direction_cosine.Rotate3DVector(sinTheta, cosTheta, sinPhi, cosPhi);
      double speed = (PhysicalConstants::_speedOfLight *
              sqrt((1.0 - ((PhysicalConstants::_neutronRestMassEnergy *
              PhysicalConstants::_neutronRestMassEnergy) /
              ((energyOut[secondaryIndex] + PhysicalConstants::_neutronRestMassEnergy) *
              (energyOut[secondaryIndex] + PhysicalConstants::_neutronRestMassEnergy))))));
      currentParticle->velocity.x = speed * mc_particle.direction_cosine.alpha;
      currentParticle->velocity.y = speed * mc_particle.direction_cosine.beta;
      currentParticle->velocity.z = speed * mc_particle.direction_cosine.gamma;

      randomNumber = rngSample(&mc_particle.random_number_seed);
      currentParticle->num_mean_free_paths = -1.0*log(randomNumber);
      if( energy_angle_size > 1 )
      {
         if (secondaryIndex > 0)
         {
            currentParticle->random_number_seed = rngSpawn_Random_Number_Seed(&mc_particle.random_number_seed);
	        currentParticle->identifier = currentParticle->random_number_seed;
         }
            monteCarlo->_particleVaultContainer->addExtraParticle(*currentParticle);
      }
   }

   return energy_angle_size == 1;
}

HOST_DEVICE_END

