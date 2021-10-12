/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef COLLISION_EVENT_HH
#define COLLISION_EVENT_HH

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

class MonteCarlo;
class MC_Particle;

inline HOST_DEVICE
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

inline HOST_DEVICE
bool CollisionEvent(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int tally_index, int particle_index, int * tallyArray)
{
   #ifdef  __HIP_DEVICE_COMPILE__
   const MC_Cell_State &cell = monteCarlo->domain_d[mc_particle.domain].cell_state[mc_particle.cell];
   #else
   const MC_Cell_State &cell = monteCarlo->domain[mc_particle.domain].cell_state[mc_particle.cell];
   #endif

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
   #ifdef  __HIP_DEVICE_COMPILE__
      int numIsos = (int)monteCarlo->_material_d[globalMatIndex]._isosize;
   #else
      int numIsos = (int)monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso.size();
   #endif
   
   for (int isoIndex = 0; isoIndex < numIsos && currentCrossSection >= 0; isoIndex++)
   {
      #ifdef  __HIP_DEVICE_COMPILE__
      int uniqueNumber = monteCarlo->_material_d[globalMatIndex]._iso[isoIndex]._gid;
      int numReacts = monteCarlo->_nuclearData_d->getNumberReactions(uniqueNumber);
      #else
      int uniqueNumber = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._gid;
      int numReacts = monteCarlo->_nuclearData->getNumberReactions(uniqueNumber);
      #endif
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
   #ifdef  __HIP_DEVICE_COMPILE__
   double mat_mass = monteCarlo->_material_d[globalMatIndex]._mass;
   monteCarlo->_nuclearData_d->_isotopes[selectedUniqueNumber]._species[0]._reactions[selectedReact].sampleCollision(
      mc_particle.kinetic_energy, mat_mass, &energyOut[0], &angleOut[0], nOut, &(mc_particle.random_number_seed), MAX_PRODUCTION_SIZE );
   #else
   double mat_mass = monteCarlo->_materialDatabase->_mat[globalMatIndex]._mass;
   monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0]._reactions[selectedReact].sampleCollision(
      mc_particle.kinetic_energy, mat_mass, &energyOut[0], &angleOut[0], nOut, &(mc_particle.random_number_seed), MAX_PRODUCTION_SIZE );
   #endif


   //--------------------------------------------------------------------------------------------------------------
   //  Post-Collision Phase 1:
   //    Tally the collision
   //--------------------------------------------------------------------------------------------------------------

   // Set the reaction for this particle.
   #ifdef __HIP_DEVICE_COMPILE__
   ATOMIC_UPDATE( tallyArray[tally_index*NUM_TALLIES+3]);
   #else
   ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._collision );
   #endif


   #ifdef  __HIP_DEVICE_COMPILE__
   NuclearDataReaction::Enum reactionType = (NuclearDataReaction::Enum)monteCarlo->_nuclearData_d->_isotopes[selectedUniqueNumber]._species[0].\
           _reactions[selectedReact]._reactionType;
   #else
   NuclearDataReaction::Enum reactionType = monteCarlo->_nuclearData->_isotopes[selectedUniqueNumber]._species[0].\
           _reactions[selectedReact]._reactionType;
   #endif

   switch (reactionType)
   {
      case NuclearDataReaction::Scatter:
	 #ifdef __HIP_DEVICE_COMPILE__
         ATOMIC_UPDATE( tallyArray[tally_index*NUM_TALLIES+4]);
         #else
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._scatter);
         #endif
         break;
      case NuclearDataReaction::Absorption:
	 #ifdef __HIP_DEVICE_COMPILE__
         ATOMIC_UPDATE( tallyArray[tally_index*NUM_TALLIES+5]);
         #else
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._absorb);
         #endif
         break;
      case NuclearDataReaction::Fission:
	 #ifdef __HIP_DEVICE_COMPILE__
         ATOMIC_UPDATE( tallyArray[tally_index*NUM_TALLIES+6]);
         ATOMIC_ADD( tallyArray[tally_index*NUM_TALLIES+7],nOut);
         #else
         ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._fission);
         ATOMIC_ADD( monteCarlo->_tallies->_balanceTask[tally_index]._produce, nOut);
         #endif
         break;
      case NuclearDataReaction::Undefined:
         #ifdef DEBUG
         printf("reactionType invalid\n");
         #endif
         qs_assert(false);
   }

   if( nOut == 0 ){ return false;}

   for (int secondaryIndex = 1; secondaryIndex < nOut; secondaryIndex++)
   {
        // Newly created particles start as copies of their parent
        MC_Particle secondaryParticle = mc_particle;
        secondaryParticle.random_number_seed = rngSpawn_Random_Number_Seed(&mc_particle.random_number_seed);
        secondaryParticle.identifier = secondaryParticle.random_number_seed;
        updateTrajectory( energyOut[secondaryIndex], angleOut[secondaryIndex], secondaryParticle );

        //Atomic capture will be called here
        monteCarlo->_particleVaultContainer->addExtraParticle(secondaryParticle);
   }

   updateTrajectory( energyOut[0], angleOut[0], mc_particle);

   // If a fission reaction produces secondary particles we also add the original
   // particle to the "extras" that we will handle later.  This avoids the 
   // possibility of a particle doing multiple fission reactions in a single
   // kernel invocation and overflowing the extra storage with secondary particles.
   if ( nOut > 1 ) 
   {
        //Atomic capture will be called here
        monteCarlo->_particleVaultContainer->addExtraParticle(mc_particle);
   }

   //If we are still tracking this particle the update its energy group
   #ifdef  __HIP_DEVICE_COMPILE__
   mc_particle.energy_group = monteCarlo->_nuclearData_d->getEnergyGroup(mc_particle.kinetic_energy);
   #else
   mc_particle.energy_group = monteCarlo->_nuclearData->getEnergyGroup(mc_particle.kinetic_energy);
   #endif

   return nOut == 1;
}

HOST_DEVICE_END

#endif

