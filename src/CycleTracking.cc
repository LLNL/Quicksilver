/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "CycleTracking.hh"
#include "MonteCarlo.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Segment_Outcome.hh"
#include "CollisionEvent.hh"
#include "MC_Facet_Crossing_Event.hh"
#include "MCT.hh"
#include "DeclareMacro.hh"
#include "AtomicMacro.hh"
#include "macros.hh"
#include "qs_assert.hh"

void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault, int * tallyArray)
{
    MC_Particle mc_particle;

    // Copy a single particle from the particle vault into mc_particle
    MC_Load_Particle(monteCarlo, mc_particle, processingVault, particle_index);

    // set the particle.task to the index of the processed vault the particle will census into.
    mc_particle.task = 0;

    // loop over this particle until we cannot do anything more with it on this processor
    //CycleTrackingFunction( monteCarlo, mc_particle, particle_index, processingVault, processedVault);
    CycleTrackingFunction( monteCarlo, mc_particle, particle_index, processingVault, processedVault, tallyArray);
   
    //monteCarlo->_particleVaultContainer->getExtraVault(tallyArray[0])->setsize(*particleIndex);

    //Make sure this particle is marked as completed
    processingVault->invalidateParticle( particle_index );
}

void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault, int * tallyArray)
{
    //bool keepTrackingThisParticle = false;
    bool keepTrackingThisParticle = true;
    unsigned int tally_index =      (particle_index) % monteCarlo->_tallies->GetNumBalanceReplications();
    unsigned int flux_tally_index = (particle_index) % monteCarlo->_tallies->GetNumFluxReplications();
    unsigned int cell_tally_index = (particle_index) % monteCarlo->_tallies->GetNumCellTallyReplications();

    int i1=0;
    //should never reach MaxIters, but if it does we will Fail the physical tests at the end (particles will be lost)
    int MaxIters=1000;

    do
    {
        // Determine the outcome of a particle at the end of this segment such as:
        //
        //   (0) Undergo a collision within the current cell,
        //   (1) Cross a facet of the current cell,
        //   (2) Reach the end of the time step and enter census,
        //
        MC_Segment_Outcome_type::Enum segment_outcome = MC_Segment_Outcome_type::Max_Number;
        i1+=1;
        if(keepTrackingThisParticle)
        {

#ifdef EXPONENTIAL_TALLY
        monteCarlo->_tallies->TallyCellValue( exp(rngSample(&mc_particle.random_number_seed)) , mc_particle.domain, cell_tally_index, mc_particle.cell);
#endif   
        segment_outcome = MC_Segment_Outcome(monteCarlo, mc_particle, flux_tally_index);

        ATOMIC_UPDATE(tallyArray[tally_index*8+0]);

        mc_particle.num_segments += 1.;  /* Track the number of segments this particle has
                                            undergone this cycle on all processes. */
        segment_outcome = keepTrackingThisParticle ? segment_outcome : MC_Segment_Outcome_type::Max_Number;
        }
        switch (segment_outcome) {


        case MC_Segment_Outcome_type::Collision:
            {
            // The particle undergoes a collision event producing:
            //   (0) Other-than-one same-species secondary particle, or
            //   (1) Exactly one same-species secondary particle.
            if (CollisionEvent(monteCarlo, mc_particle, tally_index,particle_index, tallyArray) == MC_Collision_Event_Return::Continue_Tracking)
            {
                keepTrackingThisParticle = true;
            }
            else
            {
                keepTrackingThisParticle = false;
            }
            }
            break;
    
        case MC_Segment_Outcome_type::Facet_Crossing:
            { 
                // The particle has reached a cell facet.
                MC_Tally_Event::Enum facet_crossing_type = MC_Facet_Crossing_Event(mc_particle, monteCarlo, particle_index, processingVault);

                if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Transit_Exit)
                {
                    keepTrackingThisParticle = true;  // Transit Event
                }
                else if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Escape)
                {
                    ATOMIC_UPDATE( tallyArray[tally_index*8+1]);
                    mc_particle.last_event = MC_Tally_Event::Facet_Crossing_Escape;
                    mc_particle.species = -1;
                    keepTrackingThisParticle = false;
                }
                else if (facet_crossing_type == MC_Tally_Event::Facet_Crossing_Reflection)
                {
                    MCT_Reflect_Particle(monteCarlo, mc_particle);
                    keepTrackingThisParticle = true;
                }
                else
                {
                    // Enters an adjacent cell in an off-processor domain.
                    //mc_particle.species = -1;
                    keepTrackingThisParticle = false;
                }
            }
            break;
    
        case MC_Segment_Outcome_type::Census:
            {
                // The particle has reached the end of the time step.
                processedVault->pushParticle(mc_particle);
                ATOMIC_UPDATE( tallyArray[tally_index*8+2]);
                keepTrackingThisParticle = false;
            }
                break;

        case MC_Segment_Outcome_type::Max_Number:
           {

           keepTrackingThisParticle = false;
           }
           break;
            
        default:
           qs_assert(false);
           keepTrackingThisParticle = false;
           break;  // should this be an error
        }
    } while (keepTrackingThisParticle && i1<MaxIters);

}

