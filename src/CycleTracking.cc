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

HOST_DEVICE
void CycleTrackingGuts( MonteCarlo *monteCarlo, int particle_index, ParticleVault *processingVault, ParticleVault *processedVault )
{
    MC_Particle mc_particle;

    // Copy a single particle from the particle vault into mc_particle
    MC_Load_Particle(monteCarlo, mc_particle, processingVault, particle_index);

    // set the particle.task to the index of the processed vault the particle will census into.
    mc_particle.task = 0;//processed_vault;

    // loop over this particle until we cannot do anything more with it on this processor
    CycleTrackingFunction( monteCarlo, mc_particle, particle_index, processingVault, processedVault );

    //Make sure this particle is marked as completed
    processingVault->invalidateParticle( particle_index );
}
HOST_DEVICE_END

HOST_DEVICE
void CycleTrackingFunction( MonteCarlo *monteCarlo, MC_Particle &mc_particle, int particle_index, ParticleVault* processingVault, ParticleVault* processedVault)
{
    bool keepTrackingThisParticle = false;
    unsigned int tally_index =      (particle_index) % monteCarlo->_tallies->GetNumBalanceReplications();
    unsigned int flux_tally_index = (particle_index) % monteCarlo->_tallies->GetNumFluxReplications();
    unsigned int cell_tally_index = (particle_index) % monteCarlo->_tallies->GetNumCellTallyReplications();
    do
    {
        // Determine the outcome of a particle at the end of this segment such as:
        //
        //   (0) Undergo a collision within the current cell,
        //   (1) Cross a facet of the current cell,
        //   (2) Reach the end of the time step and enter census,
        //
#ifdef EXPONENTIAL_TALLY
        monteCarlo->_tallies->TallyCellValue( exp(rngSample(&mc_particle.random_number_seed)) , mc_particle.domain, cell_tally_index, mc_particle.cell);
#endif   
        MC_Segment_Outcome_type::Enum segment_outcome = MC_Segment_Outcome(monteCarlo, mc_particle, flux_tally_index);

        ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._numSegments);

        mc_particle.num_segments += 1.;  /* Track the number of segments this particle has
                                            undergone this cycle on all processes. */
        switch (segment_outcome) {
        case MC_Segment_Outcome_type::Collision:
            {
            // The particle undergoes a collision event producing:
            //   (0) Other-than-one same-species secondary particle, or
            //   (1) Exactly one same-species secondary particle.
            if (CollisionEvent(monteCarlo, mc_particle, tally_index ) == MC_Collision_Event_Return::Continue_Tracking)
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
                    ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._escape);
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
                ATOMIC_UPDATE( monteCarlo->_tallies->_balanceTask[tally_index]._census);
                keepTrackingThisParticle = false;
                break;
            }
            
        default:
           qs_assert(false);
           break;  // should this be an error
        }
    
    } while ( keepTrackingThisParticle );
}
HOST_DEVICE_END

