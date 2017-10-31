#include "MC_Segment_Outcome.hh"
#include "MC_Nearest_Facet.hh"
#include "MC_Location.hh"
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Particle.hh"
#include "MC_RNG_State.hh"
#include "MC_Cell_State.hh"
#include "Tallies.hh"
#include "utils.hh"
#include "macros.hh"
#include "MacroscopicCrossSection.hh"
#include "MCT.hh"
#include "PhysicalConstants.hh"
#include "DeclareMacro.hh"


HOST_DEVICE
static inline unsigned int MC_Find_Min(const double *array,
                                       int     num_elements);
HOST_DEVICE_END

//--------------------------------------------------------------------------------------------------
//  Routine MC_Segment_Outcome determines whether the next segment of the particle's trajectory will result in:
//    (i) collision within the current cell,
//   (ii) exiting from the current cell, or
//  (iii) census at the end of the time step.
//--------------------------------------------------------------------------------------------------

HOST_DEVICE 
MC_Segment_Outcome_type::Enum MC_Segment_Outcome(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int &flux_tally_index)
{
    // initialize distances to large number
    int number_of_events = 3;
    double distance[3];
    distance[0] = distance[1] = distance[2] = 1e80;

    // Calculate the particle speed
    double particle_speed = mc_particle.Get_Velocity()->Length();

    // Force collision if a census event narrowly preempts a collision
    int force_collision = 0 ;
    if ( mc_particle.num_mean_free_paths < 0.0 )
    {
        force_collision = 1 ;

        if ( mc_particle.num_mean_free_paths > -900.0 )
        {
#if 1
            printf(" MC_Segment_Outcome: mc_particle.num_mean_free_paths > -900.0 \n");
 #else
            std::string output_string;
            MC_Warning( "Forced Collision: num_mean_free_paths < 0 \n"
                             "Particle record:\n%s", output_string.c_str());
#endif
        }

        mc_particle.num_mean_free_paths = PhysicalConstants::_smallDouble;
    }

    // Randomly determine the distance to the next collision
    // based upon the composition of the current cell.
    double macroscopic_total_cross_section = weightedMacroscopicCrossSection(monteCarlo, 0,
                             mc_particle.domain, mc_particle.cell, mc_particle.energy_group);

    // Cache the cross section
    mc_particle.totalCrossSection = macroscopic_total_cross_section;
    if (macroscopic_total_cross_section == 0.0)
    {
        mc_particle.mean_free_path = PhysicalConstants::_hugeDouble;
    }
    else
    {
        mc_particle.mean_free_path = 1.0 / macroscopic_total_cross_section;
    }

    if ( mc_particle.num_mean_free_paths == 0.0)
    {
        // Sample the number of mean-free-paths remaining before
        // the next collision from an exponential distribution.
        double random_number = rngSample(&mc_particle.random_number_seed);

        mc_particle.num_mean_free_paths = -1.0*log(random_number);
    }

    // Calculate the distances to collision, nearest facet, and census.

    // Forced collisions do not need to move far.
    if (force_collision)
    {
        distance[MC_Segment_Outcome_type::Collision] = PhysicalConstants::_smallDouble;
    }
    else
    {
        distance[MC_Segment_Outcome_type::Collision] = mc_particle.num_mean_free_paths*mc_particle.mean_free_path;
    }

    // process census
    distance[MC_Segment_Outcome_type::Census] = particle_speed*mc_particle.time_to_census;


    //  DEBUG  Turn off threshold for now
    double distance_threshold = 10.0 * PhysicalConstants::_hugeDouble;
    // Get the current winning distance.
    double current_best_distance = PhysicalConstants::_hugeDouble;

    DirectionCosine *direction_cosine = mc_particle.Get_Direction_Cosine();

    bool new_segment =  (mc_particle.num_segments == 0 ||
                         mc_particle.last_event == MC_Tally_Event::Collision);

    MC_Location location(mc_particle.Get_Location());

    // Calculate the minimum distance to each facet of the cell.
    MC_Nearest_Facet nearest_facet;
        nearest_facet = MCT_Nearest_Facet(&mc_particle, location, mc_particle.coordinate,
                                  direction_cosine, distance_threshold, current_best_distance, new_segment, monteCarlo);

    mc_particle.normal_dot = nearest_facet.dot_product;

    distance[MC_Segment_Outcome_type::Facet_Crossing] = nearest_facet.distance_to_facet;


    // Get out of here if the tracker failed to bound this particle's volume.
    if (mc_particle.last_event == MC_Tally_Event::Facet_Crossing_Tracking_Error)
    {
        return MC_Segment_Outcome_type::Facet_Crossing;
    }

    // Calculate the minimum distance to the selected events.

    // Force a collision (if required).
    if ( force_collision == 1 )
    {
        distance[MC_Segment_Outcome_type::Facet_Crossing] = PhysicalConstants::_hugeDouble;
        distance[MC_Segment_Outcome_type::Census]         = PhysicalConstants::_hugeDouble;
        distance[MC_Segment_Outcome_type::Collision]      = PhysicalConstants::_tinyDouble ;
    }

    // we choose our segment outcome here
    MC_Segment_Outcome_type::Enum segment_outcome =
        (MC_Segment_Outcome_type::Enum) MC_Find_Min(distance, number_of_events);

    if (distance[segment_outcome] < 0)
    {
        MC_Fatal_Jump( "Negative distances to events are NOT permitted!\n"
                       "identifier              = %" PRIu64 "\n"
                       "(Collision              = %g,\n"
                       " Facet Crossing         = %g,\n"
                       " Census                 = %g,\n",
                       mc_particle.identifier,
                       distance[MC_Segment_Outcome_type::Collision],
                       distance[MC_Segment_Outcome_type::Facet_Crossing],
                       distance[MC_Segment_Outcome_type::Census]);
    }
    mc_particle.segment_path_length = distance[segment_outcome];

    mc_particle.num_mean_free_paths -= mc_particle.segment_path_length / mc_particle.mean_free_path;

    // Before using segment_outcome as an index, verify it is valid
    if (segment_outcome < 0 || segment_outcome >= MC_Segment_Outcome_type::Max_Number)
    {
        MC_Fatal_Jump( "segment_outcome '%d' is invalid\n", (int)segment_outcome );
    }

    MC_Tally_Event::Enum SegmentOutcome_to_LastEvent[MC_Segment_Outcome_type::Max_Number] =
    {
        MC_Tally_Event::Collision,
        MC_Tally_Event::Facet_Crossing_Transit_Exit,
        MC_Tally_Event::Census,
    };

    mc_particle.last_event = SegmentOutcome_to_LastEvent[segment_outcome];

    // Set the segment path length to be the minimum of
    //   (i)   the distance to collision in the cell, or
    //   (ii)  the minimum distance to a facet of the cell, or
    //   (iii) the distance to census at the end of the time step
    if (segment_outcome == MC_Segment_Outcome_type::Collision)
    {
        mc_particle.num_mean_free_paths = 0.0;
    }
    else if (segment_outcome == MC_Segment_Outcome_type::Facet_Crossing)
    {
        mc_particle.facet = nearest_facet.facet;
    }
    else if (segment_outcome == MC_Segment_Outcome_type::Census)
    {
        mc_particle.time_to_census = MC_MIN(mc_particle.time_to_census, 0.0);
    }

    // If collision was forced, set mc_particle.num_mean_free_paths = 0
    // so that a new value is randomly selected on next pass.
    if (force_collision == 1) { mc_particle.num_mean_free_paths = 0.0; }

    // Do not perform any tallies if the segment path length is zero.
    //   This only introduces roundoff errors.
    if (mc_particle.segment_path_length == 0.0)
    {
        return segment_outcome;
    }

    // Move particle to end of segment, accounting for some physics processes along the segment.

    // Project the particle trajectory along the segment path length.
    mc_particle.Move_Particle(mc_particle.direction_cosine, mc_particle.segment_path_length);

    double segment_path_time = (mc_particle.segment_path_length/particle_speed);

    // Decrement the time to census and increment age.
    mc_particle.time_to_census -= segment_path_time;
    mc_particle.age += segment_path_time;

    // Ensure mc_particle.time_to_census is non-negative.
    if (mc_particle.time_to_census < 0.0)
    {
        mc_particle.time_to_census = 0.0;
    }

    // Accumulate the particle's contribution to the scalar flux.
    monteCarlo->_tallies->TallyScalarFlux(mc_particle.segment_path_length * mc_particle.weight, mc_particle.domain,
                                    flux_tally_index, mc_particle.cell, mc_particle.energy_group);

    return segment_outcome;
}
HOST_DEVICE_END




HOST_DEVICE
static inline unsigned int MC_Find_Min(const double *array,
                                       int     num_elements)
{
    double min = array[0];
    int    min_index = 0;

    for (int element_index = 1; element_index < num_elements; ++element_index)
    {
        if ( array[element_index] < min )
        {
            min = array[element_index];
            min_index = element_index;
        }
    }

    return min_index;
}
HOST_DEVICE_END
