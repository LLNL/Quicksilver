#ifndef MC_PARTICLE_INCLUDE
#define MC_PARTICLE_INCLUDE

#include "DirectionCosine.hh"
#include "Tallies.hh"

#include "MC_Vector.hh"
#include "MC_Facet_Adjacency.hh"
#include "MC_Location.hh"

#include "DeclareMacro.hh"

class MC_Base_Particle;


class MC_Particle
{
 public:

    // the current position of the particle
    MC_Vector coordinate;

    // the velocity of the particle
    MC_Vector velocity;

    // the direction of the particle
    DirectionCosine direction_cosine;

    // the kinetic energy of the particle
    double kinetic_energy;

    // the weight of the particle
    double weight;

    // the time remaining for this particle to hit census
    double time_to_census;

    // cacheing the current total cross section
    double totalCrossSection;

    // the age of this particle
    double age;

    // the number of mean free paths to a collision
    double num_mean_free_paths;

    // distance to a collision
    double mean_free_path;

    // the distance this particle travels in a segment.
    double segment_path_length;

    // the random number seed for the random number generator for this particle
    uint64_t random_number_seed;

    // unique identifier used to identify and track individual particles in the simulation
    uint64_t identifier;

   // the last event this particle underwent
    MC_Tally_Event::Enum last_event;

    int num_collisions;

    double num_segments;

    // task working on, used to index into
    int task;

    // species of the particle
    int species;

    // the breed of the particle how it was produced
    int breed;

    // current energy group of the particle
    int energy_group;

    // its current domain in the spatial decomposition
    int domain;

    // the current cell in its current domain
    int cell;

    int facet;

    // When crossing a facet, keep the surface normal dot product
    double normal_dot;

public:
   HOST_DEVICE_CUDA
   MC_Particle();

   HOST_DEVICE_CUDA
   MC_Particle( const MC_Base_Particle &from_particle );

   HOST_DEVICE_CUDA
   void Copy_From_Base( const MC_Base_Particle &from_particle);

   HOST_DEVICE_CUDA
   MC_Location Get_Location() const;

   // format a string with the contents of the particle
   void Copy_Particle_To_String(std::string &output_string) const;

   // move a particle a distance in the direction_cosine direction
   HOST_DEVICE_CUDA
   void Move_Particle(const DirectionCosine & direction_cosine, const double distance);

   HOST_DEVICE_CUDA
   void PrintParticle();

   HOST_DEVICE_CUDA
   DirectionCosine *Get_Direction_Cosine()
   {
      return &this->direction_cosine;
   }

   HOST_DEVICE_CUDA
   MC_Vector *Get_Velocity()
   {
      return &this->velocity;
   }
};

//----------------------------------------------------------------------------------------------------------------------
//  Return a MC_Location given domain, cell, facet.
//----------------------------------------------------------------------------------------------------------------------
inline MC_Location MC_Particle::Get_Location() const
{
    return MC_Location(domain, cell, facet);
}

//----------------------------------------------------------------------------------------------------------------------
//  Move the particle a straight-line distance along a specified cosine.
//----------------------------------------------------------------------------------------------------------------------
inline void MC_Particle::Move_Particle( const DirectionCosine &my_direction_cosine,
                                      const double distance)
{
    coordinate.x += (my_direction_cosine.alpha * distance);
    coordinate.y += (my_direction_cosine.beta  * distance);
    coordinate.z += (my_direction_cosine.gamma * distance);
}

//----------------------------------------------------------------------------------------------------------------------
//  Print all of the particles components
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE_CUDA
inline void MC_Particle::PrintParticle()
{
    printf( "coordiante:          %g\t%g\t%g\n", coordinate.x, coordinate.y, coordinate.z );
    printf( "velocity:            %g\t%g\t%g\n", velocity.x, velocity.y, velocity.z );
    printf( "direction_cosine:     %g\t%g\t%g\n", direction_cosine.alpha, direction_cosine.beta, direction_cosine.gamma );
    printf( "kinetic_energy:      %g\n", kinetic_energy );
    printf( "Weight:              %g\n", weight);
    printf( "time_to_census:      %g\n", time_to_census);
    printf( "totalCrossSection:   %g\n", totalCrossSection);
    printf( "age:                 %g\n", age);
    printf( "num_mean_free_paths: %g\n", num_mean_free_paths);
    printf( "mean_free_path:      %g\n", mean_free_path);
    printf( "segment_path_length: %g\n", segment_path_length);
    printf( "random_number_seed:  %zu\n", random_number_seed);
    printf( "identifier:          %zu\n", identifier);
    printf( "last_event:          %d\n", last_event);
    printf( "num_collision:       %d\n", num_collisions);
    printf( "num_segments:        %g\n", num_segments);
    printf( "task:                %d\n", task);
    printf( "species:             %d\n", species);
    printf( "breed:               %d\n", breed);
    printf( "energy_group:        %d\n", energy_group);
    printf( "domain:              %d\n", domain);
    printf( "cell:                %d\n", cell);
    printf( "facet:               %d\n", facet);
    printf( "normal_dot:          %g\n", normal_dot);
    printf("\n");
}


int                         MC_Copy_Particle_Get_Num_Fields();


#endif //  MC_PARTICLE_INCLUDE
