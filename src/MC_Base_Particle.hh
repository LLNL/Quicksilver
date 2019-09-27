#ifndef MC_BASE_PARTICLE
#define MC_BASE_PARTICLE

#include "portability.hh"

#include "MC_Vector.hh"
#include "MC_RNG_State.hh"
#include "MC_Location.hh"
#include "DirectionCosine.hh"
#include "Tallies.hh"

struct MC_Data_Member_Operation
{
    public:
    enum Enum
    {
        Count   = 0,
        Pack    = 1,
        Unpack  = 2,
        Reset   = 3
    };
};

HOST_DEVICE_CLASS

class MC_Base_Particle
{
  public:

    static void Cycle_Setup();
    static void Update_Counts();

    HOST_DEVICE_CUDA
    MC_Base_Particle();
    HOST_DEVICE_CUDA
    MC_Base_Particle(  const MC_Base_Particle &particle);

    HOST_DEVICE_CUDA
    int particle_id_number() const;
    HOST_DEVICE_CUDA
    int invalidate();



   // move a particle a distance in the direction_cosine direction
   HOST_DEVICE_CUDA
   void Move_Particle(const DirectionCosine & direction_cosine, const double distance);




  

    // serialize the vault
    void Serialize(int *int_data, double *float_data, char *char_data,
                  int &int_index, int &float_index, int &char_index,
                  MC_Data_Member_Operation::Enum mode);

    // return a location
    HOST_DEVICE_CUDA
    MC_Location Get_Location() const;

    // copy contents to a string
    void Copy_Particle_Base_To_String(std::string &output_string) const;

    // aliases for the type of particle that we have
    HOST_DEVICE_CUDA
    inline int type() const     { return species; }
    HOST_DEVICE_CUDA
    inline int index() const    { return species; }
    HOST_DEVICE_CUDA
    inline int is_valid() const { return (0 <= species); }

    HOST_DEVICE_CUDA
    inline double Get_Energy() const                     { return kinetic_energy; }
    HOST_DEVICE_CUDA
    inline MC_Vector *Get_Velocity() { return &velocity; }


    MC_Vector                          coordinate;
    MC_Vector                          velocity;

  DirectionCosine                    direction_cosine; //new
  
    double                             kinetic_energy;
    double                             weight;
    double                             time_to_census;
  double                              totalCrossSection; //new
    double                             age;
    double                             num_mean_free_paths;
  double                              mean_free_path;  //new
  double                              segment_path_length; //new
  double                             num_segments;
  double                              normal_dot; //new
  
    uint64_t                           random_number_seed;
    uint64_t                           identifier;

    MC_Tally_Event::Enum               last_event;
    int                                num_collisions;
    int                                breed;
  int                                energy_group; //new
  int                                task; //new
  int                                species;
    int                                domain;
    int                                cell;
  int                                facet;

    static int                         num_base_ints;   // Number of ints for communication
    static int                         num_base_floats; // Number of floats for communication
    static int                         num_base_chars;  // Number of chars for communication

  private:
    
};

HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//  Return a MC_Location given domain, cell, facet.
//----------------------------------------------------------------------------------------------------------------------

HOST_DEVICE
inline MC_Location MC_Base_Particle::Get_Location() const
{
  return MC_Location(domain, cell, facet); // was zero for facet

}
HOST_DEVICE_END

HOST_DEVICE
//----------------------------------------------------------------------------------------------------------------------
//  Move the particle a straight-line distance along a specified cosine.
//----------------------------------------------------------------------------------------------------------------------
inline void MC_Base_Particle::Move_Particle( const DirectionCosine &my_direction_cosine,
                                      const double distance)
{
    coordinate.x += (my_direction_cosine.alpha * distance);
    coordinate.y += (my_direction_cosine.beta  * distance);
    coordinate.z += (my_direction_cosine.gamma * distance);
}
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
// Invalidate a particle.
//
// This public method "invalidates" this particle by setting its particle type to UNKNOWN. This
// method will fail if this particle is already invalid.
//
// return: A value of 1 (true) is returned on success, 0 (false) on failure.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE
inline int MC_Base_Particle::invalidate()
{
   if (is_valid())
   {
      species = -1;
      return 1;
   }
   else return 0;
}
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//  Base information for a particle.
//----------------------------------------------------------------------------------------------------------------------

//----------------------------------------------------------------------------------------------------------------------
// Default constructor.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE
inline MC_Base_Particle::MC_Base_Particle( ) :
        coordinate(),
        velocity(),
        kinetic_energy(0.0),
        weight(0.0),
        time_to_census(0.0),
        age(0.0),
        num_mean_free_paths(0.0),
        num_segments(0.0),
        random_number_seed((uint64_t)0),
        identifier((uint64_t)0),
        last_event(MC_Tally_Event::Census),
        num_collisions(0),
        breed(0),
        // species == -1 is a special signifier for invalidated particle
        species(-1),
        domain(0),
        cell(0)
{
}
HOST_DEVICE_END


//----------------------------------------------------------------------------------------------------------------------
// Constructor from a base particle type.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE
inline MC_Base_Particle::MC_Base_Particle(const MC_Base_Particle &particle)
{
    coordinate          = particle.coordinate;
    velocity            = particle.velocity;
    kinetic_energy      = particle.kinetic_energy;
    weight              = particle.weight;
    time_to_census      = particle.time_to_census;
    age                 = particle.age;
    num_mean_free_paths = particle.num_mean_free_paths;
    num_segments        = particle.num_segments;
    random_number_seed  = particle.random_number_seed;
    identifier          = particle.identifier;
    last_event          = particle.last_event;
    num_collisions      = particle.num_collisions;
    breed               = particle.breed;
    species             = particle.species;
    domain              = particle.domain;
    cell                = particle.cell;
}
HOST_DEVICE_END


//----------------------------------------------------------------------------------------------------------------------
// MC_Particle Constructor.
//----------------------------------------------------------------------------------------------------------------------
// HOST_DEVICE
// inline MC_Particle::MC_Particle()
//    : coordinate(),
//      velocity(),
//      direction_cosine(),
//      kinetic_energy(0.0),
//      weight(0.0),
//      time_to_census(0.0),
//      totalCrossSection(0.0),
//      age(0.0),
//      num_mean_free_paths(0.0),
//      mean_free_path(0.0),
//      segment_path_length(0.0),
//      random_number_seed((uint64_t)0),
//      identifier( (uint64_t)0),
//      last_event(MC_Tally_Event::Census),
//      num_collisions (0),
//      num_segments(0.0),

//      task(0),
//      species(0),
//      breed(0),
//      energy_group(0),
//      domain(0),
//      cell(0),
//      facet(0),
//      normal_dot(0.0)
// {
// }
// HOST_DEVICE_END


//----------------------------------------------------------------------------------------------------------------------
// MC_Particle Constructor.
//----------------------------------------------------------------------------------------------------------------------
// HOST_DEVICE
// inline MC_Particle::MC_Particle( const MC_Base_Particle &from_particle )
//    : coordinate(from_particle.coordinate),
//      velocity(from_particle.velocity),
//      direction_cosine(), // define this from velocity in body of this function
//      kinetic_energy(from_particle.kinetic_energy),

//      weight(from_particle.weight),
//      time_to_census(from_particle.time_to_census),
//      age(from_particle.age),
//      num_mean_free_paths(from_particle.num_mean_free_paths),

//      mean_free_path(0.0),
//      segment_path_length(0.0),

//      random_number_seed(from_particle.random_number_seed),
//      identifier( from_particle.identifier ),
//      last_event(from_particle.last_event),

//      num_collisions (from_particle.num_collisions),
//      num_segments(from_particle.num_segments),


//      task(0),
//      species(from_particle.species),
//      breed(from_particle.breed),
//      energy_group(0),
//      domain(from_particle.domain),
//      cell(from_particle.cell),
//      normal_dot(0.0)
// {
//     double speed = from_particle.velocity.Length();

//     if ( speed > 0 )
//     {
//         double factor = 1.0/speed;
//         direction_cosine.alpha = factor * from_particle.velocity.x;
//         direction_cosine.beta  = factor * from_particle.velocity.y;
//         direction_cosine.gamma = factor * from_particle.velocity.z;
//     }
//     else
//     {
//         qs_assert(false);
//     }
// }
// HOST_DEVICE_END



//----------------------------------------------------------------------------------------------------------------------
//  Print the input particle to a string.
//----------------------------------------------------------------------------------------------------------------------
inline void MC_Base_Particle::Copy_Particle_Base_To_String(std::string &output_string) const
{
  qs_assert(false);
}


#endif

