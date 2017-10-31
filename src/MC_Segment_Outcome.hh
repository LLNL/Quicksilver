#ifndef MC_SEGMENT_OUTCOME_INCLUDE
#define MC_SEGMENT_OUTCOME_INCLUDE

class MC_Particle;
class MC_Vector;
class MonteCarlo;


struct MC_Segment_Outcome_type
{
    public:
    enum Enum
    {
        Initialize                    = -1,
        Collision                     = 0,
        Facet_Crossing                = 1,
        Census                        = 2,
        Max_Number                    = 3
    };
};


struct MC_Collision_Event_Return
{
    public:
    enum Enum
    {
        Stop_Tracking     = 0,
        Continue_Tracking = 1,
        Continue_Collision = 2
    };
};

#include "DeclareMacro.hh"
HOST_DEVICE
MC_Segment_Outcome_type::Enum MC_Segment_Outcome(MonteCarlo* monteCarlo, MC_Particle &mc_particle, unsigned int &flux_tally_index);
HOST_DEVICE_END

#endif
