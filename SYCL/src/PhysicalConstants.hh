#ifndef PHYSICAL_CONSTANTS_HH
#define PHYSICAL_CONSTANTS_HH

#include "DeclareMacro.hh"
HOST_DEVICE_CLASS
namespace PhysicalConstants
{

const double _neutronRestMassEnergy = 9.395656981095e+2; /* MeV */
const double _pi = 3.1415926535897932;
const double _speedOfLight  = 2.99792458e+10;                // cm / s

// Constants used in math for computer science, roundoff, and other reasons
 const double _tinyDouble           = 1.0e-13;
 const double _smallDouble          = 1.0e-10;
 const double _hugeDouble           = 1.0e+75;
//
}
HOST_DEVICE_END


#endif
