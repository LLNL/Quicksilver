#include "PhysicalConstants.hh"

   // The values of all physical constants are taken from:
   // 2006 CODATA which is located on the web at
   // http://physics.nist.gov/cuu/Constants/codata.pdf

   // The units of physical quantities used by the code are:
   //    Mass         -  gram (g)
   //    Length       -  centimeter (cm)
   //    Time         -  second (s)
   //    Energy       -  million electron-volts (MeV) : of a particle
   //    Energy       -  erg (g cm^2/s^2): in some background calculation
   //    Temperature  -  thousand electron-volts (keV)

const double PhysicalConstants::_neutronRestMassEnergy = 9.395656981095e+2; /* MeV */
const double PhysicalConstants::_pi = 3.1415926535897932;
const double PhysicalConstants::_speedOfLight  = 2.99792458e+10;                // cm / s

// Constants used in math for computer science, roundoff, and other reasons
const double PhysicalConstants::_tinyDouble           = 1.0e-13;
const double PhysicalConstants::_smallDouble          = 1.0e-10;
const double PhysicalConstants::_hugeDouble           = 1.0e+75;
