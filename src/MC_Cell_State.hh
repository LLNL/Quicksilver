#ifndef MC_CELL_STATE_INCLUDE
#define MC_CELL_STATE_INCLUDE

#include <cstdio>
#include "QS_Vector.hh"
#include "macros.hh"


// this stores all the material information on a cell
class MC_Cell_State
{
 public:

   int _material; // gid of material

   // pre-computed cross-sections for material
   double* _total;  // [energy groups]

   double  _volume;                 // cell volume
   double  _cellNumberDensity;         // number density of ions in cel

   uint64_t _id;
   unsigned _sourceTally;
   
   MC_Cell_State();

 private:
};

inline MC_Cell_State::MC_Cell_State()
  : _material(0),
    _total(),
    _volume(0.0),
    _cellNumberDensity(0.0),
    _sourceTally(0)
{
}

#endif
