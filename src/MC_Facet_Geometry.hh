#ifndef MCT_FACET_GEOMETRY_3D_INCLUDE
#define MCT_FACET_GEOMETRY_3D_INCLUDE

#include "macros.hh"
#include <cstddef> // NULL

// A x + B y + C z + D = 0,  (A,B,C) is the plane normal and is normalized.
class MC_General_Plane
{
public:
    double A;
    double B;
    double C;
    double D;

   // Code to compute coefficients stolen from MCT_Facet_Adjacency_3D_G
   MC_General_Plane(){};
   MC_General_Plane(const MC_Vector& r0, const MC_Vector& r1, const MC_Vector& r2)
   {
      A = ((r1.y - r0.y)*(r2.z - r0.z)) - ((r1.z - r0.z)*(r2.y - r0.y));
      B = ((r1.z - r0.z)*(r2.x - r0.x)) - ((r1.x - r0.x)*(r2.z - r0.z));
      C = ((r1.x - r0.x)*(r2.y - r0.y)) - ((r1.y - r0.y)*(r2.x - r0.x));
      D = -1.0*(A*r0.x + B*r0.y + C*r0.z);

      double magnitude = sqrt(A * A + B * B + C * C);

      if ( magnitude == 0.0 )
      {
         A = 1.0;
         magnitude = 1.0;
      }
      // Normalize the planar-facet geometric cofficients.
      double inv_denominator = 1.0 / magnitude;

      A *= inv_denominator;
      B *= inv_denominator;
      C *= inv_denominator;
      D *= inv_denominator;
   }

};


class MC_Facet_Geometry_Cell
{
 public:
   MC_General_Plane* _facet;
   int _size;
};

#endif
