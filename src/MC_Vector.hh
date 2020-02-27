/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef MC_VECTOR_INCLUDE
#define MC_VECTOR_INCLUDE

#include <cmath>
#include "DeclareMacro.hh"

HOST_DEVICE_CLASS
class MC_Vector
{
 public:
   double x;
   double y;
   double z;

   HOST_DEVICE_HIP
   MC_Vector() : x(0), y(0), z(0) {}
   HOST_DEVICE_HIP
   MC_Vector(double a, double b, double c) : x(a), y(b), z(c) {}

   HOST_DEVICE_HIP
   MC_Vector& operator=( const MC_Vector&tmp )
   {
      if ( this == &tmp ) { return *this; }

      x = tmp.x;
      y = tmp.y;
      z = tmp.z;

      return *this;
   }

   HOST_DEVICE_HIP
   bool operator==( const MC_Vector& tmp )
   {
      return tmp.x == x && tmp.y == y && tmp.z == z;
   }

   HOST_DEVICE_HIP
   MC_Vector& operator+=( const MC_Vector &tmp )
   {
      x += tmp.x;
      y += tmp.y;
      z += tmp.z;
      return *this;
   }

   HOST_DEVICE_HIP
   MC_Vector& operator-=( const MC_Vector &tmp )
   {
      x -= tmp.x;
      y -= tmp.y;
      z -= tmp.z;
      return *this;
   }

   HOST_DEVICE_HIP
   MC_Vector& operator*=(const double scalar)
   {
      x *= scalar;
      y *= scalar;
      z *= scalar;
      return *this;
   }

   HOST_DEVICE_HIP
   MC_Vector& operator/=(const double scalar)
   {
      x /= scalar;
      y /= scalar;
      z /= scalar;
      return *this;
   }

   HOST_DEVICE_HIP
   const MC_Vector operator+( const MC_Vector &tmp ) const
   {
      return MC_Vector(x + tmp.x, y + tmp.y, z + tmp.z);
   }

   HOST_DEVICE_HIP
   const MC_Vector operator-( const MC_Vector &tmp ) const
   {
      return MC_Vector(x - tmp.x, y - tmp.y, z - tmp.z);
   }

   HOST_DEVICE_HIP
   const MC_Vector operator*(const double scalar) const
   {
      return MC_Vector(scalar*x, scalar*y, scalar*z);
   }

   HOST_DEVICE_HIP
   inline double Length() const { return sqrt(x*x + y*y + z*z); }

   // Distance from this vector to another point.
   HOST_DEVICE_HIP
   inline double Distance(const MC_Vector& vv) const
   { return sqrt((x - vv.x)*(x - vv.x) + (y - vv.y)*(y - vv.y)+ (z - vv.z)*(z - vv.z)); }

   HOST_DEVICE_HIP
   inline double Dot(const MC_Vector &tmp) const
   {
      return this->x*tmp.x + this->y*tmp.y + this->z*tmp.z;
   }

   HOST_DEVICE_HIP
   inline MC_Vector Cross(const MC_Vector &v) const
   {
      return MC_Vector(y * v.z - z * v.y,
                       z * v.x - x * v.z,
                       x * v.y - y * v.x);
   }

};
HOST_DEVICE_END


#endif
