#ifndef DIRECTION_COSINE_INCLUDE
#define DIRECTION_COSINE_INCLUDE

#include <cmath>
#include "portability.hh"
#include "DeclareMacro.hh"

HOST_DEVICE_CLASS
class DirectionCosine
{
public:
   double alpha;
   double beta;
   double gamma;

   HOST_DEVICE_CUDA
   DirectionCosine();

   HOST_DEVICE_CUDA
   DirectionCosine(double alpha, double beta, double gamma);

   HOST_DEVICE_CUDA
   DirectionCosine &operator=(const DirectionCosine &dc) 
   {
       alpha = dc.alpha;
       beta  = dc.beta;
       gamma = dc.gamma;
       return *this;
    }

   void Sample_Isotropic(uint64_t *seed);

   // rotate a direction cosine given the sine/cosine of theta and phi
   HOST_DEVICE_CUDA
   inline void Rotate3DVector( double sine_Theta,
                               double cosine_Theta,
                               double sine_Phi,
                               double cosine_Phi );

};
HOST_DEVICE_END

HOST_DEVICE
inline DirectionCosine::DirectionCosine()
   : alpha(0.0), beta(0.0), gamma(0.0)
{
}
HOST_DEVICE_END

HOST_DEVICE
inline DirectionCosine::DirectionCosine(double a_alpha, double a_beta, double a_gamma)
   : alpha(a_alpha),
     beta(a_beta),
     gamma(a_gamma)
{
}
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//
//  This function rotates a three-dimensional vector that is defined by the angles
//  Theta and Phi in a local coordinate frame about a polar angle and azimuthal angle described by
//  the direction cosine.  Hence, caller passes in sin_Theta and cos_Theta referenced from
//  the local z-axis and sin_Phi and cos_Phi referenced from the local x-axis to describe the
//  vector V to be rotated.  The direction cosine describes global theta and phi angles that the
//  vector V is to be rotated about.
//  Example:  Caller wishes to rotate a vector V described by direction cosines of (-1,0,0),
//            which corresponds to local angles Theta and Phi of
//            Theta =  Pi/2 so sin_Theta =  1, cos_Theta =  0,
//            Phi   = -Pi/2 so sin_Phi   = -1, cos_Phi   = -1. (wrong)
//            Phi   =  Pi   so sin_Phi   =  0, cos_Phi   = -1.
//
//            We wish to rotate this vector V by Pi/2 in global theta and Pi/2 in phi.
//            The resulting direction cosine is (alpha,beta,gamma) = (0,1,0).  The return result
//            is (1,0,0).  The rotation is about the forward y-axis followed by the (for) z-axis.
//
//            Note: this rotation operator fails when trying to rotate a vector V in the x-y plane
//            to another location in the x-y plane.  It essentially randomizes the rotation in x-y
//            instead of returning the exact rotation requested.
//
// ---------------------------- theory on the function
//
//        lowercase theta and phi are the spherical coordinates for direction_cosine.
//        in general spherical coordinates, phi is in the x,y plane, and theta is the angle with +z.
//        x = r cos(phi) sin(theta)
//        y = r sin(phi) sin(theta)
//        z = r cos(theta)
//        or
//        r     = sqrt(x*x + y*y + z*z)
//        phi   = atan(y/x)
//        theta = acos(z/r)
//
//        (x,y,z) = direction_cosine = (alpha, beta, gamma), with r = alpha^2 + beta^2 + gamma^2 = 1 so
//        alpha = cos(phi) sin(theta)
//        beta  = sin(phi) sin(theta)
//        gamma = cos(theta)
//        or
//        phi   = atan(beta/alpha)
//        theta = acos(gamma)
//        so
//        sin(phi) =  beta/sqrt(alpha^2 + beta^2) =  beta/sqrt(1 - gamma^2) =  beta/sin(theta)
//        cos(phi) = alpha/sqrt(alpha^2 + beta^2) = alpha/sqrt(1 - gamma^2) = alpha/sin(theta)
//
//        Rotation matrix, lower case, times Upper Case Unit Vector.
//        The rotation matrix maps the x-axis (1,0,0) to the 1st column,
//                                     y-axis (0,1,0) to the 2nd column
//                                     z-axis (0,0,1) to the 3rd column.
//        (it maps the z-axis (0,0,1) to standard polar coordinates (cos(phi)*sin(theta), sin(phi)*sin(theta), cos(theta))
//
//        [alpha] = [cos(phi)*cos(theta)   -sin(phi)   cos(phi)*sin(theta)]  [sin(Theta)*cos(Phi)]
//        [beta ] = [sin(phi)*cos(theta)    cos(phi)   sin(phi)*sin(theta)]  [sin(Theta)*sin(Phi)]
//        [gamma] = [        -sin(theta)      0                 cos(theta)]  [cos(Theta)         ]
//
//        double Alpha = sin_Theta*cos_Phi;
//        double Beta  = sin_Theta*sin_Phi;
//        double Gamma = cos_Theta;
//
//        direction_cosine.alpha =  cos_theta*cos_phi*Alpha - sin_phi*Beta + sin_theta*cos_phi*Gamma;
//        direction_cosine.beta =   cos_theta*sin_phi*Alpha + cos_phi*Beta + sin_theta*sin_phi*Gamma;
//        direction_cosine.gamma = -sin_theta        *Alpha +                cos_theta        *Gamma;
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE
inline void DirectionCosine::Rotate3DVector(double sin_Theta, double cos_Theta, double sin_Phi, double cos_Phi)
{
    // Calculate additional variables in the rotation matrix.
    double cos_theta = this->gamma;
    double sin_theta = sqrt((1.0 - (cos_theta*cos_theta)));

    double cos_phi;
    double sin_phi;
    if (sin_theta < 1e-6) // Order of sqrt(PhysicalConstants::tiny_double)
    {
        cos_phi = 1.0; // assume phi  = 0.0;
        sin_phi = 0.0;
    }
    else
    {
        cos_phi = this->alpha/sin_theta;
        sin_phi = this->beta/sin_theta;
    }

    // Calculate the rotated direction cosine
    this->alpha =  cos_theta*cos_phi*(sin_Theta*cos_Phi) - sin_phi*(sin_Theta*sin_Phi) + sin_theta*cos_phi*cos_Theta;
    this->beta  =  cos_theta*sin_phi*(sin_Theta*cos_Phi) + cos_phi*(sin_Theta*sin_Phi) + sin_theta*sin_phi*cos_Theta;
    this->gamma = -sin_theta        *(sin_Theta*cos_Phi) +                               cos_theta        *cos_Theta;
}
HOST_DEVICE_END

#endif
