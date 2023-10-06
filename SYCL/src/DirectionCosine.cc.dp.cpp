#include "DirectionCosine.hh"
#include "MC_RNG_State.hh"
#include "PhysicalConstants.hh"

void DirectionCosine::Sample_Isotropic(uint64_t *seed)
{
    this->gamma  = 1.0 - 2.0*rngSample(seed);
    double sine_gamma  = sqrt((1.0 - (gamma*gamma)));
    double phi         = PhysicalConstants::_pi*(2.0*rngSample(seed) - 1.0);

    this->alpha  = sine_gamma * cos(phi);
    this->beta   = sine_gamma * sin(phi);
}
