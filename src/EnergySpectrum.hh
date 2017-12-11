#ifndef ENERGYSPECTRUM_HH
#define ENERGYSPECTRUM_HH
#include <string>

class MonteCarlo;

void updateSpectrum( MonteCarlo* monteCarlo, uint64_t *hist );
void printSpectrum( uint64_t *hist, MonteCarlo* monteCarlo);
#endif

