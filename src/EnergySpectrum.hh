#ifndef ENERGYSPECTRUM_HH
#define ENERGYSPECTRUM_HH
#include <string>

class MonteCarlo;

class EnergySpectrum
{
    public:
        EnergySpectrum() : fileName("") {}
        void Allocate(std::string name, uint64_t size);
        void UpdateSpectrum(MonteCarlo* monteCarlo);
        void PrintSpectrum(MonteCarlo* monteCarlo);

    private:
        std::string fileName;
        uint64_t *CensusEnergySpectrum;
};

#endif

