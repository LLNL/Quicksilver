#ifndef ENERGYSPECTRUM_HH
#define ENERGYSPECTRUM_HH
#include <string>
#include <vector>

class MonteCarlo;

class EnergySpectrum
{
    public:
        EnergySpectrum(std::string name, uint64_t size) : _fileName(name), _censusEnergySpectrum(size,0) {};
        void UpdateSpectrum(MonteCarlo* monteCarlo);
        void PrintSpectrum(MonteCarlo* monteCarlo);

    private:
        std::string _fileName;
        std::vector<uint64_t> _censusEnergySpectrum;
};

#endif

