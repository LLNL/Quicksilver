#ifndef MACROSCOPIC_CROSS_SECTION_HH
#define MACROSCOPIC_CROSS_SECTION_HH

#include "DeclareMacro.hh"

class MonteCarlo;

HOST_DEVICE
double macroscopicCrossSection(MonteCarlo* monteCarlo, int reactionIndex, int domainIndex, int cellIndex,
                               int isoIndex, int energyGroup);
HOST_DEVICE_END

HOST_DEVICE
double weightedMacroscopicCrossSection(MonteCarlo* monteCarlo, int taskIndex, int domainIndex,
                                       int cellIndex, int energyGroup);
HOST_DEVICE_END

#endif
