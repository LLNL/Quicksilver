#ifndef MONTECARLO_HH
#define MONTECARLO_HH

#include "QS_Vector.hh"
#include "MC_Domain.hh"
#include "Parameters.hh"

class MC_RNG_State;
class NuclearData;
class MaterialDatabase;
class ParticleVaultContainer;
class Tallies;
class MC_Processor_Info;
class MC_Time_Info;
class MC_Particle_Buffer;
class MC_Fast_Timer_Container;

class MonteCarlo
{
public:

   MonteCarlo(const Parameters& params);
   ~MonteCarlo();

public:

   void clearCrossSectionCache();

   qs_vector<MC_Domain> domain;

    Parameters _params;
    NuclearData* _nuclearData;
    ParticleVaultContainer* _particleVaultContainer;
    MaterialDatabase* _materialDatabase;
    Tallies *_tallies;
    MC_Time_Info *time_info;
    MC_Fast_Timer_Container *fast_timer;
    MC_Processor_Info *processor_info;
    MC_Particle_Buffer *particle_buffer;

    double source_particle_weight;

private:
   // Disable copy constructor and assignment operator
   MonteCarlo(const MonteCarlo&);
   MonteCarlo& operator=(const MonteCarlo&);
};

#endif
