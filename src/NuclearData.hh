#ifndef NUCLEAR_DATA_HH
#define NUCLEAR_DATA_HH

#include <cstdio>
#include <string>
#include "QS_Vector.hh"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "qs_assert.hh"
#include "DeclareMacro.hh"

class Polynomial
{
 public:
   Polynomial(double aa, double bb, double cc, double dd, double ee)
   :
   _aa(aa), _bb(bb), _cc(cc), _dd(dd), _ee(ee){}

   double operator()(double xx) const
   {
      return _ee + xx * (_dd + xx * (_cc + xx * (_bb + xx * (_aa))));
   }

 private:
   double _aa, _bb, _cc, _dd, _ee;
};

// Lowest level class at the reaction level
class NuclearDataReaction
{
 public:
   // The types of reactions
   enum Enum
   {
      Undefined = 0,
      Scatter,
      Absorption,
      Fission
   };
   
   NuclearDataReaction(){};

   NuclearDataReaction(Enum reactionType, double nuBar, const qs_vector<double>& energies,
                       const Polynomial& polynomial, double reationCrossSection);
   

   HOST_DEVICE_CUDA
   double getCrossSection(unsigned int group);
   HOST_DEVICE_CUDA
   void sampleCollision(double incidentEnergy, double material_mass, double* energyOut,
                        double* angleOut, int &nOut, uint64_t* seed, int max_production_size);
   
   
   qs_vector<double> _crossSection; //!< tabular data for microscopic cross section
   Enum _reactionType;                //!< What type of reaction is this
   double _nuBar;                     //!< If this is a fission, specify the nu bar

};

// This class holds an array of reactions for neutrons
class NuclearDataSpecies
{
 public:
   
   void addReaction(NuclearDataReaction::Enum type, double nuBar, qs_vector<double>& energies,
                    const Polynomial& polynomial, double reactionCrossSection);
   
   qs_vector<NuclearDataReaction> _reactions;
};

// For this isotope, store the cross sections. In this case the species is just neutron.
class NuclearDataIsotope
{
 public:
   
   NuclearDataIsotope()
   : _species(1,VAR_MEM){}
   
   qs_vector<NuclearDataSpecies> _species;

};

// Top level class to handle all things related to nuclear data
class NuclearData
{
 public:
   
   NuclearData(int numGroups, double energyLow, double energyHigh);

   int addIsotope(int nReactions,
                  const Polynomial& fissionFunction,
                  const Polynomial& scatterFunction,
                  const Polynomial& absorptionFunction,
                  double nuBar,
                  double totalCrossSection,
                  double fissionWeight, double scatterWeight, double absorptionWeight);

   HOST_DEVICE_CUDA
   int getEnergyGroup(double energy);
   HOST_DEVICE_CUDA
   int getNumberReactions(unsigned int isotopeIndex);
   HOST_DEVICE_CUDA
   double getTotalCrossSection(unsigned int isotopeIndex, unsigned int group);
   HOST_DEVICE_CUDA
   double getReactionCrossSection(unsigned int reactIndex, unsigned int isotopeIndex, unsigned int group);

   int _numEnergyGroups;
   // Store the cross sections and reactions by isotope, which stores
   // it by species
   qs_vector<NuclearDataIsotope> _isotopes;
   // This is the overall energy layout. If we had more than just
   // neutrons, this array would be a vector of vectors.
   qs_vector<double> _energies;

};

#endif

// The input for the nuclear data comes from the material section
// The input looks may like
//
// material NAME
// nIsotope=XXX
// nReactions=XXX
// fissionCrossSection="XXX"
// scatterCrossSection="XXX"
// absorptionCrossSection="XXX"
// nuBar=XXX
// totalCrossSection=XXX
// fissionWeight=XXX
// scatterWeight=XXX
// absorptionWeight=XXX
//
// Material NAME2
// ...
//
// table NAME
// a=XXX
// b=XXX
// c=XXX
// d=XXX
// e=XXX
//
// table NAME2
//
// Each isotope inside a material will have identical cross sections.
// However, it will be treated as unique in the nuclear data.
// Cross sectionsare strings that refer to tables
