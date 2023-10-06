/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef NUCLEAR_DATA_HH
#define NUCLEAR_DATA_HH

#include <sycl/sycl.hpp>
#include <cstdio>
#include <string>
#include "QS_Vector.hh"
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include "qs_assert.hh"
#include "DeclareMacro.hh"
#include "MC_RNG_State.hh"
#include "EnergySpectrum.hh"

using std::pow;

#if defined(HAVE_SYCL)
extern sycl::queue sycl_device_queue; // global variable for device queue
#endif

class Polynomial
{
public:
   Polynomial(double aa, double bb, double cc, double dd, double ee)
       : _aa(aa), _bb(bb), _cc(cc), _dd(dd), _ee(ee) {}

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

   NuclearDataReaction()
       : _reactionType(Enum::Undefined),
         _nuBar(0.0){};

   // Set the cross section values and reaction type
   // Cross sections are scaled to produce the supplied reactionCrossSection at 1MeV.
   inline NuclearDataReaction(
       Enum reactionType, double nuBar, const qs_vector<double> &energies,
       const Polynomial &polynomial, double reactionCrossSection)
       : _crossSection(energies.size() - 1, 0., VAR_MEM),
         _reactionType(reactionType),
         _nuBar(nuBar)
   {
      int nGroups = _crossSection.size();

      for (int ii = 0; ii < nGroups; ++ii)
      {
         double energy = (energies[ii] + energies[ii + 1]) / 2.0;
         _crossSection[ii] = sycl::pow(10.0, (double)polynomial(sycl::log10(energy)));
      }

      // Find the normalization value for the polynomial.  This is the
      // value of the energy group that contains 1 MeV
      double normalization = 0.0;
      for (unsigned ii = 0; ii < nGroups; ++ii)
         if (energies[ii + 1] > 1.) // 1 MeV
         {
            normalization = _crossSection[ii];
            break;
         }
      qs_assert(normalization > 0.);

      // scale to specified reaction cross section
      double scale = reactionCrossSection / normalization;
      for (int ii = 0; ii < nGroups; ++ii)
         _crossSection[ii] *= scale;
   };

   inline HOST_DEVICE
       // Return the cross section for this energy group
       double
       getCrossSection(unsigned int group)
   {
      qs_assert(group < _crossSection.size());
      return _crossSection[group];
   };
   HOST_DEVICE_END

   inline HOST_DEVICE void sampleCollision(
       double incidentEnergy, double material_mass, double *energyOut,
       double *angleOut, int &nOut, uint64_t *seed, int max_production_size)
   {
      double randomNumber;
      switch (_reactionType)
      {
      case Scatter:
         nOut = 1;
         randomNumber = rngSample(seed);
         energyOut[0] = incidentEnergy * (1.0 - (randomNumber * (1.0 / material_mass)));
         randomNumber = rngSample(seed) * 2.0 - 1.0;
         angleOut[0] = randomNumber;
         break;
      case Absorption:
         break;
      case Fission:
      {
         int numParticleOut = (int)(_nuBar + rngSample(seed));
         qs_assert(numParticleOut <= max_production_size);
         nOut = numParticleOut;
         for (int outIndex = 0; outIndex < numParticleOut; outIndex++)
         {
            randomNumber = rngSample(seed) / 2.0 + 0.5;
            energyOut[outIndex] = (20 * randomNumber * randomNumber);
            randomNumber = rngSample(seed) * 2.0 - 1.0;
            angleOut[outIndex] = randomNumber;
         }
      }
      break;
      case Undefined:
#ifdef DEBUG
         printf("_reactionType invalid\n");
#endif
         qs_assert(false);
      }
   };
   HOST_DEVICE_END

   qs_vector<double> _crossSection; //!< tabular data for microscopic cross section
   Enum _reactionType;              //!< What type of reaction is this
   double _nuBar;                   //!< If this is a fission, specify the nu bar
};

// This class holds an array of reactions for neutrons
class NuclearDataSpecies
{
public:
   // Then call this for each reaction to set cross section values
   inline void addReaction(
       NuclearDataReaction::Enum type, double nuBar,
       qs_vector<double> &energies, const Polynomial &polynomial, double reactionCrossSection)
   {
      _reactions.Open();
      _reactions.push_back(NuclearDataReaction(type, nuBar, energies, polynomial, reactionCrossSection));
      _reactions.Close();
   };

   qs_vector<NuclearDataReaction> _reactions;
};

// For this isotope, store the cross sections. In this case the species is just neutron.
class NuclearDataIsotope
{
public:
   NuclearDataIsotope()
       : _species(1, VAR_MEM) {}

   qs_vector<NuclearDataSpecies> _species;
};

// Top level class to handle all things related to nuclear data
class NuclearData
{
public:
   // Set up the energies boundaries of the neutron
   inline NuclearData(int numGroups, double energyLow, double energyHigh) : _energies(numGroups + 1, VAR_MEM)
   {
      qs_assert(energyLow < energyHigh);
      _numEnergyGroups = numGroups;
      _energies[0] = energyLow;
      _energies[numGroups] = energyHigh;
      double logLow = log(energyLow);
      double logHigh = log(energyHigh);
      double delta = (logHigh - logLow) / (numGroups + 1.0);
      for (int energyIndex = 1; energyIndex < numGroups; energyIndex++)
      {
         double logValue = logLow + delta * energyIndex;
         _energies[energyIndex] = exp(logValue);
      }
   };

   inline int addIsotope(
       int nReactions,
       const Polynomial &fissionFunction,
       const Polynomial &scatterFunction,
       const Polynomial &absorptionFunction,
       double nuBar,
       double totalCrossSection,
       double fissionWeight, double scatterWeight, double absorptionWeight)
   {
      _isotopes.Open();
      _isotopes.push_back(NuclearDataIsotope());
      _isotopes.Close();

      double totalWeight = fissionWeight + scatterWeight + absorptionWeight;

      int nFission = nReactions / 3;
      int nScatter = nReactions / 3;
      int nAbsorption = nReactions / 3;
      switch (nReactions % 3)
      {
      case 0:
         break;
      case 1:
         ++nScatter;
         break;
      case 2:
         ++nScatter;
         ++nFission;
         break;
      }

      double fissionCrossSection = (totalCrossSection * fissionWeight) / (nFission * totalWeight);
      double scatterCrossSection = (totalCrossSection * scatterWeight) / (nScatter * totalWeight);
      double absorptionCrossSection = (totalCrossSection * absorptionWeight) / (nAbsorption * totalWeight);

      _isotopes.back()._species[0]._reactions.reserve(nReactions, VAR_MEM);

      for (int ii = 0; ii < nReactions; ++ii)
      {
         NuclearDataReaction::Enum type;
         Polynomial polynomial(0.0, 0.0, 0.0, 0.0, 0.0);
         double reactionCrossSection = 0.;
         // reaction index % 3 is one of the 3 reaction types
         switch (ii % 3)
         {
         case 0:
            type = NuclearDataReaction::Scatter;
            polynomial = scatterFunction;
            reactionCrossSection = scatterCrossSection;
            break;
         case 1:
            type = NuclearDataReaction::Fission;
            polynomial = fissionFunction;
            reactionCrossSection = fissionCrossSection;
            break;
         case 2:
            type = NuclearDataReaction::Absorption;
            polynomial = absorptionFunction;
            reactionCrossSection = absorptionCrossSection;
            break;
         }
         _isotopes.back()._species[0].addReaction(type, nuBar, _energies, polynomial, reactionCrossSection);
      }

      return _isotopes.size() - 1;
   };

   // For this energy, return the group index
   inline HOST_DEVICE int getEnergyGroup(double energy)
   {
      int numEnergies = (int)_energies.size();
      if (energy <= _energies[0])
         return 0;
      if (energy > _energies[numEnergies - 1])
         return numEnergies - 1;

      int high = numEnergies - 1;
      int low = 0;

      while (high != low + 1)
      {
         int mid = (high + low) / 2;
         if (energy < _energies[mid])
            high = mid;
         else
            low = mid;
      }

      return low;
   };
   HOST_DEVICE_END

   inline HOST_DEVICE int getNumberReactions(unsigned int isotopeIndex)
   {
      qs_assert(isotopeIndex < _isotopes.size());
      return (int)_isotopes[isotopeIndex]._species[0]._reactions.size();
   };
   HOST_DEVICE_END

   // General routines to help access data lower down
   // Return the total cross section for this energy group
   inline HOST_DEVICE double getTotalCrossSection(unsigned int isotopeIndex, unsigned int group)
   {
      qs_assert(isotopeIndex < _isotopes.size());
      int numReacts = (int)_isotopes[isotopeIndex]._species[0]._reactions.size();
      double totalCrossSection = 0.0;
      for (int reactIndex = 0; reactIndex < numReacts; reactIndex++)
      {
         totalCrossSection += _isotopes[isotopeIndex]._species[0]._reactions[reactIndex].getCrossSection(group);
      }
      return totalCrossSection;
   };

   // Return the total cross section for this energy group
   inline HOST_DEVICE double getReactionCrossSection(
       unsigned int reactIndex, unsigned int isotopeIndex, unsigned int group)
   {
      qs_assert(isotopeIndex < _isotopes.size());
      qs_assert(reactIndex < _isotopes[isotopeIndex]._species[0]._reactions.size());
      return _isotopes[isotopeIndex]._species[0]._reactions[reactIndex].getCrossSection(group);
   };
   HOST_DEVICE_END

   int _numEnergyGroups;
   // Store the cross sections and reactions by isotope, which stores
   // it by species
   qs_vector<NuclearDataIsotope> _isotopes;
   // This is the overall energy layout. If we had more than just
   // neutrons, this array would be a vector of vectors.
   qs_vector<double> _energies;
};

// Lowest level class at the reaction level
class NuclearDataReaction_d
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

   NuclearDataReaction_d(){};

   // Set the cross section values and reaction type
   // Cross sections are scaled to produce the supplied reactionCrossSection at 1MeV.
   inline NuclearDataReaction_d(
       Enum reactionType, double nuBar, const double *energies, int energiessize,
       const Polynomial &polynomial, double reactionCrossSection)
       : _reactionType(reactionType),
         _nuBar(nuBar)
   {

      _crossSectionSize = energiessize - 1;

      _crossSection = new double[_crossSectionSize];

      int nGroups = _crossSectionSize;

      for (int ii = 0; ii < nGroups; ++ii)
      {
         double energy = (energies[ii] + energies[ii + 1]) / 2.0;
         _crossSection[ii] = sycl::pow(10.0, (double)polynomial(sycl::log10(energy)));
      }

      // Find the normalization value for the polynomial.  This is the
      // value of the energy group that contains 1 MeV
      double normalization = 0.0;
      for (unsigned ii = 0; ii < nGroups; ++ii)
         if (energies[ii + 1] > 1.) // 1 MeV
         {
            normalization = _crossSection[ii];
            break;
         }
      qs_assert(normalization > 0.);

      // scale to specified reaction cross section
      double scale = reactionCrossSection / normalization;
      for (int ii = 0; ii < nGroups; ++ii)
         _crossSection[ii] *= scale;
   };

   inline HOST_DEVICE
       // Return the cross section for this energy group
       double
       getCrossSection(unsigned int group)
   {
      qs_assert(group < _crossSectionSize);
      return _crossSection[group];
   };
   HOST_DEVICE_END

   inline HOST_DEVICE void sampleCollision(
       double incidentEnergy, double material_mass, double *energyOut,
       double *angleOut, int &nOut, uint64_t *seed, int max_production_size)
   {
      double randomNumber;
      switch (_reactionType)
      {
      case Scatter:
         nOut = 1;
         randomNumber = rngSample(seed);
         energyOut[0] = incidentEnergy * (1.0 - (randomNumber * (1.0 / material_mass)));
         randomNumber = rngSample(seed) * 2.0 - 1.0;
         angleOut[0] = randomNumber;
         break;
      case Absorption:
         break;
      case Fission:
      {
         int numParticleOut = (int)(_nuBar + rngSample(seed));
         qs_assert(numParticleOut <= max_production_size);
         nOut = numParticleOut;
         for (int outIndex = 0; outIndex < numParticleOut; outIndex++)
         {
            randomNumber = rngSample(seed) / 2.0 + 0.5;
            energyOut[outIndex] = (20 * randomNumber * randomNumber);
            randomNumber = rngSample(seed) * 2.0 - 1.0;
            angleOut[outIndex] = randomNumber;
         }
      }
      break;
      case Undefined:
#ifdef DEBUG
         printf("_reactionType invalid\n");
#endif
         qs_assert(false);
      }
   };
   HOST_DEVICE_END

   double *_crossSection; //!< tabular data for microscopic cross section
   int _crossSectionSize;
   Enum _reactionType; //!< What type of reaction is this
   double _nuBar;      //!< If this is a fission, specify the nu bar
};

// This class holds an array of reactions for neutrons
class NuclearDataSpecies_d
{
public:
   NuclearDataReaction_d *_reactions;
   int _reactionsSize;
};

// For this isotope, store the cross sections. In this case the species is just neutron.
class NuclearDataIsotope_d
{
public:
   NuclearDataSpecies_d *_species;
   int _speciesSize;
};

// Top level class to handle all things related to nuclear data
class NuclearData_d
{
public:
   // Set up the energies boundaries of the neutron
   inline NuclearData_d(int numGroups, double energyLow, double energyHigh)
   {

      _energies = new double[numGroups + 1];
      qs_assert(energyLow < energyHigh);
      _numEnergyGroups = numGroups;
      _energies[0] = energyLow;
      _energies[numGroups] = energyHigh;
      double logLow = log(energyLow);
      double logHigh = log(energyHigh);
      double delta = (logHigh - logLow) / (numGroups + 1.0);
      for (int energyIndex = 1; energyIndex < numGroups; energyIndex++)
      {
         double logValue = logLow + delta * energyIndex;
         _energies[energyIndex] = exp(logValue);
      }
   };

   // For this energy, return the group index
   inline HOST_DEVICE int getEnergyGroup(double energy)
   {
      int numEnergies = (int)_energiesSize;
      if (energy <= _energies[0])
         return 0;
      if (energy > _energies[numEnergies - 1])
         return numEnergies - 1;

      int high = numEnergies - 1;
      int low = 0;

      while (high != low + 1)
      {
         int mid = (high + low) / 2;
         if (energy < _energies[mid])
            high = mid;
         else
            low = mid;
      }

      return low;
   };
   HOST_DEVICE_END

   inline HOST_DEVICE int getNumberReactions(unsigned int isotopeIndex)
   {
      qs_assert(isotopeIndex < _isotopesSize);
      return (int)_isotopes[isotopeIndex]._species[0]._reactionsSize;
   };
   HOST_DEVICE_END

   // General routines to help access data lower down
   // Return the total cross section for this energy group
   inline HOST_DEVICE double getTotalCrossSection(unsigned int isotopeIndex, unsigned int group)
   {
      qs_assert(isotopeIndex < _isotopesSize);
      int numReacts = (int)_isotopes[isotopeIndex]._species[0]._reactionsSize;
      double totalCrossSection = 0.0;
      for (int reactIndex = 0; reactIndex < numReacts; reactIndex++)
      {
         totalCrossSection += _isotopes[isotopeIndex]._species[0]._reactions[reactIndex].getCrossSection(group);
      }
      return totalCrossSection;
   };

   // Return the total cross section for this energy group
   inline HOST_DEVICE double getReactionCrossSection(
       unsigned int reactIndex, unsigned int isotopeIndex, unsigned int group)
   {
      qs_assert(isotopeIndex < _isotopesSize);
      qs_assert(reactIndex < _isotopes[isotopeIndex]._species[0]._reactionsSize);
      return _isotopes[isotopeIndex]._species[0]._reactions[reactIndex].getCrossSection(group);
   };
   HOST_DEVICE_END

   int _numEnergyGroups;
   // Store the cross sections and reactions by isotope, which stores
   // it by species
   NuclearDataIsotope_d *_isotopes;
   int _isotopesSize;
   // This is the overall energy layout. If we had more than just
   // neutrons, this array would be a vector of vectors.
   double *_energies;
   int _energiesSize;
};
// This has problems as written for GPU code so replaced vectors with arrays
#if 0
// Sample the collision
void NuclearDataReaction::sampleCollision(
   double incidentEnergy, qs_vector<double> &energyOut,
   qs_vector<double> &angleOut, uint64_t* seed)
#endif

inline void copyNuclearData_device(NuclearData *nuclearData, NuclearData_d *NuclearData_h_o)
{
   NuclearData_d *NuclearData_h = (NuclearData_d *)malloc(sizeof(NuclearData_d));

   int isotopesSize = nuclearData->_isotopes.size();
   NuclearDataIsotope_d *nuclearIsotope_I_d;
   nuclearIsotope_I_d = sycl::malloc_device<NuclearDataIsotope_d>(
       isotopesSize, sycl_device_queue);
   NuclearDataIsotope_d *nuclearIsotope_h = (NuclearDataIsotope_d *)malloc(isotopesSize * sizeof(NuclearDataIsotope_d));

   int energiesSize = nuclearData->_energies.size();
   double *nuclearEnergy_I_d;
   nuclearEnergy_I_d =
       sycl::malloc_device<double>(energiesSize, sycl_device_queue);
   double *nuclearEnergy_h = (double *)malloc(energiesSize * sizeof(double));

   for (int j = 0; j < isotopesSize; j++)
   {
      int speciesSize = nuclearData->_isotopes[j]._species.size();

      NuclearDataSpecies_d *nuclearSpecies_I_d;
      nuclearSpecies_I_d = sycl::malloc_device<NuclearDataSpecies_d>(
          speciesSize, sycl_device_queue);
      NuclearDataSpecies_d *nuclearSpecies_h = (NuclearDataSpecies_d *)malloc(speciesSize * sizeof(NuclearDataSpecies_d));
      for (int k = 0; k < speciesSize; k++)
      {

         int reactionsSize = nuclearData->_isotopes[j]._species[k]._reactions.size();

         NuclearDataReaction_d *nuclear_I_d;
         nuclear_I_d = sycl::malloc_device<NuclearDataReaction_d>(
             reactionsSize, sycl_device_queue);

         NuclearDataReaction_d *nuclear_h = (NuclearDataReaction_d *)malloc(reactionsSize * sizeof(NuclearDataReaction_d));
         for (int l = 0; l < reactionsSize; l++)
         {
            double *crossSections_I_d;
            int NumcrossSectionSize = nuclearData->_isotopes[j]._species[k]._reactions[l]._crossSection.size();
            crossSections_I_d = sycl::malloc_device<double>(
                NumcrossSectionSize, sycl_device_queue);

            sycl_device_queue
                .memcpy(crossSections_I_d,
                        nuclearData->_isotopes[j]
                            ._species[k]
                            ._reactions[l]
                            ._crossSection.outputPointer(),
                        NumcrossSectionSize * sizeof(double))
                .wait();
            nuclear_h[l]._crossSectionSize = NumcrossSectionSize;
            nuclear_h[l]._crossSection = crossSections_I_d;
            nuclear_h[l]._reactionType = (NuclearDataReaction_d::Enum)nuclearData->_isotopes[j]._species[k]._reactions[l]._reactionType;
            nuclear_h[l]._nuBar = nuclearData->_isotopes[j]._species[k]._reactions[l]._nuBar;
         }
         sycl_device_queue
             .memcpy(nuclear_I_d, nuclear_h,
                     reactionsSize * sizeof(NuclearDataReaction_d))
             .wait();
         free(nuclear_h);
         nuclearSpecies_h[k]._reactionsSize = reactionsSize;
         nuclearSpecies_h[k]._reactions = nuclear_I_d;
      }

      sycl_device_queue
          .memcpy(nuclearSpecies_I_d, nuclearSpecies_h,
                  speciesSize * sizeof(NuclearDataSpecies_d))
          .wait();
      free(nuclearSpecies_h);
      nuclearIsotope_h[j]._speciesSize = speciesSize;
      nuclearIsotope_h[j]._species = nuclearSpecies_I_d;
   }

   sycl_device_queue
       .memcpy(nuclearIsotope_I_d, nuclearIsotope_h,
               isotopesSize * sizeof(NuclearDataIsotope_d))
       .wait();
   free(nuclearIsotope_h);
   NuclearData_h->_isotopesSize = isotopesSize;
   NuclearData_h->_isotopes = nuclearIsotope_I_d;

   sycl_device_queue
       .memcpy(nuclearEnergy_I_d, nuclearData->_energies.outputPointer(),
               energiesSize * sizeof(double))
       .wait();
   // cudaMemcpy(nuclearEnergy_I_d,nuclearEnergy_h,energiesSize*sizeof(double),cudaMemcpyHostToDevice);
   free(nuclearEnergy_h);
   NuclearData_h->_energiesSize = energiesSize;
   NuclearData_h->_energies = nuclearEnergy_I_d;

   NuclearData_h->_numEnergyGroups = nuclearData->_numEnergyGroups;

   sycl_device_queue
       .memcpy(NuclearData_h_o, NuclearData_h, sizeof(NuclearData_d))
       .wait();
   free(NuclearData_h);
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
