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

#ifndef MACROSCOPIC_CROSS_SECTION_HH
#define MACROSCOPIC_CROSS_SECTION_HH

#include "DeclareMacro.hh"

#include "MacroscopicCrossSection.hh"
#include "MonteCarlo.hh"
#include "MaterialDatabase.hh"
#include "NuclearData.hh"
#include "MC_Cell_State.hh"

class MonteCarlo;

HOST_DEVICE
double macroscopicCrossSection(MonteCarlo *monteCarlo, int reactionIndex, int domainIndex, int cellIndex,
                               int isoIndex, int energyGroup);
HOST_DEVICE_END

HOST_DEVICE
double weightedMacroscopicCrossSection(MonteCarlo *monteCarlo, int taskIndex, int domainIndex,
                                       int cellIndex, int energyGroup);
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//  Routine MacroscopicCrossSection calculates the number-density-weighted macroscopic cross
//  section of a cell.
//
//  A reactionIndex of -1 means total cross section.
//----------------------------------------------------------------------------------------------------------------------

inline HOST_DEVICE double macroscopicCrossSection(MonteCarlo *monteCarlo, int reactionIndex, int domainIndex, int cellIndex,
                                                  int isoIndex, int energyGroup)
{
// Initialize various data items.
#ifdef __SYCL_DEVICE_ONLY__
   int globalMatIndex = monteCarlo->domain_d[domainIndex].cell_state[cellIndex]._material;
   double atomFraction = monteCarlo->_material_d[globalMatIndex]._iso[isoIndex]._atomFraction;
#else
   int globalMatIndex = monteCarlo->domain[domainIndex].cell_state[cellIndex]._material;
   double atomFraction = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._atomFraction;
#endif

   double microscopicCrossSection = 0.0;
   // The cell number density is the fraction of the atoms in cell
   // volume of this isotope.  We set this (elsewhere) to 1/nIsotopes.
   // This is a statement that we treat materials as if all of their
   // isotopes are present in equal amounts

#ifdef __SYCL_DEVICE_ONLY__
   double cellNumberDensity = monteCarlo->domain_d[domainIndex].cell_state[cellIndex]._cellNumberDensity;
   int isotopeGid = monteCarlo->_material_d[globalMatIndex]._iso[isoIndex]._gid;
#else
   double cellNumberDensity = monteCarlo->domain[domainIndex].cell_state[cellIndex]._cellNumberDensity;
   int isotopeGid = monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso[isoIndex]._gid;
#endif
   if (atomFraction == 0.0 || cellNumberDensity == 0.0)
   {
      return 1e-20;
   }

#ifdef __SYCL_DEVICE_ONLY__
   if (reactionIndex < 0)
   {
      // Return total cross section
      microscopicCrossSection = monteCarlo->_nuclearData_d->getTotalCrossSection(isotopeGid, energyGroup);
   }
   else
   {
      // Return the reaction cross section
      microscopicCrossSection = monteCarlo->_nuclearData_d->getReactionCrossSection((unsigned int)reactionIndex, isotopeGid, energyGroup);
   }
#else
   if (reactionIndex < 0)
   {
      // Return total cross section
      microscopicCrossSection = monteCarlo->_nuclearData->getTotalCrossSection(isotopeGid, energyGroup);
   }
   else
   {
      // Return the reaction cross section
      microscopicCrossSection = monteCarlo->_nuclearData->getReactionCrossSection((unsigned int)reactionIndex, isotopeGid, energyGroup);
   }
#endif

   return atomFraction * cellNumberDensity * microscopicCrossSection;
}
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//  Routine weightedMacroscopicCrossSection calculates the number-density-weighted
//  macroscopic cross section of the collection of isotopes in a cell.
// dfr Weighted is a bit of a misnomer here, since there is no weighting
// applied by this routine.  In Mercury we would weight for multiple
// materials in a cell.
//----------------------------------------------------------------------------------------------------------------------
inline HOST_DEVICE double weightedMacroscopicCrossSection(MonteCarlo *monteCarlo, int taskIndex, int domainIndex,
                                                          int cellIndex, int energyGroup)
{
#ifdef __SYCL_DEVICE_ONLY__
   double *precomputedCrossSection =
       &monteCarlo->domain_d[domainIndex].cell_state[cellIndex]._total[energyGroup];
#else
   double *precomputedCrossSection =
       &monteCarlo->domain[domainIndex].cell_state[cellIndex]._total[energyGroup];
#endif
   qs_assert(precomputedCrossSection != NULL);
   if (*precomputedCrossSection > 0.0)
      return *precomputedCrossSection;

#ifdef __SYCL_DEVICE_ONLY__
   int globalMatIndex = monteCarlo->domain_d[domainIndex].cell_state[cellIndex]._material;
   int nIsotopes = (int)monteCarlo->_material_d[globalMatIndex]._isosize;
#else
   int globalMatIndex = monteCarlo->domain[domainIndex].cell_state[cellIndex]._material;
   int nIsotopes = (int)monteCarlo->_materialDatabase->_mat[globalMatIndex]._iso.size();
#endif
   double sum = 0.0;
   for (int isoIndex = 0; isoIndex < nIsotopes; isoIndex++)
   {
      sum += macroscopicCrossSection(monteCarlo, -1, domainIndex, cellIndex,
                                     isoIndex, energyGroup);
   }

   ATOMIC_WRITE(*precomputedCrossSection, sum);

   return sum;
}
HOST_DEVICE_END
#endif
