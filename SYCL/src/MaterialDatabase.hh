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

#ifndef MATERIALDATABASE_HH
#define MATERIALDATABASE_HH

#include <sycl/sycl.hpp>
#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "qs_assert.hh"

#if defined(HAVE_SYCL)
extern sycl::queue sycl_device_queue; // global variable for device queue
#endif

// For this material, store the global id in NuclearData of the isotope
class Isotope
{
public:
   Isotope()
       : _gid(0), _atomFraction(0) {}

   Isotope(int isotopeGid, double atomFraction)
       : _gid(isotopeGid), _atomFraction(atomFraction) {}

   ~Isotope() {}

   int _gid; //!< index into NuclearData
   double _atomFraction;
};

// Material information
class Material
{
public:
   std::string _name;
   double _mass;
   qs_vector<Isotope> _iso;

   Material()
       : _name("0"), _mass(1000.0) {}

   Material(const std::string &name)
       : _name(name), _mass(1000.0) {}

   Material(const std::string &name, double mass)
       : _name(name), _mass(mass) {}

   ~Material() {}

   void addIsotope(const Isotope &isotope)
   {
      _iso.Open();
      _iso.push_back(isotope);
      _iso.Close();
   }
};

// Top level class to store material information
class MaterialDatabase
{
public:
   void addMaterial(const Material &material)
   {
      _mat.Open();
      _mat.push_back(material);
      _mat.Close();
   }

   int findMaterial(const std::string &name) const
   {
      for (int matIndex = 0; matIndex < _mat.size(); matIndex++)
      {
         if (_mat[matIndex]._name == name)
         {
            return matIndex;
         }
      }
      qs_assert(false);
      return -1;
   }

   // Store the cross sections and reactions by isotope, which stores it by species
   qs_vector<Material> _mat;
};

// Material information
class Material_d
{
public:
   double _mass;
   int _isosize;
   Isotope *_iso;

   Material_d()
       : _mass(1000.0) {}

   Material_d(const std::string &name)
       : _mass(1000.0) {}

   Material_d(const std::string &name, double mass)
       : _mass(mass) {}

   ~Material_d() {}
};

inline void copyMaterialDatabase_device(MonteCarlo *mcco)
{
   int numMaterials = mcco->_materialDatabase->_mat.size();
   Material_d *materials_h = (Material_d *)malloc(numMaterials * sizeof(Material_d));

   for (int j = 0; j < numMaterials; j++)
   {
      int isosize = mcco->_materialDatabase->_mat[j]._iso.size();
      Isotope *local_I_d;
      local_I_d =
          sycl::malloc_device<Isotope>(isosize, sycl_device_queue);
      sycl_device_queue
          .memcpy(local_I_d,
                  mcco->_materialDatabase->_mat[j]._iso.outputPointer(),
                  isosize * sizeof(Isotope))
          .wait();

      materials_h[j]._isosize = isosize;
      materials_h[j]._iso = local_I_d;
      materials_h[j]._mass = mcco->_materialDatabase->_mat[j]._mass;
   }
   sycl_device_queue
       .memcpy(mcco->_material_d, materials_h,
               numMaterials * sizeof(Material_d))
       .wait();
   free(materials_h);
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
