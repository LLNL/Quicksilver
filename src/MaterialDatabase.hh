#ifndef MATERIALDATABASE_HH
#define MATERIALDATABASE_HH

#include <string>
#include <vector>
#include <cstdlib>
#include <cmath>
#include "qs_assert.hh"

// For this material, store the global id in NuclearData of the isotope
class Isotope
{
 public:
   Isotope()
   : _gid(0), _atomFraction(0) { }
   
   Isotope(int isotopeGid, double atomFraction) 
   : _gid(isotopeGid), _atomFraction(atomFraction) { }
  
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
   :   _name(name), _mass(1000.0){}

   Material(const std::string &name, double mass)
   :   _name(name), _mass(mass){}
   
   ~Material() {}

   void addIsotope(const Isotope& isotope)
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
   
   void addMaterial(const Material& material)
   {
      _mat.Open();
      _mat.push_back(material);
      _mat.Close();
   }
   
   int findMaterial(const std::string& name) const
   {
      for (int matIndex = 0; matIndex < _mat.size(); matIndex++)
      {
         if (_mat[matIndex]._name == name) { return matIndex; }
      }
      qs_assert(false);
      return -1;
   }
   
   // Store the cross sections and reactions by isotope, which stores it by species
   qs_vector<Material> _mat;

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
