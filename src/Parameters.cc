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

/// \file
/// Read parameters from command line arguments and input file.

// ToDo:
// 1. set the default number of mpi ranks in each direction
//
// 2. what should we do about cross sections with zero ratio?  is that
//    allowed?  Should they get skipped in building the total cross
//    sections?  Can the corresponding name be omitted from input?
//
// 3. Current input setup allows for different energy range (and other
//    characteristics) for each cross section.  That isn't the way that
//    a real code works.  All of the tables are the same.  Do we need to
//    promote these parameters into the simulation block?  If so, do
//    they become command line parameters?
//
// 4.  How do we warn folks that getParameters is a collective call?
//
// 5.  cmdLineParser.cc could be rewritten to take advantage of C++
//     features such as stringstream.  That would allow tyoe checking of
//     the arguments.  Also general improvement of the code structure.
//
// 6.  Implement anything marked with qs_assert(false)
#include "Parameters.hh"

#include "qs_assert.hh"
#include <utility>
#include <fstream>

#include "cmdLineParser.hh"
#include "parseUtils.hh"
#include "InputBlock.hh"
#include "utilsMpi.hh"

using std::endl;
using std::ifstream;
using std::make_pair;
using std::map;
using std::ostream;
using std::string;
using std::vector;

namespace
{
   void parseCommandLine(int argc, char **argv, Parameters &pp);
   void parseInputFile(const string &filename, Parameters &pp);
   void supplyDefaults(Parameters &params);

   void scanSimulationBlock(const InputBlock &input, Parameters &pp);
   void scanGeometryBlock(const InputBlock &input, Parameters &pp);
   void scanMaterialBlock(const InputBlock &input, Parameters &pp);
   void scanCrossSectionBlock(const InputBlock &input, Parameters &pp);

   void badInputFile(const string &filename);
   void badGeometryBlock(const InputBlock &input);
   void badMaterialBlock(const InputBlock &input);
   void badCrossSectionBlock(const InputBlock &input);
}

/// A one stop shop to get all parameters from the command line and the
/// input file.
///
///  - All MPI ranks scan the command line arguments.
///  - Only rank 0 reads the input file.  The resulting parse tree is
///    broadcast to all other ranks, then all ranks independently scan
///    the parse tree.
///
///  Warnings:
///  0. Everything is case sensitive.
///  1. Misspelled keywords are silently ignored.
///  2. No type checking for command line arguments
///  3. Order of geometries matters.
///  4. For duplicated keywords, last value wins. (Includes multiple
///     Simulation: blocks or multiple Material: or CrossSection: blocks
///     with the same name.
///

Parameters getParameters(int argc, char **argv)
{
   Parameters params;
   parseCommandLine(argc, argv, params);
   const string &filename = params.simulationParams.inputFile;
   const string energyName = params.simulationParams.energySpectrum;
   const string xsecOut = params.simulationParams.crossSectionsOut;

   if (!filename.empty())
      parseInputFile(filename, params);
   if (energyName != "")
      params.simulationParams.energySpectrum = energyName;
   if (xsecOut != "")
      params.simulationParams.crossSectionsOut = xsecOut;
   supplyDefaults(params);
   return params;
}

void printParameters(const Parameters &pp, ostream &out)
{
   int rank = -1;
   mpiComm_rank(MPI_COMM_WORLD, &rank);

   if (rank != 0)
   {
      return;
   }

   out << pp.simulationParams;

   for (unsigned ii = 0; ii < pp.geometryParams.size(); ++ii)
      out << pp.geometryParams[ii];

   for (map<string, MaterialParameters>::const_iterator iter = pp.materialParams.begin();
        iter != pp.materialParams.end(); ++iter)
      out << iter->second;

   for (map<string, CrossSectionParameters>::const_iterator iter = pp.crossSectionParams.begin();
        iter != pp.crossSectionParams.end(); ++iter)
      out << iter->second;
}

ostream &operator<<(ostream &out, const SimulationParameters &pp)
{
   out << "Simulation:\n";
   out << "   dt: " << pp.dt << "\n";
   out << "   fMax: " << pp.fMax << "\n";
   out << "   inputFile: " << pp.inputFile << "\n";
   out << "   energySpectrum: " << pp.energySpectrum << "\n";
   out << "   boundaryCondition: " << pp.boundaryCondition << "\n";
   out << "   loadBalance: " << pp.loadBalance << "\n";
   out << "   cycleTimers: " << pp.cycleTimers << "\n";
   out << "   debugThreads: " << pp.debugThreads << "\n";
   out << "   lx: " << pp.lx << "\n";
   out << "   ly: " << pp.ly << "\n";
   out << "   lz: " << pp.lz << "\n";
   out << "   nParticles: " << pp.nParticles << "\n";
   out << "   batchSize: " << pp.batchSize << "\n";
   out << "   nBatches: " << pp.nBatches << "\n";
   out << "   nSteps: " << pp.nSteps << "\n";
   out << "   nx: " << pp.nx << "\n";
   out << "   ny: " << pp.ny << "\n";
   out << "   nz: " << pp.nz << "\n";
   out << "   seed: " << pp.seed << "\n";
   out << "   xDom: " << pp.xDom << "\n";
   out << "   yDom: " << pp.yDom << "\n";
   out << "   zDom: " << pp.zDom << "\n";
   out << "   eMax: " << pp.eMax << "\n";
   out << "   eMin: " << pp.eMin << "\n";
   out << "   nGroups: " << pp.nGroups << "\n";
   out << "   lowWeightCutoff: " << pp.lowWeightCutoff << "\n";
   out << "   bTally: " << pp.balanceTallyReplications << "\n";
   out << "   fTally: " << pp.fluxTallyReplications << "\n";
   out << "   cTally: " << pp.cellTallyReplications << "\n";
   out << "   coralBenchmark: " << pp.coralBenchmark << "\n";
   out << "   crossSectionsOut:" << pp.crossSectionsOut << "\n";
   out << endl;
   return out;
}

ostream &operator<<(ostream &out, const GeometryParameters &pp)
{
   out << "Geometry:\n";
   out << "   material: " << pp.materialName << "\n";
   switch (pp.shape)
   {
   case GeometryParameters::BRICK:
      out << "   shape: brick\n";
      out << "   xMax: " << pp.xMax << "\n";
      out << "   xMin: " << pp.xMin << "\n";
      out << "   yMax: " << pp.yMax << "\n";
      out << "   yMin: " << pp.yMin << "\n";
      out << "   zMax: " << pp.zMax << "\n";
      out << "   zMin: " << pp.zMin << "\n";
      break;
   case GeometryParameters::SPHERE:
      out << "   shape: sphere\n";
      out << "   xCenter: " << pp.xCenter << "\n";
      out << "   yCenter: " << pp.yCenter << "\n";
      out << "   zCenter: " << pp.zCenter << "\n";
      break;
   default:
      qs_assert(false);
   }
   out << endl;
   return out;
}

ostream &operator<<(ostream &out, const MaterialParameters &pp)
{
   out << "Material:\n";
   out << "   name: " << pp.name << "\n";
   out << "   mass: " << pp.mass << "\n";
   out << "   nIsotopes: " << pp.nIsotopes << "\n";
   out << "   nReactions: " << pp.nReactions << "\n";
   out << "   sourceRate: " << pp.sourceRate << "\n";
   out << "   totalCrossSection: " << pp.totalCrossSection << "\n";
   out << "   absorptionCrossSection: " << pp.absorptionCrossSection << "\n";
   out << "   fissionCrossSection: " << pp.fissionCrossSection << "\n";
   out << "   scatteringCrossSection: " << pp.scatteringCrossSection << "\n";
   out << "   absorptionCrossSectionRatio: " << pp.absorptionCrossSectionRatio << "\n";
   out << "   fissionCrossSectionRatio: " << pp.fissionCrossSectionRatio << "\n";
   out << "   scatteringCrossSectionRatio: " << pp.scatteringCrossSectionRatio << "\n";
   out << endl;
   return out;
}

ostream &operator<<(ostream &out, const CrossSectionParameters &pp)
{
   out << "CrossSection:\n";
   out << "   name: " << pp.name << "\n";
   out << "   A: " << pp.aa << "\n";
   out << "   B: " << pp.bb << "\n";
   out << "   C: " << pp.cc << "\n";
   out << "   D: " << pp.dd << "\n";
   out << "   E: " << pp.ee << "\n";
   out << "   nuBar: " << pp.nuBar << "\n";

   return out;
}

namespace
{
   void parseCommandLine(int argc, char **argv, Parameters &pp)
   {
      SimulationParameters &sp = pp.simulationParams;
      int help = 0;
      char name[1024];
      name[0] = '\0';
      char esName[1024];
      esName[0] = '\0';
      char xsec[1024];
      xsec[0] = '\0';

      addArg("help", 'h', 0, 'i', &(help), 0, "print this message");
      addArg("dt", 'D', 1, 'd', &(sp.dt), 0, "time step (seconds)");
      addArg("fMax", 'f', 1, 'd', &(sp.fMax), 0, "max random mesh node displacement");
      addArg("inputFile", 'i', 1, 's', &(name), sizeof(name), "name of input file");
      addArg("energySpectrum", 'e', 1, 's', &(esName), sizeof(esName), "name of energy spectrum output file");
      addArg("crossSectionsOut", 'S', 1, 's', &(xsec), sizeof(xsec), "name of cross section output file");
      addArg("loadBalance", 'l', 0, 'i', &(sp.loadBalance), 0, "enable/disable load balancing");
      addArg("cycleTimers", 'c', 1, 'i', &(sp.cycleTimers), 0, "enable/disable cycle timers");
      addArg("debugThreads", 't', 1, 'i', &(sp.debugThreads), 0, "set thread debug level to 1, 2, 3");
      addArg("lx", 'X', 1, 'd', &(sp.lx), 0, "x-size of simulation (cm)");
      addArg("ly", 'Y', 1, 'd', &(sp.ly), 0, "y-size of simulation (cm)");
      addArg("lz", 'Z', 1, 'd', &(sp.lz), 0, "z-size of simulation (cm)");
      addArg("nParticles", 'n', 1, 'u', &(sp.nParticles), 0, "number of particles");
      addArg("batchSize", 'g', 1, 'u', &(sp.batchSize), 0, "number of particles in a vault/batch");
      addArg("nBatches", 'b', 1, 'u', &(sp.nBatches), 0, "number of vault/batch to start (sets batchSize automaticaly)");
      addArg("nSteps", 'N', 1, 'i', &(sp.nSteps), 0, "number of time steps");
      addArg("nx", 'x', 1, 'i', &(sp.nx), 0, "number of mesh elements in x");
      addArg("ny", 'y', 1, 'i', &(sp.ny), 0, "number of mesh elements in y");
      addArg("nz", 'z', 1, 'i', &(sp.nz), 0, "number of mesh elements in z");
      addArg("seed", 's', 1, 'i', &(sp.seed), 0, "random number seed");
      addArg("xDom", 'I', 1, 'i', &(sp.xDom), 0, "number of MPI ranks in x");
      addArg("yDom", 'J', 1, 'i', &(sp.yDom), 0, "number of MPI ranks in y");
      addArg("zDom", 'K', 1, 'i', &(sp.zDom), 0, "number of MPI ranks in z");
      addArg("bTally", 'B', 1, 'i', &(sp.balanceTallyReplications), 0, "number of balance tally replications");
      addArg("fTally", 'F', 1, 'i', &(sp.fluxTallyReplications), 0, "number of scalar flux tally replications");
      addArg("cTally", 'C', 1, 'i', &(sp.cellTallyReplications), 0, "number of scalar cell tally replications");

      processArgs(argc, argv);

      sp.inputFile = name;
      sp.energySpectrum = esName;
      sp.crossSectionsOut = xsec;

      if (help)
      {
         int rank = -1;
         mpiComm_rank(MPI_COMM_WORLD, &rank);
         if (rank == 0)
         {
            printArgs();
         }
         freeArgs();
         exit(2);
      }
      freeArgs();
   }
}

namespace
{
   void parseInputFile(const string &filename, Parameters &pp)
   {
      vector<InputBlock> parseTree;
      int myRank;
      mpiComm_rank(MPI_COMM_WORLD, &myRank);
      if (myRank == 0)
      { // fill parse tree
         ifstream in(filename.c_str());
         if (!in)
            badInputFile(filename);
         if (!in.good())
         {
            std::cerr << "ERROR : Input file '" << filename << "' does not exist " << std::endl;
            return;
         }
         string line;
         getline(in, line);
         while (!in.eof())
         {
            string blockName;
            if (blockStart(line, blockName))
            {
               parseTree.push_back(InputBlock(blockName));
               line = readBlock(parseTree.back(), in);
            }
            else
               getline(in, line);
         }
      } // fill parse tree

      { // broadcast parse tree
         int nBlocks = parseTree.size();
         mpiBcast(&nBlocks, 1, MPI_INT, 0, MPI_COMM_WORLD);
         for (unsigned ii = 0; ii < nBlocks; ++ii)
         {
            vector<char> buffer;
            ;
            if (myRank == 0)
               parseTree[ii].serialize(buffer);

            int size = buffer.size();
            mpiBcast(&size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            if (myRank != 0)
               buffer.resize(size);
            mpiBcast(&buffer[0], size, MPI_BYTE, 0, MPI_COMM_WORLD);

            if (myRank != 0)
            {
               parseTree.push_back(InputBlock(""));
               parseTree.back().deserialize(buffer);
            }
         }
      } // broadcast

      for (unsigned ii = 0; ii < parseTree.size(); ++ii)
      {
         if (parseTree[ii].name() == "Simulation")
            scanSimulationBlock(parseTree[ii], pp);
         if (parseTree[ii].name() == "Geometry")
            scanGeometryBlock(parseTree[ii], pp);
         if (parseTree[ii].name() == "Material")
            scanMaterialBlock(parseTree[ii], pp);
         if (parseTree[ii].name() == "CrossSection")
            scanCrossSectionBlock(parseTree[ii], pp);
      }
   }
}

namespace
{
   // Examine the parameter values supplied by the user and determine
   // whether the default problem needs to be added to the specification.
   //
   // Method:  If a geometry has been specified, the user is on their own
   // for a complete set of geometries, materials, and cross sections.
   void supplyDefaults(Parameters &params)
   {
      if (!params.geometryParams.empty())
         return;

      CrossSectionParameters flatCrossSection;
      flatCrossSection.name = "flat";

      params.crossSectionParams[flatCrossSection.name] = flatCrossSection;

      MaterialParameters sourceMaterial;
      sourceMaterial.name = "sourceMaterial";
      sourceMaterial.mass = 1000.0;
      sourceMaterial.sourceRate = 1e10;
      sourceMaterial.scatteringCrossSection = "flat";
      sourceMaterial.absorptionCrossSection = "flat";
      sourceMaterial.fissionCrossSection = "flat";
      sourceMaterial.fissionCrossSectionRatio = 0.1;

      params.materialParams[sourceMaterial.name] = sourceMaterial;

      GeometryParameters sourceGeometry;
      sourceGeometry.materialName = "sourceMaterial";
      sourceGeometry.shape = GeometryParameters::BRICK;
      sourceGeometry.xMax = params.simulationParams.lx;
      sourceGeometry.yMax = params.simulationParams.ly;
      sourceGeometry.zMax = params.simulationParams.lz;

      params.geometryParams.push_back(sourceGeometry);
   }
}

namespace
{
   void scanSimulationBlock(const InputBlock &input, Parameters &pp)
   {
      SimulationParameters &sp = pp.simulationParams;
      input.getValue<string>("energySpectrum", sp.energySpectrum);
      input.getValue<string>("crossSectionsOut", sp.crossSectionsOut);
      input.getValue<string>("boundaryCondition", sp.boundaryCondition);
      input.getValue<double>("dt", sp.dt);
      input.getValue<double>("fMax", sp.fMax);
      input.getValue<int>("loadBalance", sp.loadBalance);
      input.getValue<int>("cycleTimers", sp.cycleTimers);
      input.getValue<int>("debugThreads", sp.debugThreads);
      input.getValue<double>("lx", sp.lx);
      input.getValue<double>("ly", sp.ly);
      input.getValue<double>("lz", sp.lz);
      input.getValue<uint64_t>("nParticles", sp.nParticles);
      input.getValue<uint64_t>("batchSize", sp.batchSize);
      input.getValue<uint64_t>("nBatches", sp.nBatches);
      input.getValue<int>("nSteps", sp.nSteps);
      input.getValue<int>("nx", sp.nx);
      input.getValue<int>("ny", sp.ny);
      input.getValue<int>("nz", sp.nz);
      input.getValue<int>("seed", sp.seed);
      input.getValue<int>("xDom", sp.xDom);
      input.getValue<int>("yDom", sp.yDom);
      input.getValue<int>("zDom", sp.zDom);
      input.getValue<double>("eMax", sp.eMax);
      input.getValue<double>("eMin", sp.eMin);
      input.getValue<int>("nGroups", sp.nGroups);
      input.getValue<double>("lowWeightCutoff", sp.lowWeightCutoff);
      input.getValue<int>("bTally", sp.balanceTallyReplications);
      input.getValue<int>("fTally", sp.fluxTallyReplications);
      input.getValue<int>("cTally", sp.cellTallyReplications);
      input.getValue<int>("coralBenchmark", sp.coralBenchmark);
   }
}

namespace
{
   void scanGeometryBlock(const InputBlock &input, Parameters &pp)
   {
      pp.geometryParams.push_back(GeometryParameters());
      GeometryParameters &gg = pp.geometryParams.back();
      input.getValue<string>("material", gg.materialName);
      string shape;
      input.getValue<string>("shape", shape);
      if (shape == "brick")
      {
         gg.shape = GeometryParameters::BRICK;
         input.getValue<double>("xMax", gg.xMax);
         input.getValue<double>("xMin", gg.xMin);
         input.getValue<double>("yMax", gg.yMax);
         input.getValue<double>("yMin", gg.yMin);
         input.getValue<double>("zMax", gg.zMax);
         input.getValue<double>("zMin", gg.zMin);
      }
      else if (shape == "sphere")
      {
         gg.shape = GeometryParameters::SPHERE;
         input.getValue<double>("radius", gg.radius);
         input.getValue<double>("xCenter", gg.xCenter);
         input.getValue<double>("yCenter", gg.yCenter);
         input.getValue<double>("zCenter", gg.zCenter);
      }
      else
         badGeometryBlock(input);
   }
}
namespace
{
   void scanMaterialBlock(const InputBlock &input, Parameters &pp)
   {
      string materialName;
      input.getValue<string>("name", materialName);
      if (materialName.empty())
         badMaterialBlock(input);
      MaterialParameters &mp = pp.materialParams[materialName];
      mp.name = materialName;
      input.getValue<double>("mass", mp.mass);
      input.getValue<string>("absorptionCrossSection", mp.absorptionCrossSection);
      input.getValue<double>("absorptionCrossSectionRatio", mp.absorptionCrossSectionRatio);
      input.getValue<string>("fissionCrossSection", mp.fissionCrossSection);
      input.getValue<double>("fissionCrossSectionRatio", mp.fissionCrossSectionRatio);
      input.getValue<int>("nIsotopes", mp.nIsotopes);
      input.getValue<int>("nReactions", mp.nReactions);
      input.getValue<double>("totalCrossSection", mp.totalCrossSection);
      input.getValue<string>("scatteringCrossSection", mp.scatteringCrossSection);
      input.getValue<double>("scatteringCrossSectionRatio", mp.scatteringCrossSectionRatio);
      input.getValue<double>("sourceRate", mp.sourceRate);
   }
}
namespace
{
   void scanCrossSectionBlock(const InputBlock &input, Parameters &pp)
   {
      string crossSectionName;
      input.getValue<string>("name", crossSectionName);
      if (crossSectionName.empty())
         badCrossSectionBlock(input);
      CrossSectionParameters &cp = pp.crossSectionParams[crossSectionName];
      cp.name = crossSectionName;
      input.getValue<double>("A", cp.aa);
      input.getValue<double>("B", cp.bb);
      input.getValue<double>("C", cp.cc);
      input.getValue<double>("D", cp.dd);
      input.getValue<double>("E", cp.ee);
      input.getValue<double>("nuBar", cp.nuBar);
   }
}

namespace
{
   void badInputFile(const string &filename) { qs_assert(false); }
   void badGeometryBlock(const InputBlock &input)
   {
      // didn't specify shape.
      // Must be brick or sphere
      qs_assert(false);
   }
   void badMaterialBlock(const InputBlock &input)
   {
      // didn't specify a name
      qs_assert(false);
   }
   void badCrossSectionBlock(const InputBlock &input) { qs_assert(false); }
}
