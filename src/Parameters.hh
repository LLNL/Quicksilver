/// \file
/// Read parameters from command line arguments and input file.

#ifndef PARAMETERS_HH
#define PARAMETERS_HH

#include <string>
#include <vector>
#include <map>
#include <iostream>

struct GeometryParameters
{
   enum Shape{UNDEFINED, BRICK, SPHERE};

   GeometryParameters()
   : materialName(),
     shape(UNDEFINED),
     radius(0.0),
     xCenter(0.0),
     yCenter(0.0),
     zCenter(0.0),
     xMin(0.0),
     yMin(0.0),
     zMin(0.0),
     xMax(0.0),
     yMax(0.0),
     zMax(0.0)
   {};

   std::string materialName;
   Shape shape;
   double radius;
   double xCenter;
   double yCenter;
   double zCenter;
   double xMin;
   double yMin;
   double zMin;
   double xMax;
   double yMax;
   double zMax;
};

struct MaterialParameters
{
   MaterialParameters()
   : name(),
     mass(1000.0),
     totalCrossSection(1.0),
     nIsotopes(10),
     nReactions(9),
     sourceRate(0.0),
     scatteringCrossSection(),
     absorptionCrossSection(),
     fissionCrossSection(),
     scatteringCrossSectionRatio(1.0),
     absorptionCrossSectionRatio(1.0),
     fissionCrossSectionRatio(1.0)
   {};

   std::string name;
   double mass;
   double totalCrossSection;
   int nIsotopes;
   int nReactions;
   double sourceRate;
   std::string scatteringCrossSection;
   std::string absorptionCrossSection;
   std::string fissionCrossSection;
   double scatteringCrossSectionRatio;
   double absorptionCrossSectionRatio;
   double fissionCrossSectionRatio;
};

struct CrossSectionParameters
{
   CrossSectionParameters()
   : name(),
     aa(0.0),
     bb(0.0),
     cc(0.0),
     dd(0.0),
     ee(1.0),
     nuBar(2.4)
   {};

   std::string name;
   double aa;
   double bb;
   double cc;
   double dd;
   double ee;
   double nuBar;
};

struct SimulationParameters
{
   SimulationParameters()
   : inputFile(),
     crossSectionsOut(""),
     boundaryCondition("reflect"),
     energySpectrum(""),
     loadBalance(0),
     cycleTimers(0),
     debugThreads(0),
     nParticles(1000000), // 10^6
     batchSize(0), // default to use nBatches
     nBatches(10),
     nSteps(10),
     nx(10), //speed up early testing
     ny(10),
     nz(10),
//     nx(100),
//     ny(100),
//     nz(100),
     seed(1029384756),
     xDom(0),
     yDom(0),
     zDom(0),
     dt(1e-8),
     fMax(0.1),
     lx(100.0),
     ly(100.0),
     lz(100.0),
     eMin(1e-9),
     eMax(20),
     nGroups(230), 
     lowWeightCutoff(0.001),
     balanceTallyReplications(1),
     fluxTallyReplications(1),
     cellTallyReplications(1),
     coralBenchmark(0)
   {};

   std::string inputFile;        //!< name of input file
   std::string energySpectrum;   //!< enble computing and printing energy spectrum via of energy spectrum file 
   std::string crossSectionsOut; //!< enable or disable printing cross section data to a file
   std::string boundaryCondition;//!< specifies boundary conditions
   int loadBalance;              //!< enable or disable load balancing
   int cycleTimers;              //!< enable or disable cycle timers 
   int debugThreads;             //!< enable or disable thread debugging lines
   uint64_t nParticles;          //!< number of particles
   uint64_t batchSize;           //!< number of particles in a batch
   uint64_t nBatches;            //!< number of batches to start
   int nSteps;                   //!< number of time steps
   int nx;                       //!< number of mesh elements in x-direction
   int ny;                       //!< number of mesh elements in y-direction
   int nz;                       //!< number of mesh elements in z-direction
   int seed;                     //!< random number seed
   int xDom;                     //!< number of MPI ranks in x-direction
   int yDom;                     //!< number of MPI ranks in y-direction
   int zDom;                     //!< number of MPI ranks in z-direction
   double dt;                    //!< time step (seconds)
   double fMax;                  //!< max random fractional displacement of mesh
   double lx;                    //!< size of problem domain in x-direction (cm)
   double ly;                    //!< size of problem domain in y-direction (cm)
   double lz;                    //!< size of problem domain in z-direction (cm)
   double eMin;                  //!< min energy of cross section
   double eMax;                  //!< max energy of cross section
   int nGroups;                  //!< number of groups for cross sections
   double lowWeightCutoff;       //!< low weight roulette cutoff
   int balanceTallyReplications; //!< Number of replications for the balance tallies
   int fluxTallyReplications;    //!< Number of replications for the scalar flux tally
   int cellTallyReplications;    //!< Number of replications for the scalar cell tally
   int coralBenchmark;           //!< enable correctness check for Coral2 benchmark
};

struct Parameters
{
   SimulationParameters                          simulationParams;
   std::vector<GeometryParameters>               geometryParams;
   std::map<std::string, MaterialParameters>     materialParams;
   std::map<std::string, CrossSectionParameters> crossSectionParams;
};

Parameters getParameters(int argc, char** argv);
void printParameters(const Parameters& params, std::ostream& out);

std::ostream& operator<<(std::ostream& out, const SimulationParameters& pp);
std::ostream& operator<<(std::ostream& out, const GeometryParameters& pp);
std::ostream& operator<<(std::ostream& out, const MaterialParameters& pp);
std::ostream& operator<<(std::ostream& out, const CrossSectionParameters& pp);

#endif
