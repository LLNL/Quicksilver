#include "initMC.hh"
#include <vector>
#include <set>
#include <map>
#include <iostream>
#include <sched.h>
#include "QS_Vector.hh"
#include "utilsMpi.hh"
#include "MonteCarlo.hh"
#include "MC_Processor_Info.hh"
#include "DecompositionObject.hh"
#include "GlobalFccGrid.hh"
#include "MeshPartition.hh"
#include "CommObject.hh"
#include "SharedMemoryCommObject.hh"
#include "MpiCommObject.hh"
#include "MC_Vector.hh"
#include "NuclearData.hh"
#include "MaterialDatabase.hh"
#include "MC_Time_Info.hh"
#include "Tallies.hh"
#include "MC_Base_Particle.hh"
#include "cudaUtils.hh"
#include "cudaFunctions.hh"

using std::vector;
using std::string;
using std::set;
using std::cout;
using std::endl;
using std::map;
using std::make_pair;

namespace
{
   void initGPUInfo(MonteCarlo* monteCarlo);
   void initNuclearData(MonteCarlo* monteCarlo, const Parameters& params);
   void initMesh(MonteCarlo* monteCarlo, const Parameters& params);
   void initTallies(MonteCarlo* monteCarlo, const Parameters& params);
   void initTimeInfo(MonteCarlo* monteCarlo, const Parameters& params);
   void initializeCentersRandomly(int nCenters,
                                  const GlobalFccGrid& grid,
                                  vector<MC_Vector>& centers);
   void initializeCentersGrid(double lx, double ly, double lz,
                              int xDom, int yDom, int zDom,
                              vector<MC_Vector>& centers);
   void consistencyCheck(int myRank, const qs_vector<MC_Domain>& domain);
   void checkCrossSections(MonteCarlo* monteCarlo, const Parameters& params);

}

MonteCarlo* initMC(const Parameters& params)
{
   MonteCarlo* monteCarlo;
   #ifdef HAVE_UVM
      void* ptr;
      cudaMallocManaged( &ptr, sizeof(MonteCarlo), cudaMemAttachGlobal );
      monteCarlo = new(ptr) MonteCarlo(params);
   #else
     monteCarlo = new MonteCarlo(params);
   #endif
   initGPUInfo(monteCarlo);
   initTimeInfo(monteCarlo, params);
   initNuclearData(monteCarlo, params);
   initMesh(monteCarlo, params);
   initTallies(monteCarlo, params);

   MC_Base_Particle::Update_Counts();

   //   used when debugging cross sections
   checkCrossSections(monteCarlo, params);
   return monteCarlo;
}

namespace
{
//Init GPU usage information
   void initGPUInfo( MonteCarlo* monteCarlo)
   {
      #if defined(HAVE_OPENMP_TARGET)
         int Ngpus = omp_get_num_devices();
      #elif defined(HAVE_CUDA)
         int Ngpus;
         cudaGetDeviceCount(&Ngpus);
      #else
         int Ngpus = 0;
      #endif

         if( Ngpus != 0 )
         {
            #if defined(HAVE_OPENMP_TARGET) || defined(HAVE_CUDA)
            monteCarlo->processor_info->use_gpu = 1;
            int GPUID = monteCarlo->processor_info->rank%Ngpus;
            monteCarlo->processor_info->gpu_id = GPUID;
            
            #if defined(HAVE_OPENMP_TARGET)
                omp_set_default_device(GPUID);
            #endif

            cudaSetDevice(GPUID);
            //cudaDeviceSetLimit( cudaLimitStackSize, 64*1024 );
            #endif
         }
         else
         {
            monteCarlo->processor_info->use_gpu = 0;
            monteCarlo->processor_info->gpu_id = -1;
         }
#ifdef USE_OPENMP_NO_GPU
         monteCarlo->processor_info->use_gpu = 0;
         monteCarlo->processor_info->gpu_id = -1;
#endif

#ifdef HAVE_CUDA
    if( monteCarlo->processor_info->use_gpu )
        warmup_kernel();
#endif

         //printf("monteCarlo->processor_info->use_gpu = %d\n", monteCarlo->processor_info->use_gpu);
         
   }
}


/// Initializes both the NuclearData and the MaterialDatabase.  These
/// two structures are inherently linked since the isotopeGids stored in
/// the MaterialDatabase must correspond to the isotope indices in the
/// NuclearData.
namespace
{
   void initNuclearData(MonteCarlo* monteCarlo, const Parameters& params)
   {
      #if defined HAVE_UVM
         void *ptr1, *ptr2;
         cudaMallocManaged( &ptr1, sizeof(NuclearData), cudaMemAttachGlobal );
         cudaMallocManaged( &ptr2, sizeof(MaterialDatabase), cudaMemAttachGlobal );

         monteCarlo->_nuclearData = new(ptr1) NuclearData(params.simulationParams.nGroups,
                                                          params.simulationParams.eMin,
                                                          params.simulationParams.eMax);
         monteCarlo->_materialDatabase = new(ptr2) MaterialDatabase();
     #else
         monteCarlo->_nuclearData = new NuclearData(params.simulationParams.nGroups,
                                                    params.simulationParams.eMin,
                                                    params.simulationParams.eMax);
         monteCarlo->_materialDatabase = new MaterialDatabase();
     #endif

     map<string, Polynomial> crossSection;
     for (auto crossSectionIter = params.crossSectionParams.begin();
          crossSectionIter != params.crossSectionParams.end();
          crossSectionIter++)
     {
        const CrossSectionParameters& cp = crossSectionIter->second;
        crossSection.insert(make_pair(cp.name, Polynomial(cp.aa, cp.bb, cp.cc, cp.dd, cp.ee)));
     }
     
     int num_isotopes  = 0;
     int num_materials = 0;
     
     for( auto matIter = params.materialParams.begin(); matIter != params.materialParams.end(); matIter++ )
     {
        const MaterialParameters& mp = matIter->second;
        num_isotopes += mp.nIsotopes;
        num_materials++;
     }
     
     monteCarlo->_nuclearData->_isotopes.reserve( num_isotopes, VAR_MEM );
     monteCarlo->_materialDatabase->_mat.reserve( num_materials, VAR_MEM );
     
     for (auto matIter = params.materialParams.begin();
          matIter != params.materialParams.end(); matIter++)
     {
        const MaterialParameters& mp = matIter->second;
        Material material(mp.name, mp.mass);
        double nuBar = params.crossSectionParams.at(mp.fissionCrossSection).nuBar;
        material._iso.reserve( mp.nIsotopes, VAR_MEM );
        
        for (int iIso=0; iIso<mp.nIsotopes; ++iIso)
        {
           int isotopeGid = monteCarlo->_nuclearData->addIsotope(
              mp.nReactions,
              crossSection.at(mp.fissionCrossSection),
              crossSection.at(mp.scatteringCrossSection),
              crossSection.at(mp.absorptionCrossSection),
              nuBar,
              mp.totalCrossSection,
              mp.fissionCrossSectionRatio,
              mp.scatteringCrossSectionRatio,
              mp.absorptionCrossSectionRatio);
           
           // atomFraction for each isotope is 1/nIsotopes.  Treats all
           // isotopes as equally prevalent.
           material.addIsotope(Isotope(isotopeGid, 1.0/mp.nIsotopes));
        }
        monteCarlo->_materialDatabase->addMaterial(material);
     }
   }
}

namespace
{
   void consistencyCheck(int myRank, const qs_vector<MC_Domain>& domain)
   {
      if (myRank == 0) { cout << "Starting Consistency Check" <<endl; }
      unsigned nDomains = domain.size();
      for (int iDomain=0; iDomain<nDomains; ++iDomain)
      {
         const MC_Mesh_Domain& mesh = domain[iDomain].mesh;
         unsigned nCells = mesh._cellConnectivity.size();
         for (unsigned iCell=0; iCell<nCells; ++iCell)
         {
            for (unsigned iFacet = 0; iFacet<24; ++iFacet)
            {
               const MC_Location& current =
                  mesh._cellConnectivity[iCell]._facet[iFacet].subfacet.current;
               qs_assert(current.cell == iCell);

               const MC_Location& adjacent =
                  mesh._cellConnectivity[iCell]._facet[iFacet].subfacet.adjacent;

               int jDomain = adjacent.domain;
               int jCell = adjacent.cell;
               int jFacet = adjacent.facet;

               const Subfacet_Adjacency& backside = domain[jDomain].mesh._cellConnectivity[jCell]._facet[jFacet].subfacet;

               qs_assert (backside.adjacent.domain == iDomain);
               qs_assert (backside.adjacent.cell == iCell);
               qs_assert (backside.adjacent.facet == iFacet);
            }
         }
      }
      if (myRank == 0) { cout << "Finished Consistency Check" <<endl; }
   }
}


namespace
{
   void initMesh(MonteCarlo* monteCarlo, const Parameters& params)
   {
      int nx = params.simulationParams.nx;
      int ny = params.simulationParams.ny;
      int nz = params.simulationParams.nz;
      double lx = params.simulationParams.lx;
      double ly = params.simulationParams.ly;
      double lz = params.simulationParams.lz;
      int xDom = params.simulationParams.xDom;
      int yDom = params.simulationParams.yDom;
      int zDom = params.simulationParams.zDom;
      
      int myRank, nRanks;
      mpiComm_rank(MPI_COMM_WORLD, &myRank);
      mpiComm_size(MPI_COMM_WORLD, &nRanks);
      
      int nDomainsPerRank = 1; // SAD set this to 1 for some types of tests
      if( xDom == 0 && yDom == 0 && zDom == 0 )
         if (nRanks == 1)
            nDomainsPerRank = 4;
      
      DecompositionObject ddc(myRank, nRanks, nDomainsPerRank, 0);
      vector<int> myDomainGid = ddc.getAssignedDomainGids();
      
      GlobalFccGrid globalGrid(nx, ny, nz, lx, ly, lz);
      
      int nCenters = nRanks*nDomainsPerRank;
      vector<MC_Vector> domainCenter;
      if (xDom == 0 && yDom == 0 && zDom == 0)
         initializeCentersRandomly(nCenters, globalGrid, domainCenter);
      else
         initializeCentersGrid(lx, ly, lz, xDom, yDom, zDom, domainCenter);
      
      qs_assert(domainCenter.size() == nCenters);
      
      vector<MeshPartition> partition;
      {
         int foremanRank = myRank;
         for (unsigned ii=0; ii<myDomainGid.size(); ++ii)
         {
            partition.push_back(MeshPartition(myDomainGid[ii], ii, foremanRank));
            qs_assert(ddc.getIndex(myDomainGid[ii]) == ii);
         }
      }
      
      CommObject* comm = 0;
      if (nRanks == 1)
         comm = new SharedMemoryCommObject(partition);
      else if (nRanks > 1 && nDomainsPerRank == 1)
         comm = new MpiCommObject(MPI_COMM_WORLD, ddc);
      else
         qs_assert(false);
      
      for (unsigned ii=0; ii<myDomainGid.size(); ++ii)
      {
         if (myRank == 0) { cout << "Building partition " << myDomainGid[ii] << endl; }
         partition[ii].buildMeshPartition(globalGrid, domainCenter, comm);
      }
      
      mpiBarrier(MPI_COMM_WORLD);
      if (myRank ==0 ) { cout << "done building" << endl; }
      mpiBarrier(MPI_COMM_WORLD);
      
      delete comm;
      
      monteCarlo->domain.reserve(myDomainGid.size(),VAR_MEM);
      monteCarlo->domain.Open();
      for (unsigned ii=0; ii<myDomainGid.size(); ++ii)
      {
         if (myRank == 0) { cout << "Building MC_Domain " << ii << endl; }
         monteCarlo->domain.push_back(
            MC_Domain(partition[ii], globalGrid, ddc, params, *monteCarlo->_materialDatabase,
                      params.simulationParams.nGroups));
      }
      monteCarlo->domain.Close();
      
      if (nRanks == 1)
         consistencyCheck(myRank, monteCarlo->domain);
      
      if (myRank == 0) { cout << "Finished initMesh" <<endl; }
   }
}

namespace
{
   void initTallies(MonteCarlo* monteCarlo, const Parameters& params)
   {
      monteCarlo->_tallies->InitializeTallies(
         monteCarlo,  
         params.simulationParams.balanceTallyReplications,
         params.simulationParams.fluxTallyReplications,
         params.simulationParams.cellTallyReplications
      );
   }
}

namespace
{
   void initTimeInfo(MonteCarlo* monteCarlo, const Parameters& params)
   {
      monteCarlo->time_info->time_step = params.simulationParams.dt;
   }
}

namespace
{
   // scatter the centers (somewhat) randomly
   void initializeCentersRandomly(int nCenters,
                          const GlobalFccGrid& grid,
                          vector<MC_Vector>& centers)
   {
      set<Tuple> picked;
      do
      {
         Tuple iTuple(drand48()*grid.nx()/2,
                      drand48()*grid.ny()/2,
                      drand48()*grid.nz()/2);

         if (!picked.insert(iTuple).second)
            continue;

         iTuple += iTuple; // iTuple *= 2;
         Long64 iCell = grid.cellTupleToIndex(iTuple);
         MC_Vector r = grid.cellCenter(iCell);
         centers.push_back(r);
      } while (centers.size() < nCenters);
   }
}

namespace
{
   void initializeCentersGrid(double lx, double ly, double lz,
                              int xDom, int yDom, int zDom,
                              vector<MC_Vector>& centers)
   {
      double dx = lx/xDom;
      double dy = ly/yDom;
      double dz = lz/zDom;
      for (int ix=0; ix<xDom; ++ix)
         for (int iy=0; iy<yDom; ++iy)
            for (int iz=0; iz<zDom; ++iz)
               centers.push_back(
                  MC_Vector( (0.5+ix)*dx, (0.5+iy)*dy, (0.5+iz)*dz )
               );
   }
}

namespace
{
   // This function is useful for debugging but is not called in ordinary
   // use of the code.  Uncomment the call to this function in initMC()
   // if you want to get plot data for the cross sections.
   void checkCrossSections(MonteCarlo* monteCarlo, const Parameters& params)
   {
      if( monteCarlo->_params.simulationParams.crossSectionsOut == "" ) return;

      struct XC_Data
      {
         XC_Data() : absorption(0.), fission(0.), scatter(0.){}
         double absorption;
         double fission;
         double scatter;
      };
  
      NuclearData* nd = monteCarlo->_nuclearData;
      int nGroups = nd->_energies.size() - 1;
      vector<double> energy(nGroups);
      for (unsigned ii=0; ii<nGroups; ++ii)
         energy[ii] = (nd->_energies[ii] + nd->_energies[ii+1])/2.0;
  
  
      MaterialDatabase* matDB = monteCarlo->_materialDatabase;
      unsigned nMaterials = matDB->_mat.size();
    
      map<string, vector<XC_Data> > xcTable;
  
    
      // for each material
      for (unsigned iMat=0; iMat<nMaterials; ++iMat)
      {
         const string& materialName = matDB->_mat[iMat]._name;
         vector<XC_Data>& xcVec = xcTable[materialName];
         xcVec.resize(nGroups);
         unsigned nIsotopes = matDB->_mat[iMat]._iso.size();
         // for each isotope
         for (unsigned iIso=0; iIso<nIsotopes; ++iIso)
         {
            int isotopeGid = monteCarlo->_materialDatabase->_mat[iMat]._iso[iIso]._gid;
            unsigned nReactions = nd->_isotopes[isotopeGid]._species[0]._reactions.size();
            // for each reaction
            for (unsigned iReact=0; iReact<nReactions; ++iReact)
            {
               // loop over energies
               NuclearDataReaction& reaction = nd->_isotopes[isotopeGid]._species[0]._reactions[iReact];
               // accumulate cross sections by reaction type
               for (unsigned iGroup=0; iGroup<nGroups; ++iGroup)
               {
                  switch (reaction._reactionType)
                  {
                    case NuclearDataReaction::Scatter:
                     xcVec[iGroup].scatter += reaction.getCrossSection(iGroup)/nIsotopes;
                     break;
                    case NuclearDataReaction::Absorption:
                     xcVec[iGroup].absorption += reaction.getCrossSection(iGroup)/nIsotopes;
                     break;
                    case NuclearDataReaction::Fission:
                     xcVec[iGroup].fission += reaction.getCrossSection(iGroup)/nIsotopes;
                     break;
                    case NuclearDataReaction::Undefined:
                     qs_assert(false);
                     break;
                  }   
               }
            }
         }
      }

    FILE* xSec;

    std::string fileName = monteCarlo->_params.simulationParams.crossSectionsOut + ".dat";

    xSec = fopen( fileName.c_str(), "w" );

      // print cross section data
      // first the header
      fprintf(xSec, "#group  energy");
      for (auto mapIter=xcTable.begin(); mapIter!=xcTable.end(); ++mapIter)
      {
         const string& materialName = mapIter->first;
         fprintf(xSec, "  %s_a  %s_f  %s_s", materialName.c_str(), materialName.c_str(), materialName.c_str());
      }
      fprintf(xSec,"\n");
  
      // now the data
      for (unsigned ii=0; ii<nGroups; ++ii)
      {
         fprintf(xSec, "%u  %g", ii, energy[ii]);
         for (auto mapIter=xcTable.begin(); mapIter!=xcTable.end(); ++mapIter)
         {
            fprintf(xSec, "  %g  %g  %g", mapIter->second[ii].absorption, mapIter->second[ii].fission, mapIter->second[ii].scatter);
         }
         fprintf(xSec, "\n");
      }
    fclose( xSec );
   }
}
