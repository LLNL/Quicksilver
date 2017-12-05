#include "MC_Domain.hh"
#include <vector>
#include <map>
#include <utility>
#include <string>

#include <iostream>
using std::cout;
using std::endl;

#include "Globals.hh"
#include "MonteCarlo.hh"
#include "MC_Cell_State.hh"
#include "macros.hh"
#include "MC_RNG_State.hh"
#include "PhysicalConstants.hh"
#include "MeshPartition.hh"
#include "GlobalFccGrid.hh"
#include "DecompositionObject.hh"
#include "MC_Facet_Adjacency.hh"
#include "Parameters.hh"
#include "MaterialDatabase.hh"
#include "MCT.hh"

using std::vector;
using std::make_pair;
using std::map;
using std::abs;
using std::string;

namespace
{
   struct FaceInfo
   {
      MC_Subfacet_Adjacency_Event::Enum _event;
      CellInfo _cellInfo;
      int _nbrIndex;
   };


   int nodeIndirect[24][3] =  { {1, 3, 8}, {3, 7, 8}, {7, 5, 8}, {5, 1, 8},
                                {0, 4, 9}, {4, 6, 9}, {6, 2, 9}, {2, 0, 9},
                                {3, 2,10}, {2, 6,10}, {6, 7,10}, {7, 3,10},
                                {0, 1,11}, {1, 5,11}, {5, 4,11}, {4, 0,11},
                                {4, 5,12}, {5, 7,12}, {7, 6,12}, {6, 4,12},
                                {0, 2,13}, {2, 3,13}, {3, 1,13}, {1, 0,13} };

   int opposingFacet[24] = { 7,  6,  5,  4,  3,  2,  1,  0, 12, 15,
                            14, 13,  8, 11, 10,  9, 20, 23, 22, 21,
                            16, 19, 18, 17};


   void bootstrapNodeMap(map<Long64, int>& nodeIndexMap,
                         const MeshPartition& partition,
                         const GlobalFccGrid& grid);

   void buildCells(qs_vector<MC_Facet_Adjacency_Cell>& cell,
                   BulkStorage<MC_Facet_Adjacency>& facetStore,
                   BulkStorage<int>& pointStore,
                   const map<Long64, int>& nodeIndexMap,
                   const qs_vector<int>& nbrDomain,
                   const MeshPartition& partition,
                   const GlobalFccGrid& grid,
                   const qs_vector<MC_Subfacet_Adjacency_Event::Enum>& boundaryCondition);

   void makeFacet(MC_Facet_Adjacency& facet,
                  const MC_Location& location,
                  int* nodeIndex,
                  const vector<FaceInfo>& faceInfo);

   string findMaterial(const Parameters& params, const MC_Vector& rr);

   qs_vector<MC_Subfacet_Adjacency_Event::Enum> getBoundaryCondition(const Parameters& params);
}


MC_Mesh_Domain::MC_Mesh_Domain(const MeshPartition& meshPartition, const GlobalFccGrid& grid,
                               const DecompositionObject& ddc,
                               const qs_vector<MC_Subfacet_Adjacency_Event::Enum>& boundaryCondition)
: _domainGid(meshPartition.domainGid())
{
   _nbrDomainGid.resize(meshPartition.nbrDomains().size());
   for (unsigned ii=0; ii< _nbrDomainGid.size(); ++ii)
      _nbrDomainGid[ii] = meshPartition.nbrDomains()[ii];



   _nbrRank.reserve(_nbrDomainGid.size(), VAR_MEM);
    _nbrRank.Open();
    for (unsigned ii=0; ii<_nbrDomainGid.size(); ++ii)
      _nbrRank.push_back(ddc.getRank(_nbrDomainGid[ii]));
    _nbrRank.Close();
   map<Long64, int> nodeIndexMap;

   bootstrapNodeMap(nodeIndexMap, meshPartition, grid);


   int totalCells = 0;
   for (auto iter=meshPartition.begin(); iter!=meshPartition.end(); ++iter)
   {
      if (iter->second._domainGid != meshPartition.domainGid())
         continue;
      ++totalCells;
   }
   
   _connectivityFacetStorage.setCapacity(totalCells*24, VAR_MEM);
   _connectivityPointStorage.setCapacity(totalCells*14, VAR_MEM);

   buildCells(_cellConnectivity, _connectivityFacetStorage, _connectivityPointStorage,
              nodeIndexMap, _nbrDomainGid, meshPartition, grid, boundaryCondition);

   _node.resize(nodeIndexMap.size(), VAR_MEM);

   for (auto iter=nodeIndexMap.begin(); iter!=nodeIndexMap.end(); ++iter)
   {
      const Long64& iNodeGid = iter->first;
      const int& iNodeIndex = iter->second;
      _node[iNodeIndex] = grid.nodeCoord(iNodeGid);
   }

   {//limit scope
      // initialize _cellGeometry
      _cellGeometry.resize(_cellConnectivity.size(), VAR_MEM);

      // First, we need to count up the total number of facets of all
      // cells in this domain and initialize the BulkStorage
      // of facets (i.e., MC_General_Plane).  This code is somewhat
      // pedantic since we know all of the cells have 24 facets.
      int totalFacets = 0;
      for (unsigned iCell=0; iCell<_cellConnectivity.size(); ++iCell)
         totalFacets += _cellConnectivity[iCell].num_facets;
      _geomFacetStorage.setCapacity(totalFacets, VAR_MEM);

      // Now initialize all of the facets.
      for (unsigned iCell=0; iCell<_cellConnectivity.size(); ++iCell)
      {
         int nFacets = _cellConnectivity[iCell].num_facets;
         qs_assert(nFacets == 24);
         _cellGeometry[iCell]._facet = _geomFacetStorage.getBlock(nFacets);
         _cellGeometry[iCell]._size = nFacets;
         for (unsigned jFacet=0; jFacet<nFacets; ++jFacet)
         {
            qs_assert(_cellConnectivity[iCell]._facet[jFacet].num_points == 3);
            const int nodeIndex0 = _cellConnectivity[iCell]._facet[jFacet].point[0];
            const int nodeIndex1 = _cellConnectivity[iCell]._facet[jFacet].point[1];
            const int nodeIndex2 = _cellConnectivity[iCell]._facet[jFacet].point[2];
            const MC_Vector& r0 = _node[nodeIndex0];
            const MC_Vector& r1 = _node[nodeIndex1];
            const MC_Vector& r2 = _node[nodeIndex2];
            _cellGeometry[iCell]._facet[jFacet] = MC_General_Plane(r0, r1, r2);
         }
      }
   } // limit scope
   

}

// To emulate data access patterns we're going to put nodes on the
// corners of the hex elements into the node list first.  nodes on
// the the face centers are added after all of the corners.
namespace
{
   void bootstrapNodeMap(map<Long64, int>& nodeIndexMap,
                         const MeshPartition& partition,
                         const GlobalFccGrid& grid)
   {
      map<Long64, int> faceCenters;
      vector<Long64> nodeGid;
      for (auto iter=partition.begin(); iter!=partition.end(); ++iter)
      {
         if (iter->second._domainGid != partition.domainGid())
            continue; // skip remote cells
         const Long64& iCellGid = iter->first;
         grid.getNodeGids(iCellGid, nodeGid);
         for (unsigned ii=0; ii<8; ++ii) //yes, 8.  Only corners.
            nodeIndexMap.insert(make_pair(nodeGid[ii], nodeIndexMap.size()));
         for (unsigned ii=8; ii<14; ++ii) // save face centers for later.
            faceCenters.insert(make_pair(nodeGid[ii], faceCenters.size()));
      }
      for (auto iter=faceCenters.begin(); iter!=faceCenters.end(); ++iter)
         iter->second += nodeIndexMap.size();

      nodeIndexMap.insert(faceCenters.begin(), faceCenters.end());
   }
}

namespace
{
   // Setting up the subfacet info is tricky because some data members
   // of Subfacet_Adjacency don't always apply.
   // * neighbor_index is meaningless for boundary facets and facets that
   //   are adjacent to cells on the same domain.  We choose to set
   //   neighbor_index to -1 in these cases.
   // * adjacent is meaningless for boundary facets.  In these cases we
   //   set adjacent = current.
   void buildCells(qs_vector<MC_Facet_Adjacency_Cell>& cell,
                   BulkStorage<MC_Facet_Adjacency>& facetStore,
                   BulkStorage<int>& pointStore,
                   const map<Long64, int>& nodeIndexMap,
                   const qs_vector<int>& nbrDomain,
                   const MeshPartition& partition,
                   const GlobalFccGrid& grid,
                   const qs_vector<MC_Subfacet_Adjacency_Event::Enum>& boundaryCondition)

   {
      map<int, int> nbrDomainIndex; // nbrDomainIndex[domainGid] = localNbrIndex;

      for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
         nbrDomainIndex[nbrDomain[ii]] = ii;
      // for boundary and non-transit facets
      nbrDomainIndex[partition.domainGid()] = -1;

      vector<Long64> nodeGid;
      vector<Long64> faceNbr;
      if( cell.size() == 0 )
      {
          cell.reserve(partition.size(), VAR_MEM);
      }
      cell.Open();
      for (auto iter=partition.begin(); iter!=partition.end(); ++iter)
      {
         if (iter->second._domainGid != partition.domainGid())
            continue;


         const Long64& iCellGid = iter->first;
         const int& domainIndex = iter->second._domainIndex;
         const int& cellIndex = iter->second._cellIndex;
         const int& foreman = iter->second._foreman;
         qs_assert(domainIndex == partition.domainIndex());
         qs_assert(cellIndex == cell.size());

         cell.push_back(MC_Facet_Adjacency_Cell());
         MC_Facet_Adjacency_Cell& newCell = cell.back();


         newCell._facet = facetStore.getBlock(newCell.num_facets);
         newCell._point = pointStore.getBlock(newCell.num_points);





         
         grid.getNodeGids(iCellGid, nodeGid);
         for (unsigned ii=0; ii<newCell.num_points; ++ii)
         {
            auto here = nodeIndexMap.find(nodeGid[ii]);
            qs_assert(here != nodeIndexMap.end());
            newCell._point[ii] = here->second;
         }

         vector<FaceInfo> faceInfo(6);
         grid.getFaceNbrGids(iCellGid, faceNbr);
         for (unsigned ii=0; ii<6; ++ii)
         {
            auto here = partition.findCell(faceNbr[ii]);
            qs_assert(here != partition.end());
            const CellInfo& jCellInfo = here->second;
            faceInfo[ii]._event = MC_Subfacet_Adjacency_Event::Adjacency_Undefined;
            faceInfo[ii]._cellInfo = jCellInfo;
            faceInfo[ii]._nbrIndex = nbrDomainIndex[jCellInfo._domainGid];
            if (faceNbr[ii] == iCellGid)
               faceInfo[ii]._event = boundaryCondition[ii];
            else
            {
               if (jCellInfo._foreman == foreman)
                  faceInfo[ii]._event = MC_Subfacet_Adjacency_Event::Transit_On_Processor;
               else
                  faceInfo[ii]._event = MC_Subfacet_Adjacency_Event::Transit_Off_Processor;
//                if (jCellInfo._domainIndex != domainIndex && jCellInfo._foreman == foreman)
//                   faceInfo[ii]._event = MC_Subfacet_Adjacency_Event::Transit_On_Processor;
//                if (jCellInfo._foreman != foreman)
//                   faceInfo[ii]._event = MC_Subfacet_Adjacency_Event::Transit_Off_Processor;
            }
         }

         MC_Location location(domainIndex, cellIndex, -1);
         for (unsigned ii=0; ii<newCell.num_facets; ++ii)
         {
            location.facet = ii;
            makeFacet(newCell._facet[ii], location, newCell._point, faceInfo);
         }

      }
      cell.Close();
   }
}

namespace
{
   void makeFacet(MC_Facet_Adjacency& facet,
                  const MC_Location& location,
                  int* nodeIndex,
                  const vector<FaceInfo>& faceInfo)
   {
      const int& facetId = location.facet;
      int faceId = facetId / 4;

      facet.num_points = 3;
      facet.point[0] = nodeIndex[nodeIndirect[facetId][0]];
      facet.point[1] = nodeIndex[nodeIndirect[facetId][1]];
      facet.point[2] = nodeIndex[nodeIndirect[facetId][2]];
      facet.subfacet.event    = faceInfo[faceId]._event;
      facet.subfacet.current  = location;
      facet.subfacet.adjacent.domain = faceInfo[faceId]._cellInfo._domainIndex;
      facet.subfacet.adjacent.cell   = faceInfo[faceId]._cellInfo._cellIndex;
      facet.subfacet.adjacent.facet = opposingFacet[facetId];
      facet.subfacet.neighbor_index = faceInfo[faceId]._nbrIndex;
      facet.subfacet.neighbor_global_domain = faceInfo[faceId]._cellInfo._domainGid;
      facet.subfacet.neighbor_foreman = faceInfo[faceId]._cellInfo._foreman;

      // handle special case
      if (facet.subfacet.event == MC_Subfacet_Adjacency_Event::Boundary_Reflection ||
          facet.subfacet.event == MC_Subfacet_Adjacency_Event::Boundary_Escape )
         facet.subfacet.adjacent.facet = facet.subfacet.current.facet;
   }
}

MC_Vector findCellCenter(const MC_Facet_Adjacency_Cell& cell,
                         const qs_vector<MC_Vector>& node)
{
   // find center of cell
   MC_Vector cellCenter(0., 0., 0.);
   for ( int iter=0; iter < cell.num_points; iter++)
      cellCenter += node[cell._point[iter]];
   cellCenter /= cell.num_points;
   return cellCenter;
}

   
// This is messed up.  Why doesn't either the cell or the mesh have a
// member function to compute the volume?
double cellVolume(const MC_Facet_Adjacency_Cell& cell,
                  const qs_vector<MC_Vector>& node)
{
   // find center of cell
   MC_Vector cellCenter(0., 0., 0.);
   for ( int iter=0; iter < cell.num_points; iter++)
      cellCenter += node[cell._point[iter]];
   cellCenter /= cell.num_points;

   double volume = 0;
   for (unsigned iFacet=0; iFacet<cell.num_facets; ++iFacet)
   {
      const int* facetCorner = cell._facet[iFacet].point;
      MC_Vector aa = node[facetCorner[0]] - cellCenter;
      MC_Vector bb = node[facetCorner[1]] - cellCenter;
      MC_Vector cc = node[facetCorner[2]] - cellCenter;

      volume += abs(aa.Dot(bb.Cross(cc)));
   }
   volume /= 6.0;
   return volume;
}


MC_Domain::MC_Domain(const MeshPartition& meshPartition, const GlobalFccGrid& grid,
                     const DecompositionObject& ddc, const Parameters& params,
                     const MaterialDatabase& materialDatabase, int numEnergyGroups)
: domainIndex(meshPartition.domainIndex()),
  global_domain(meshPartition.domainGid()),
  mesh(meshPartition, grid, ddc, getBoundaryCondition(params))
{
   cell_state.resize(mesh._cellGeometry.size(), VAR_MEM);
   _cachedCrossSectionStorage.setCapacity(cell_state.size() * numEnergyGroups, VAR_MEM);

   
   for (unsigned ii=0; ii<cell_state.size(); ++ii)
   {
      cell_state[ii]._volume = cellVolume(mesh._cellConnectivity[ii],
                                         mesh._node);

      MC_Vector point = MCT_Cell_Position_3D_G(*this, ii);

      std::string matName = findMaterial(params, point);
      cell_state[ii]._material = materialDatabase.findMaterial(matName);

      cell_state[ii]._total = _cachedCrossSectionStorage.getBlock(numEnergyGroups);
      for (unsigned jj=0; jj<numEnergyGroups; ++jj)
         cell_state[ii]._total[jj] = 0.;
      
      int numIsos = static_cast<int>(materialDatabase._mat[cell_state[ii]._material]._iso.size());
      //  The cellNumberDensity scales the crossSections so we choose to
      //  set this density to 1.0 so that the totalCrossSection will be
      //  as requested by the user.
      cell_state[ii]._cellNumberDensity = 1.0;

      MC_Vector cellCenter = findCellCenter(mesh._cellConnectivity[ii], mesh._node);
      cell_state[ii]._id = grid.whichCell(cellCenter) * UINT64_C(0x0100000000);
      cell_state[ii]._sourceTally = 0;
   }

}

void MC_Domain::clearCrossSectionCache(int numEnergyGroups)
{
   for (unsigned ii=0; ii<cell_state.size(); ++ii)
      for (unsigned jj=0; jj<numEnergyGroups; ++jj)
         cell_state[ii]._total[jj] = 0.;
}

namespace
{
   // Returns true if the specified coordinate in inside the specified
   // geometry.  False otherwise
   bool isInside(const GeometryParameters& geom, const MC_Vector& rr)
   {
      bool inside = false;
      switch (geom.shape)
      {
        case GeometryParameters::BRICK:
         {
            if ( (rr.x >= geom.xMin && rr.x <= geom.xMax) &&
                 (rr.y >= geom.yMin && rr.y <= geom.yMax) &&
                 (rr.z >= geom.zMin && rr.z <= geom.zMax) )
               inside = true;
         }
         break;
        case GeometryParameters::SPHERE:
         {
            MC_Vector center(geom.xCenter, geom.yCenter, geom.zCenter);
            if ( (rr-center).Length() <= geom.radius)
               inside = true;
         }

         break;
        default:
         qs_assert(false);
      }
      return inside;
   }
}


// Returns the name of the material present at coordinate rr.  If
// multiple materials overlap return the last material found.
namespace
{
   string findMaterial(const Parameters& params, const MC_Vector& rr)
   {
      string materialName;
      for (unsigned ii=0; ii< params.geometryParams.size(); ++ii)
         if (isInside(params.geometryParams[ii], rr))
            materialName = params.geometryParams[ii].materialName;

      qs_assert(materialName.size() > 0);
      return materialName;
   }
}


namespace
{
   qs_vector<MC_Subfacet_Adjacency_Event::Enum> getBoundaryCondition(const Parameters& params)
   {
      qs_vector<MC_Subfacet_Adjacency_Event::Enum> bc(6);
      if (params.simulationParams.boundaryCondition == "reflect")
         bc = qs_vector<MC_Subfacet_Adjacency_Event::Enum>(6, MC_Subfacet_Adjacency_Event::Boundary_Reflection);
      else if (params.simulationParams.boundaryCondition == "escape")
         bc = qs_vector<MC_Subfacet_Adjacency_Event::Enum>(6, MC_Subfacet_Adjacency_Event::Boundary_Escape);
      else if  (params.simulationParams.boundaryCondition == "octant")
         for (unsigned ii=0; ii<6; ++ii)
         {
            if (ii % 2 == 0) bc[ii] = MC_Subfacet_Adjacency_Event::Boundary_Escape;
            if (ii % 2 == 1) bc[ii] = MC_Subfacet_Adjacency_Event::Boundary_Reflection;
         }
      else
         qs_assert(false);
      return bc;
   }
}

