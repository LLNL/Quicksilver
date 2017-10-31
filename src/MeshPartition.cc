#include "MeshPartition.hh"

#include <queue>
#include <set>
#include <utility>

#include "qs_assert.hh"
#include "GlobalFccGrid.hh"
#include "MC_Vector.hh"
#include "GridAssignmentObject.hh"
#include "CommObject.hh"

using std::make_pair;
using std::map;
using std::vector;
using std::queue;
using std::set;

namespace
{
   void assignCellsToDomain(MeshPartition::MapType& domainMap,
                            vector<int>& nbrDomains,
                            int myDomainGid,
                            const vector<MC_Vector>& domainCenter,
                            const GlobalFccGrid& grid);

   void buildCellIndexMap(MeshPartition::MapType& cellIndexMap,
                          int myDomainGid,
                          int myForeman,
                          int myDomainIndex,
                          const vector<int>& nbrDomains,
                          const GlobalFccGrid& grid,
                          CommObject* comm);

   void addNbrsToFlood(Long64 iCell,
                       const GlobalFccGrid& grid,
                       queue<Long64>& floodQueue,
                       set<Long64>& wetCells);

}

MeshPartition::MeshPartition(int domainGid, int domainIndex, int foreman)
: _domainGid(domainGid),
  _domainIndex(domainIndex),
  _foreman(foreman)
{
}

// The flooding algorithm used to identify all of the cells in the given
// domain can fail for "difficult" arrangements of the centers.  It is
// possible to create voronoi diagrams where the set of cells assigned
// to a domain do not mutually connect through even a common corner.  I
// assume this is due to very "spikey" voronoi cells.  When this
// happens, the flood will fail to find cells that are part of the domain.
//
// Rather than fix this, we plan to use domain center arrangements that
// don't create such issues.
//
// Possible fixes include increasing the width of the flood stencil
// (though it migh become necessary to prune cells that aren't connected
// to any cell in the target domain to avoid downstream problems), but
// this isn't a true fix as a sufficiently spikey voronoi domain can
// still trigger a problem.
//
// A completely proper fix would involve having each rank calculate
// assignments for a portion of the grid, then communicating the
// calculated assiignments.  This is considerably more complicated, so
// we're staying with the simple method for now.
void MeshPartition::buildMeshPartition(const GlobalFccGrid& grid,
                                       const vector<MC_Vector> centers,
                                       CommObject* comm)
{
   assignCellsToDomain(_cellInfoMap, _nbrDomains,
                       _domainGid,  centers, grid);

   buildCellIndexMap(_cellInfoMap,
                     _domainGid, _foreman, _domainIndex, _nbrDomains, grid, comm);
}


namespace
{
   void assignCellsToDomain(MeshPartition::MapType& assignedDomainMap,
                            vector<int>& nbrDomains,
                            int myDomainGid,
                            const vector<MC_Vector>& domainCenter,
                            const GlobalFccGrid& grid)
   {
      GridAssignmentObject assigner(domainCenter);
      queue<Long64> floodQueue;
      set<Long64>   wetCells;
      set<int>      remoteDomainSet;

      Long64 root = grid.whichCell(domainCenter[myDomainGid]);
      Tuple tmp = grid.cellIndexToTuple(root);

      floodQueue.push(root);
      wetCells.insert(root);
      addNbrsToFlood(root, grid, floodQueue, wetCells);

      while (floodQueue.size() > 0)
      {
         Long64 iCell = floodQueue.front();
         floodQueue.pop();
         MC_Vector rr = grid.cellCenter(iCell);
         int domain = assigner.nearestCenter(rr);
         auto here = assignedDomainMap.insert(
            make_pair(iCell, CellInfo(domain, -2, -2, -myDomainGid)));
         if (here.second == false) // must have been added by nbr
            qs_assert(here.first->second._domainGid == domain);
         if (domain == myDomainGid)
            addNbrsToFlood(iCell, grid, floodQueue, wetCells);
         else
            remoteDomainSet.insert(domain);
      }

      int ind = 0;
      nbrDomains.resize(remoteDomainSet.size());
      for( auto iter = remoteDomainSet.begin(); iter != remoteDomainSet.end(); ++iter )
      {
          nbrDomains[ind++] = *iter;
      }
   }
}

namespace
{
   void buildCellIndexMap(MeshPartition::MapType& cellInfoMap,
                          int myDomainGid,
                          int myForeman,
                          int myDomainIndex,
                          const vector<int>& nbrDomain,
                          const GlobalFccGrid& grid,
                          CommObject* comm)
   {
      int nLocalCells = 0;
      vector<set<Long64> > sendSet(nbrDomain.size());
      vector<set<Long64> > recvSet(nbrDomain.size());

      map<int, int> remoteDomainMap;
      for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
         remoteDomainMap[nbrDomain[ii]] = ii;

      for (auto iter=cellInfoMap.begin(); iter!=cellInfoMap.end(); ++iter)
      {
         const Long64& iCellGid = iter->first;
         const int&    iDomainGid = iter->second._domainGid;
         CellInfo& iCellInfo = iter->second;

         if (iDomainGid == myDomainGid) // local cell
         {
            iCellInfo._cellIndex = nLocalCells++;
            iCellInfo._domainIndex = myDomainIndex;
            iCellInfo._foreman = myForeman;
         }
         else // iCell is a remote cell
         {
            const int& remoteNbrIndex = remoteDomainMap[iDomainGid];
            Tuple iTuple = grid.cellIndexToTuple(iCellGid);
            vector<Long64> faceNbr;
            grid.getFaceNbrGids(iCellGid, faceNbr);
            for (unsigned ii=0; ii<6; ++ii)
            {
               Long64 jCellGid = faceNbr[ii];
               auto jCellIter = cellInfoMap.find(jCellGid);
               if (jCellIter == cellInfoMap.end() ||
                   jCellIter->second._domainGid != myDomainGid)
                  continue;
               // jCell is local
               const int& jDomainGid = jCellIter->second._domainGid;
               qs_assert(jDomainGid == myDomainGid);
               sendSet[remoteNbrIndex].insert(jCellGid);
               recvSet[remoteNbrIndex].insert(iCellGid);
            }
         }
      } // loop over cells in cellInfoMap

      comm->exchange(cellInfoMap, nbrDomain, sendSet, recvSet);
   }
}


namespace
{
   void addNbrsToFlood(Long64 iCell,
                       const GlobalFccGrid& grid,
                       queue<Long64>& floodQueue,
                       set<Long64>& wetCells)
   {
      Tuple tt = grid.cellIndexToTuple(iCell);
      for (int ii=-1; ii<2; ++ii)
         for (int jj=-1; jj<2; ++jj)
            for (int kk=-1; kk<2; ++kk)
            {
               if (ii==0 && jj==0 && kk==0) continue;
               Tuple nbrTuple = tt + Tuple(ii, jj, kk);
               grid.snapTuple(nbrTuple);
               Long64 nbrIndex = grid.cellTupleToIndex(nbrTuple);
               if (wetCells.find(nbrIndex) == wetCells.end())
               {
                  floodQueue.push(nbrIndex);
                  wetCells.insert(nbrIndex);
               }
            }
   }
}
