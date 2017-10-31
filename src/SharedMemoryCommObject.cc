#include "SharedMemoryCommObject.hh"
#include "qs_assert.hh"
#include "MeshPartition.hh"

using std::set;
using std::vector;


SharedMemoryCommObject::SharedMemoryCommObject(vector<MeshPartition>& meshPartition)
:_partitions(meshPartition)
{
   _gidToIndex.resize(_partitions.size());
   for (unsigned ii=0; ii<_partitions.size(); ++ii)
   {
      int gid = _partitions[ii].domainGid();
      qs_assert(gid < _partitions.size());
      _gidToIndex[gid] = ii;
   }

}

void SharedMemoryCommObject::exchange(MeshPartition::MapType& cellInfoMap,
                                      const vector<int>& nbrDomain,
                                      vector<set<Long64> > sendSet,
                                      vector<set<Long64> > recvSet)

{
   for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
   {
      const int& targetDomainGid = nbrDomain[ii];
      MeshPartition& targetPartition = _partitions[_gidToIndex[targetDomainGid]];
      qs_assert(targetPartition.domainGid() == targetDomainGid);

      for (auto iter=sendSet[ii].begin(); iter!=sendSet[ii].end(); ++iter)
      {
         const CellInfo& cellToSend = cellInfoMap[*iter];
         qs_assert(cellToSend._domainIndex >= 0);
         qs_assert(cellToSend._cellIndex >= 0);
         targetPartition.addCell(*iter, cellToSend);
      }
   }
}

void SharedMemoryCommObject::exchange(vector<FacetPair> sendBuf,
                                      vector<FacetPair>& recvBuf)
{
   // This type of exchange should never occur in SharedMemory spaces.
   qs_assert(false);
}


