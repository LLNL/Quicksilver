#ifndef SHARED_MEMORY_COMM_OBJECT_HH
#define SHARED_MEMORY_COMM_OBJECT_HH

#include "CommObject.hh"

#include <set>

#include <vector>
#include "MeshPartition.hh"
#include "Long64.hh"

class SharedMemoryCommObject : public CommObject
{
 public:
   SharedMemoryCommObject(std::vector<MeshPartition>& meshPartition);

   void exchange(MeshPartition::MapType& cellInfo,
                 const std::vector<int>& nbrDomain,
                 std::vector<std::set<Long64> > sendSet,
                 std::vector<std::set<Long64> > recvSet);

   void exchange(std::vector<FacetPair> sendBuf,
                 std::vector<FacetPair>& recvBuf);


 private:
   std::vector<MeshPartition>& _partitions;
   std::vector<int> _gidToIndex;
};

#endif
