#ifndef DECOMPOSITION_OBJECT_HH
#define DECOMPOSITION_OBJECT_HH

#include <vector>

class DecompositionObject
{
 public:
   DecompositionObject(int myRank, int nRanks, int nDomainsPerRank, int mode);

   int getRank(int domainGid) const {return _rank[domainGid];}
   int getIndex(int domainGid) const {return _index[domainGid];}
   const std::vector<int>& getAssignedDomainGids() const {return _assignedGids;}

 private:
   std::vector<int> _assignedGids;
   std::vector<int> _rank;  // rank for given gid
   std::vector<int> _index; // index for given gid
};

#endif
