#include "DecompositionObject.hh"
#include <stdlib.h>
#include <algorithm>
#include <utility>
#include <set>
#include "qs_assert.hh"

using std::vector;
using std::find;
using std::swap;
using std::pair;
using std::set;
using std::make_pair;
namespace
{
   void fisherYates(vector<int>& vv)
   {
      int nItems = vv.size();
      for (unsigned ii=0; ii<nItems-1; ++ii)
      {
         int jj = (drand48() * (nItems - ii)) + ii;
         swap(vv[ii], vv[jj]);
      }
   }
}

DecompositionObject::DecompositionObject(
   int myRank, int nRanks, int nDomainsPerRank, int mode)
{
   qs_assert(mode == 0 || mode == 1);

   int nDomains = nRanks*nDomainsPerRank;
   _rank.resize(nDomains);
   _index.resize(nDomains);

   //assign domains to ranks
   for (unsigned ii=0; ii<nDomains; ++ii)
      _rank[ii] = ii/nDomainsPerRank;

   if (mode == 1)
      fisherYates(_rank);



   // set up the local domain indices for all ranks
   for (unsigned iRank=0; iRank<nRanks; ++iRank)
   {
      vector<int> localGid;
      localGid.reserve(nDomainsPerRank);
      for (unsigned jGid=0; jGid<nDomains; ++jGid)
         if (_rank[jGid] == iRank)
            localGid.push_back(jGid);
      qs_assert(localGid.size() == nDomainsPerRank);

      if (mode == 1)
         fisherYates(localGid);


      for (unsigned ii=0; ii<localGid.size(); ++ii)
         _index[localGid[ii]] = ii;

      if (iRank == myRank)
         _assignedGids = localGid;
   }

   // tests
   for (unsigned ii=0; ii<nDomainsPerRank; ++ii)
      qs_assert(_rank[_assignedGids[ii]] == myRank);

   set<pair<int, int> > tmp;
   for (unsigned ii=0; ii<nDomains; ++ii)
   {
      qs_assert(_rank[ii] < nRanks);
      qs_assert(_index[ii] <nDomainsPerRank);
      tmp.insert(make_pair(_rank[ii], _index[ii]));
   }
   qs_assert(tmp.size() == nDomains);
}

