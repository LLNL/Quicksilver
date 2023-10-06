#include "MpiCommObject.hh"
#include "qs_assert.hh"
#include <algorithm>
#include "MeshPartition.hh"

using std::set;
using std::sort;
using std::vector;

namespace
{
   MPI_Datatype cellInfoMpiType();
   MPI_Datatype facetPairMpiType();
   bool compareDomainGid2(const FacetPair& a, const FacetPair& b);
}



MpiCommObject::MpiCommObject(const MPI_Comm& comm, const DecompositionObject& ddc)
:_comm(comm),
 _ddc(ddc)
{
}

void MpiCommObject::exchange(MeshPartition::MapType& cellInfoMap,
                             const vector<int>& nbrDomain,
                             vector<set<Long64> > sendSet,
                             vector<set<Long64> > recvSet)

{
   int nRanks, myRank;
   mpiComm_rank(MPI_COMM_WORLD, &myRank);
   mpiComm_size(MPI_COMM_WORLD, &nRanks);
   mpiBarrier(MPI_COMM_WORLD);

   qs_assert(sendSet.size() == nbrDomain.size());
   qs_assert(recvSet.size() == nbrDomain.size());

   const int xTag = 13;
   int nRequest = 2*nbrDomain.size();
   int nItemsToSend = 0;
   int nItemsToRecv = 0;
   for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
   {
      nItemsToSend += sendSet[ii].size();
      nItemsToRecv += recvSet[ii].size();
   }
   CellInfo *sendBuf = new CellInfo[nItemsToSend];
   CellInfo *recvBuf = new CellInfo[nItemsToRecv];
   MPI_Request *request = new MPI_Request[nRequest];

   MPI_Datatype datatype = cellInfoMpiType();

   // post recvs
   {
      CellInfo* recvPtr = recvBuf;
      for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
      {
         const int sourceRank = _ddc.getRank(nbrDomain[ii]);
         const int nRecv = recvSet[ii].size();
         mpiIrecv(recvPtr, nRecv, datatype, sourceRank, xTag, _comm, request+2*ii);
         recvPtr += nRecv;
      }
   }

   // assemble buffers and send
   {
      int count = 0;
      CellInfo* sendPtr = sendBuf;
      for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
      {
         for (auto iter=sendSet[ii].begin(); iter!=sendSet[ii].end(); ++iter)
            sendBuf[count++] = cellInfoMap[*iter];
         const int targetRank = _ddc.getRank(nbrDomain[ii]);
         const int nSend = sendSet[ii].size();
         mpiIsend(sendPtr, nSend, datatype, targetRank, xTag, _comm, request+(2*ii)+1);
         sendPtr += nSend;
      }
   }

   // wait for comm
   mpiWaitall(nRequest, request, MPI_STATUSES_IGNORE);

   // process recvd data
   {
      int count = 0;
      for (unsigned ii=0; ii<nbrDomain.size(); ++ii)
      {
         for (auto iter=recvSet[ii].begin(); iter!=recvSet[ii].end(); ++iter)
            cellInfoMap[*iter] = recvBuf[count++];
      }
   }

   delete [] request;
   delete [] sendBuf;
   delete [] recvBuf;
}

void MpiCommObject::exchange(vector<FacetPair> sendBuf,
                             vector<FacetPair>& recvBuf)
{
   sort(&sendBuf[0], &sendBuf[sendBuf.size()-1], compareDomainGid2);
   vector<int> sendOffset(sendBuf.size());
   sendOffset.push_back(0);
   for (unsigned ii=1; ii<sendBuf.size(); ++ii)
      if (sendBuf[ii-1]._domainGid2 != sendBuf[ii]._domainGid2)
         sendOffset.push_back(ii);
   sendOffset.push_back(sendBuf.size());

   int nSend = sendOffset.size()-1;
   int nRecv = nSend;
   int nRequest = nSend + nRecv;
   MPI_Request *request = new MPI_Request[nRequest];
   MPI_Datatype datatype = facetPairMpiType();
   MC_Location dummy(-1, -1, -1);
   recvBuf.resize(sendBuf.size());
   const int xTag = 14;

   {
      for (int ii=0; ii<nRecv; ++ii)
      {
         FacetPair* recvPtr = &(recvBuf[0]) + sendOffset[ii];
         int nItems = sendOffset[ii+1] - sendOffset[ii];
         int sourceRank = _ddc.getRank(sendBuf[sendOffset[ii]]._domainGid2);
         mpiIrecv(recvPtr, nItems, datatype, sourceRank, xTag, _comm, request+2*ii);
      }
   }
   {
      for (int ii=0; ii<nSend; ++ii)
      {
         FacetPair* sendPtr = &(sendBuf[0]) + sendOffset[ii];
         int nItems = sendOffset[ii+1] - sendOffset[ii];
         int targetRank = _ddc.getRank(sendBuf[sendOffset[ii]]._domainGid2);
         mpiIsend(sendPtr, nItems, datatype, targetRank, xTag, _comm, request+(2*ii)+1);
      }
   }

   mpiWaitall(nRequest, request, MPI_STATUSES_IGNORE);

   delete [] request;
}





namespace
{
   MPI_Datatype cellInfoMpiType()
   {
      static MPI_Datatype datatype;
      static bool inititalized = false;
      if (! inititalized )
      {
         mpiType_contiguous(4, MPI_INT, &datatype);
         mpiType_commit(&datatype);
         inititalized = true;
      }
      return datatype;
   }
}

namespace
{
   MPI_Datatype facetPairMpiType()
   {
      static MPI_Datatype datatype;
      static bool inititalized = false;
      if (! inititalized )
      {
         mpiType_contiguous(8, MPI_INT, &datatype);
         mpiType_commit(&datatype);
         inititalized = true;
      }
      return datatype;
   }
}

namespace
{
   // We only care that items with the same domainGid2 are grouped.
   // Ordering among items with the same domainGid2 is irrelevant.
   bool compareDomainGid2(const FacetPair& a, const FacetPair& b)
   {
      return a._domainGid2 < b._domainGid2;
   }
}
