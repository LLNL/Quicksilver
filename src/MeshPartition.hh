#ifndef MESH_PARTITION_HH
#define MESH_PARTITION_HH

#include <map>
#include <vector>
#include "Long64.hh"

class MC_Vector;
class GlobalFccGrid;
class CommObject;

struct CellInfo
{
   CellInfo()
   : _domainGid(-1), _foreman(-1), _domainIndex(-1), _cellIndex(-1){}
   CellInfo(int domainGid, int foreman, int domainIndex, int cellIndex)
   :_domainGid(domainGid), _foreman(foreman), _domainIndex(domainIndex), _cellIndex(cellIndex){}

   int _domainGid;
   int _foreman;
   int _domainIndex;
   int _cellIndex;
};


class MeshPartition
{
 public:

   typedef std::map<Long64, CellInfo> MapType;

   MeshPartition(){};
   MeshPartition(int domainGid, int domainIndex, int foreman);

   const int& domainGid() const {return _domainGid;}
   const int& domainIndex() const {return _domainIndex;}
   const int& foreman() const {return _foreman;}
   const std::vector<int>& nbrDomains() const {return _nbrDomains;}

   const CellInfo& getCell(Long64 cellGid){return _cellInfoMap[cellGid];}
   MapType::const_iterator findCell(Long64 cellGid) const
   {return _cellInfoMap.find(cellGid);}

   MapType::const_iterator begin() const {return _cellInfoMap.begin();}
   MapType::const_iterator end()   const {return _cellInfoMap.end();}
   int size() const { return _cellInfoMap.size(); }


   void addCell(Long64 cellGid, const CellInfo& cellInfo){_cellInfoMap[cellGid] = cellInfo;}

   // Warning: parition will contain some remote cells with invalid
   // domainIndex and cellIndex.  These cells are not connected by a
   // face to any local cell so they are harmless.  We could write code
   // to delete them if having them around is a problem.
   void buildMeshPartition(const GlobalFccGrid& grid,
                           const std::vector<MC_Vector> centers,
                           CommObject* comm);

 private:
   int _domainGid;   //!< gid of this domain
   int _domainIndex; //!< local index of this domain
   int _foreman;
   MapType _cellInfoMap;
   std::vector<int> _nbrDomains; //<! gids of nbr domains
};

#endif
