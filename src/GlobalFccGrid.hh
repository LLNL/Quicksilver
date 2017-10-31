#ifndef GLOBAL_FCC_GRID_HH
#define GLOBAL_FCC_GRID_HH

#include <vector>
#include "TupleToIndex.hh"
#include "IndexToTuple.hh"
#include "Tuple4ToIndex.hh"
#include "IndexToTuple4.hh"


class MC_Vector;

class GlobalFccGrid
{
 public:
   GlobalFccGrid(int nx, int ny, int nz,
                 double lx, double ly, double lz);

   double lx() const {return _lx;}
   double ly() const {return _ly;}
   double lz() const {return _lz;}
   double nx() const {return _nx;}
   double ny() const {return _ny;}
   double nz() const {return _nz;}

   Long64 whichCell(const MC_Vector& r) const;

   MC_Vector cellCenter(Long64 iCell) const;
   Tuple  cellIndexToTuple(Long64 iCell)    const {return _cellIndexToTuple(iCell);}
   Long64 cellTupleToIndex(const Tuple& tt) const {return _cellTupleToIndex(tt);}

   Long64 nodeIndex(const Tuple4& tt) const {return _nodeTupleToIndex(tt);}

   const std::vector<Tuple4>& cornerTupleOffsets() const;
   void getNodeGids(Long64 cellGid, std::vector<Long64>& nodeGid) const;
   void getFaceNbrGids(Long64 cellGid, std::vector<Long64>& nbrCellGid) const;

   MC_Vector nodeCoord(Long64 index) const;
   MC_Vector nodeCoord(const Tuple4& tt) const;

   // We should get rid of snap tuple and provide a way to get the
   // indices of face nbrs.
   void snapTuple(Tuple& tt) const;

 private:
   int _nx, _ny, _nz;     // number of cells (i.e., elements)
   double _lx, _ly, _lz;  // size of problem space (in cm)
   double _dx, _dy, _dz;  // size of a mesh cell (in cm)

   TupleToIndex  _cellTupleToIndex;
   IndexToTuple  _cellIndexToTuple;
   Tuple4ToIndex _nodeTupleToIndex;
   IndexToTuple4 _nodeIndexToTuple;
};

#endif
