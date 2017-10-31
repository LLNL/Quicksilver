#include "GlobalFccGrid.hh"
#include <algorithm>
#include <cstdio>
#include "MC_Vector.hh"
#include "Tuple.hh"

using std::vector;
using std::min;
using std::max;

namespace
{
   const vector<Tuple>& getFaceTupleOffset();
}


GlobalFccGrid::GlobalFccGrid(int nx, int ny, int nz,
                             double lx, double ly, double lz)
: _nx(nx), _ny(ny), _nz(nz),
  _lx(lx), _ly(ly), _lz(lz),
  _cellTupleToIndex(nx, ny, nz),
  _cellIndexToTuple(nx, ny, nz),
  _nodeTupleToIndex(nx+1, ny+1, nz+1, 4),
  _nodeIndexToTuple(nx+1, ny+1, nz+1, 4)
{
   _dx = _lx/_nx;
   _dy = _ly/_ny;
   _dz = _lz/_nz;
}

Long64 GlobalFccGrid::whichCell(const MC_Vector& r) const
{
   int ix = r.x/_dx;
   int iy = r.y/_dy;
   int iz = r.z/_dz;
   return _cellTupleToIndex(ix, iy, iz);
}


MC_Vector GlobalFccGrid::cellCenter(Long64 iCell) const
{
   Tuple tt = _cellIndexToTuple(iCell);
   MC_Vector r = nodeCoord(Tuple4(tt.x(), tt.y(), tt.z(), 0) );
   r += MC_Vector(_dx/2., _dy/2., _dz/2.);
   return r;
}

const vector<Tuple4>& GlobalFccGrid::cornerTupleOffsets() const
{
   static vector<Tuple4> offset;
   if (offset.size() == 0)
   {
      offset.reserve(14);
      offset.push_back(Tuple4(0, 0, 0, 0)); // 0
      offset.push_back(Tuple4(1, 0, 0, 0)); // 1
      offset.push_back(Tuple4(0, 1, 0, 0)); // 2
      offset.push_back(Tuple4(1, 1, 0, 0)); // 3
      offset.push_back(Tuple4(0, 0, 1, 0)); // 4
      offset.push_back(Tuple4(1, 0, 1, 0)); // 5
      offset.push_back(Tuple4(0, 1, 1, 0)); // 6
      offset.push_back(Tuple4(1, 1, 1, 0)); // 7
      offset.push_back(Tuple4(1, 0, 0, 1)); // 8
      offset.push_back(Tuple4(0, 0, 0, 1)); // 9
      offset.push_back(Tuple4(0, 1, 0, 2)); // 10
      offset.push_back(Tuple4(0, 0, 0, 2)); // 11
      offset.push_back(Tuple4(0, 0, 1, 3)); // 12
      offset.push_back(Tuple4(0, 0, 0, 3)); // 13
   }
   return offset;
}

void GlobalFccGrid::getNodeGids(Long64 cellGid, vector<Long64>& nodeGid) const
{
    if( nodeGid.size() == 0 )
    {
        nodeGid.resize(14);
    }

   Tuple tt = _cellIndexToTuple(cellGid);
   Tuple4 baseNodeTuple = Tuple4(tt.x(), tt.y(), tt.z(), 0);
   const vector<Tuple4>& cornerTupleOffset = cornerTupleOffsets();
   for (unsigned ii=0; ii<14; ++ii)
      nodeGid[ii] = _nodeTupleToIndex(baseNodeTuple + cornerTupleOffset[ii]);
}

// for faces on the outer surface of the global grid, the returned cell
// gid will be the same as the input cellGid
void GlobalFccGrid::getFaceNbrGids(Long64 cellGid, vector<Long64>& nbrCellGid) const
{
    if( nbrCellGid.size() == 0 )
    {
        nbrCellGid.resize(6);
    }

    Tuple cellTuple = _cellIndexToTuple(cellGid);
    const vector<Tuple>& faceTupleOffset = getFaceTupleOffset();

   for (unsigned ii=0; ii<6; ++ii)
   {
      Tuple faceNbr = cellTuple + faceTupleOffset[ii];
      snapTuple(faceNbr);
      nbrCellGid[ii] = _cellTupleToIndex(faceNbr);
   }
}


MC_Vector GlobalFccGrid::nodeCoord(Long64 index) const
{
   return nodeCoord(_nodeIndexToTuple(index));
}

MC_Vector GlobalFccGrid::nodeCoord(const Tuple4& tt) const
{
   vector<MC_Vector> basisOffset;
   basisOffset.reserve(4);
   if (basisOffset.size() == 0)
   {
      basisOffset.push_back(MC_Vector(0.,      0.,      0.     ));
      basisOffset.push_back(MC_Vector(0.,      _dy/2.0, _dz/2.0));
      basisOffset.push_back(MC_Vector(_dx/2.0, 0.,      _dz/2.0));
      basisOffset.push_back(MC_Vector(_dx/2.0, _dy/2.0, 0.     ));
   }

   double rx = tt.x()*_dx;
   double ry = tt.y()*_dy;
   double rz = tt.z()*_dz;

   MC_Vector rr = MC_Vector(rx, ry, rz) + basisOffset[tt.b()];

   return rr;
}

void GlobalFccGrid::snapTuple(Tuple& tt) const
{
   tt.x() = min(max(0, tt.x()), _nx-1);
   tt.y() = min(max(0, tt.y()), _ny-1);
   tt.z() = min(max(0, tt.z()), _nz-1);
}

namespace
{
   const vector<Tuple>& getFaceTupleOffset()
   {
      static vector<Tuple> faceTupleOffset;

      if (faceTupleOffset.size() == 0)
      {
         faceTupleOffset.reserve(6);
         faceTupleOffset.push_back( Tuple( 1,  0,  0) );
         faceTupleOffset.push_back( Tuple(-1,  0,  0) );
         faceTupleOffset.push_back( Tuple( 0,  1,  0) );
         faceTupleOffset.push_back( Tuple( 0, -1,  0) );
         faceTupleOffset.push_back( Tuple( 0,  0,  1) );
         faceTupleOffset.push_back( Tuple( 0,  0, -1) );
      }

      return faceTupleOffset;
   }
}

