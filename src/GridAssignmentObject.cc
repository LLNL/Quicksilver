#include "GridAssignmentObject.hh"
#include <stdlib.h>
#include <algorithm>
#include <cstdio>
#include "qs_assert.hh"

#define DIFFSQ(a,b) (MC_Vector(a-b).Dot(MC_Vector(a-b)))

using std::vector;
using std::queue;
using std::min;
using std::max;
using std::floor;

/** The present implementation of GridAssignmentObject is judged to be
 *  sufficiently fast to meet the needs of initial assignment of
 *  particles to domains.  The best way to speed up the code would be to
 *  more strictly limit the number of cells that are flooded by
 *  implementing an improved distance calculation in minDist2.
 *
 *  The next best optimization possibility probably involves reducing
 *  the number of indexToTuple and tupleToIndex calculations (probably
 *  at the expense of a higher memory footprint.
*/


GridAssignmentObject::GridAssignmentObject(const vector<MC_Vector>& centers)
: _centers(centers)
{
   // This sets the length scale of the grid cells.  The value 5 is
   // pretty arbitrary.  It could just as easily be 1 or 10.  If
   // necessary it could be made a parameter that is wired out to the
   // input deck.
   int centersPerCell = 5;

   MC_Vector minCoord = _centers[0];
   MC_Vector maxCoord = _centers[0];
   for (int ii=1; ii<_centers.size(); ++ii)
   {
      const MC_Vector& iCenter = _centers[ii];
      minCoord.x = min(minCoord.x, iCenter.x);
      minCoord.y = min(minCoord.y, iCenter.y);
      minCoord.z = min(minCoord.z, iCenter.z);
      maxCoord.x = max(maxCoord.x, iCenter.x);
      maxCoord.y = max(maxCoord.y, iCenter.y);
      maxCoord.z = max(maxCoord.z, iCenter.z);
   }
   _corner = minCoord;

   // It is possible that all of the centers lie on the x-, y-, or
   // z-plane.  If so, arbitrarily set the length in that direction to
   // 1.
   double lx = max(1., (maxCoord.x - minCoord.x));
   double ly = max(1., (maxCoord.y - minCoord.y));
   double lz = max(1., (maxCoord.z - minCoord.z));

   double x = _centers.size()/centersPerCell/(lx*ly*lz);
   x = pow(x, 1.0/3.0);
   _nx = max(1., floor(x*lx));
   _ny = max(1., floor(x*ly));
   _nz = max(1., floor(x*lz));
   _dx = lx/_nx;
   _dy = ly/_ny;
   _dz = lz/_nz;

   int nCells = _nx * _ny * _nz;

   _grid.resize( nCells );

   for (int ii=0; ii<_centers.size(); ++ii)
   {
      int iCell = whichCell(_centers[ii]);
      _grid[iCell]._myCenters.push_back(ii);
   }
}

int GridAssignmentObject::nearestCenter(const MC_Vector r)
{
   double r2Min = 1e300;
   int minCenter = -1;

   addTupleToQueue(whichCellTuple(r));

   while (_floodQueue.size() > 0)
   {
      // pop the next cell to check
      int iCell = _floodQueue.front(); _floodQueue.pop();
      // if cell is too far away to bother continue.
      if (minDist2(r, iCell) > r2Min)
         continue;
      // check all centers in this cell
      for (int ii=0; ii<_grid[iCell]._myCenters.size(); ++ii)
      {
         int iCenter = _grid[iCell]._myCenters[ii];

         const MC_Vector& rCenter = _centers[iCenter];
         double r2 = DIFFSQ(r, rCenter);
         if (r2 == r2Min)
            minCenter = min(minCenter, iCenter);
         if (r2 < r2Min)
         {
            r2Min = r2;
            minCenter = iCenter;
         }
      }
      // push any unused nbrs to queue.  Mark as used.
      addNbrsToQueue(iCell);
   }

   while (_wetList.size() > 0)
   {
      _grid[_wetList.front()]._burned = false;
      _wetList.pop();
   }

   qs_assert(minCenter >= 0);
   return minCenter;
}


Tuple GridAssignmentObject::whichCellTuple(const MC_Vector r) const
{
   int ix = (r.x-_corner.x)/_dx;
   int iy = (r.y-_corner.y)/_dy;
   int iz = (r.z-_corner.z)/_dz;
   ix = max(0, ix);
   iy = max(0, iy);
   iz = max(0, iz);
   ix = min(_nx-1, ix);
   iy = min(_ny-1, iy);
   iz = min(_nz-1, iz);

   return Tuple(ix, iy, iz);
}

int GridAssignmentObject::whichCell(const MC_Vector r) const
{
   return tupleToIndex(whichCellTuple(r));
}

int GridAssignmentObject::tupleToIndex(Tuple tuple) const
{
   return tuple.x() + _nx * (tuple.y() + _ny*tuple.z());
}

Tuple GridAssignmentObject::indexToTuple(int index) const
{
   int ix = index % _nx;
   index /= _nx;
   int iy = index % _ny;
   int iz = index / _ny;
   return Tuple(ix, iy, iz);
}

/** Finds a lower bound of the squared distance from the point r to the
 * cell with index iCell.  As presently implemented this calculation is
 * very conservative.  We could set a larger lower bound by considering
 * the location of the particle within the cell in which it lies.  */
double GridAssignmentObject::minDist2(const MC_Vector r, int iCell) const
{
   Tuple ir = whichCellTuple(r);
   Tuple iTuple = indexToTuple(iCell);

   double rx = _dx*(abs(iTuple.x() - ir.x()) - 1); rx = max(0., rx);
   double ry = _dy*(abs(iTuple.y() - ir.y()) - 1); ry = max(0., ry);
   double rz = _dz*(abs(iTuple.z() - ir.z()) - 1); rz = max(0., rz);

   return rx*rx + ry*ry + rz*rz;
}

void GridAssignmentObject::addTupleToQueue(Tuple iTuple)
{
   int index = tupleToIndex(iTuple);
   if (_grid[index]._burned)
      return;
   _floodQueue.push(index);
   _wetList.push(index);
   _grid[index]._burned = true;
}

void GridAssignmentObject::addNbrsToQueue(int iCell)
{
   Tuple iTuple = indexToTuple(iCell);
   iTuple.x() += 1; if (iTuple.x() < _nx) addTupleToQueue(iTuple);
   iTuple.x() -= 2; if (iTuple.x() >= 0)  addTupleToQueue(iTuple);
   iTuple.x() += 1;

   iTuple.y() += 1; if (iTuple.y() < _ny) addTupleToQueue(iTuple);
   iTuple.y() -= 2; if (iTuple.y() >= 0)  addTupleToQueue(iTuple);
   iTuple.y() += 1;

   iTuple.z() += 1; if (iTuple.z() < _nz) addTupleToQueue(iTuple);
   iTuple.z() -= 2; if (iTuple.z() >= 0)  addTupleToQueue(iTuple);
   iTuple.z() += 1;
}
