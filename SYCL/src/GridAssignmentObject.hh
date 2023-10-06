#ifndef GRID_ASSIGNMENT_OBJECT_HH
#define GRID_ASSIGNMENT_OBJECT_HH

#include <vector>
#include <queue>
#include "MC_Vector.hh"
#include "Tuple.hh"

/** The GRID_ASSIGNMENT_OBJECT computes the closest center to a given
 * particle coordinate using a grid/flood approach.  The intent of this
 * code is to provide an initial assignment method that scales only as
 * the number of particles to assign.  (I.e., it is independent of the
 * number of centers).
 *
 * To vastly simplify the code we completely ignore periodic boundary
 * conditions.  We can get away with this because the initial assignment
 * doesn't have to be perfect, it only needs to be close.  If we can get
 * a particle into a domain that is close to its correct Voronoi domain
 * then the regular assignment will do the right thing.  */

class GridAssignmentObject
{
 public:

   GridAssignmentObject(const std::vector<MC_Vector>& centers);

   int nearestCenter(const MC_Vector rr);

 private:

   struct GridCell
   {
      GridCell() : _burned(false) {};

      bool _burned;
      std::vector<int> _myCenters;
   };

   Tuple whichCellTuple(const MC_Vector r) const;
   int whichCell(const MC_Vector r) const;
   int tupleToIndex(Tuple tuple) const;
   Tuple indexToTuple(int index) const;
   double minDist2(const MC_Vector r, int iCell) const;
   void addTupleToQueue(Tuple iTuple);
   void addNbrsToQueue(int iCell);

   int _nx, _ny, _nz;
   double _dx, _dy, _dz;
   MC_Vector _corner;
   const std::vector<MC_Vector>& _centers;

   std::vector<GridCell> _grid;
   std::queue<int> _floodQueue;
   std::queue<int> _wetList;
};

#endif
