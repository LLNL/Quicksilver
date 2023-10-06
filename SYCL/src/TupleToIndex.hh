#ifndef TUPLE_TO_INDEX_HH
#define TUPLE_TO_INDEX_HH

#include "Long64.hh"
#include "Tuple.hh"

class TupleToIndex
{
 public:
   TupleToIndex(int nx, int ny, int nz);

   Long64 operator()(int ix, int iy, int iz) const;
   Long64 operator()(const Tuple& tt) const;

 private:
    Long64 nx_;  // needs to be Long64 to force 64-bit math below
    Long64 ny_;
    Long64 nz_;
};

inline TupleToIndex::TupleToIndex(int nx, int ny, int nz)
: nx_(nx), ny_(ny), nz_(nz)
{}

inline Long64
TupleToIndex::operator()(int ix, int iy, int iz) const
{
   return ix + nx_*(iy + ny_*(iz));
}

inline Long64
TupleToIndex::operator()(const Tuple& tt) const
{
   return tt.x() + nx_*(tt.y() + ny_*(tt.z()));
}

#endif
