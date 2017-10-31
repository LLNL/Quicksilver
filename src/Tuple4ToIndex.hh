#ifndef TUPLE4_TO_INDEX_HH
#define TUPLE4_TO_INDEX_HH

#include "Long64.hh"
#include "Tuple4.hh"

class Tuple4ToIndex
{
 public:
   Tuple4ToIndex(int nx, int ny, int nz, int nb);

   Long64 operator()(int ix, int iy, int iz, int ib) const;
   Long64 operator()(const Tuple4& tt) const;

 private:
   Long64 nx_;  // needs to be Long64 to force 64-bit math below
   Long64 ny_;
   Long64 nz_;
   Long64 nb_;
};

inline Tuple4ToIndex::Tuple4ToIndex(int nx, int ny, int nz, int nb)
: nx_(nx), ny_(ny), nz_(nz), nb_(nb)
{}

inline Long64
Tuple4ToIndex::operator()(int ix, int iy, int iz, int ib) const
{
   return ix + nx_*(iy + ny_*(iz + nz_*(ib)));
}

inline Long64
Tuple4ToIndex::operator()(const Tuple4& tt) const
{
   return tt.x() + nx_*(tt.y() + ny_*(tt.z() + nz_*(tt.b())));
}

#endif
