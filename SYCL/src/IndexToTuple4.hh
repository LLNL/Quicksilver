#ifndef INDEX_TO_TUPLE4_HH
#define INDEX_TO_TUPLE4_HH

#include "Tuple4.hh"
#include "Long64.hh"

class IndexToTuple4
{
 public:
   IndexToTuple4(int nx, int ny, int nz, int nb)
   : nx_(nx), ny_(ny), nz_(nz), nb_(nb)
   {};

   Tuple4 operator()(Long64 index) const
   {
      int x = index % nx_;
      index /= nx_;
      int y = index % ny_;
      index /= ny_;
      int z = index % nz_;
      int b = index / nz_;

      return Tuple4(x, y, z, b);
   }

 private:
   int nx_;
   int ny_;
   int nz_;
   int nb_;
};

#endif
