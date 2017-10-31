#ifndef FACET_PAIR_HH
#define FACET_PAIR_HH

#include "MC_Location.hh"

class FacetPair
{
 public:
   FacetPair(){};
   FacetPair(int domainGid1, const MC_Location& location1,
             int domainGid2, const MC_Location& location2)
   : _domainGid1(domainGid1),
     _domainIndex1(location1.domain),
     _cellIndex1(location1.cell),
     _facetIndex1(location1.facet),
     _domainGid2(domainGid2),
     _domainIndex2(location2.domain),
     _cellIndex2(location2.cell),
     _facetIndex2(location2.facet)
   {
   }

   int _domainGid1;
   int _domainIndex1;
   int _facetIndex1;
   int _cellIndex1;
   int _domainGid2;
   int _domainIndex2;
   int _facetIndex2;
   int _cellIndex2;
};

#endif
