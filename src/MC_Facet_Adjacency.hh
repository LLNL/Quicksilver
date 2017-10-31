#ifndef MCT_FACET_ADJACENCY_INCLUDE
#define MCT_FACET_ADJACENCY_INCLUDE


#include <vector>
#include "MC_Location.hh"
#include "macros.hh"

struct MC_Subfacet_Adjacency_Event
{
 public:
   enum Enum
   {
      Adjacency_Undefined = 0,
      Boundary_Escape,
      Boundary_Reflection,
      Transit_On_Processor,
      Transit_Off_Processor
   };
};

class Subfacet_Adjacency
{
 public:
   MC_Subfacet_Adjacency_Event::Enum event;
   MC_Location current;
   MC_Location adjacent;
   int neighbor_index;
   int neighbor_global_domain;
   int neighbor_foreman;


   Subfacet_Adjacency()
   : event(MC_Subfacet_Adjacency_Event::Adjacency_Undefined),
     current(),
     adjacent(),
     neighbor_index(-1),
     neighbor_global_domain(-1),
     neighbor_foreman(-1)
   {}
};

class MC_Facet_Adjacency
{
 public:
   Subfacet_Adjacency   subfacet;
   int                  num_points;   // the number of points defining that facet, for polyhedra
   int                  point[3];       //  the points defining that facet, for polyhedra

   MC_Facet_Adjacency() : subfacet(), num_points(3) {point[0] = point[1] = point[2] = -1;}
};

class MC_Facet_Adjacency_Cell
{
 public:
   int                  num_facets; // 6 quad faces, each quad has 3 triangles = 24 faces
   MC_Facet_Adjacency*  _facet;
   int                  num_points;  // 8 hex corners + 6 face centers = 14 points
   int*                 _point;       

   MC_Facet_Adjacency_Cell() : num_facets(24), _facet(0), num_points(14), _point(0) {}
private:

};


#endif
