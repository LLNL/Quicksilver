#ifndef MC_DOMAIN_INCLUDE
#define MC_DOMAIN_INCLUDE


#include "QS_Vector.hh"
#include "MC_Facet_Adjacency.hh"
#include "MC_Vector.hh"
#include "MC_Cell_State.hh"
#include "MC_Facet_Geometry.hh"
#include "BulkStorage.hh"

class Parameters;
class MeshPartition;
class GlobalFccGrid;
class DecompositionObject;
class MaterialDatabase;


//----------------------------------------------------------------------------------------------------------------------
// class that manages data set on a mesh like geometry
//----------------------------------------------------------------------------------------------------------------------

class MC_Mesh_Domain
{
 public:

   int _domainGid; //dfr: Might be able to delete this later.

   qs_vector<int> _nbrDomainGid;
   qs_vector<int> _nbrRank;

   qs_vector<MC_Vector> _node;
   qs_vector<MC_Facet_Adjacency_Cell> _cellConnectivity;

   qs_vector<MC_Facet_Geometry_Cell> _cellGeometry;



   BulkStorage<MC_Facet_Adjacency> _connectivityFacetStorage;
   BulkStorage<int> _connectivityPointStorage;
   BulkStorage<MC_General_Plane> _geomFacetStorage;
   
    // -------------------------- public interface
   MC_Mesh_Domain(){};
   MC_Mesh_Domain(const MeshPartition& meshPartition,
                  const GlobalFccGrid& grid,
                  const DecompositionObject& ddc,
                  const qs_vector<MC_Subfacet_Adjacency_Event::Enum>& boundaryCondition);

};


//----------------------------------------------------------------------------------------------------------------------
// class that manages a region on a domain.
//----------------------------------------------------------------------------------------------------------------------

class MC_Domain
{
public:
   int domainIndex;  // This appears to be unused.
   int global_domain;

   qs_vector<MC_Cell_State> cell_state;

   BulkStorage<double> _cachedCrossSectionStorage;
   
    // hold mesh information
    MC_Mesh_Domain mesh;

   // -------------------------- public interface
    MC_Domain(){};
    MC_Domain(const MeshPartition& meshPartition, const GlobalFccGrid& grid,
              const DecompositionObject& ddc, const Parameters& params,
              const MaterialDatabase& materialDatabase, int numEnergyGroups);


   void clearCrossSectionCache(int numEnergyGroups);
};

#endif
