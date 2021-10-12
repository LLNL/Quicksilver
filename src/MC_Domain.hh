/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


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

class MC_Mesh_Domain_d
{
 public:

   int _domainGid; //dfr: Might be able to delete this later.

   int * _nbrRank;
   int _nbrRankSize;

   MC_Vector * _node;
   int _nodeSize;

   MC_Facet_Adjacency_Cell *_cellConnectivity;
   int _cellConnectivitySize;

   MC_Facet_Geometry_Cell * _cellGeometry;
   int _cellGeometrySize;

};


//----------------------------------------------------------------------------------------------------------------------
// class that manages a region on a domain.
//----------------------------------------------------------------------------------------------------------------------

class MC_Domain_d
{
public:
   int domainIndex;  // This appears to be unused.
   int global_domain;

   MC_Cell_State * cell_state;
   int cell_stateSize;

   
    // hold mesh information
    MC_Mesh_Domain_d mesh;

};

inline void copyDomainDevice(const int numEnergyGroups,qs_vector<MC_Domain> domain, MC_Domain_d * domain_d, int & domainSize)
{
    //Create vector of domains that will live on the host, but have pointers to device memory.
    domainSize = domain.size();
    MC_Domain_d * domain_h = (MC_Domain_d *) malloc(domainSize*sizeof(MC_Domain_d));

    //loop over the number of domains creating them one at a time
    for(int i=0;i<domainSize;i++)
    {
        //set the domain index and "global_domain"
        domain_h[i].domainIndex = domain[i].domainIndex;
        domain_h[i].global_domain = domain[i].global_domain;
    
        //Create an array of cell states
        domain_h[i].cell_stateSize = domain[i].cell_state.size();
        
        MC_Cell_State * cell_state_h = (MC_Cell_State *) malloc(domain[i].cell_state.size()*sizeof(MC_Cell_State));
        hipMemcpy(cell_state_h,domain[i].cell_state.outputPointer(),domain[i].cell_state.size()*sizeof(MC_Cell_State),hipMemcpyHostToHost);


        for(int j=0;j<domain[i].cell_state.size();j++)
        {
            hipMalloc((void **) &(cell_state_h[j]._total),numEnergyGroups*sizeof(double));
            hipMemcpy(cell_state_h[j]._total,domain[i].cell_state[j]._total,numEnergyGroups*sizeof(double),hipMemcpyHostToDevice);   
        }

        hipMalloc((void **) &domain_h[i].cell_state,domain[i].cell_state.size()*sizeof(MC_Cell_State));
        //hipMemcpy(domain_h[i].cell_state,domain[i].cell_state.outputPointer(),domain[i].cell_state.size()*sizeof(MC_Cell_State),hipMemcpyHostToDevice);
        hipMemcpy(domain_h[i].cell_state,cell_state_h,domain[i].cell_state.size()*sizeof(MC_Cell_State),hipMemcpyHostToDevice);
        domain_h[i].mesh._domainGid = domain[i].mesh._domainGid;

	free(cell_state_h);

        domain_h[i].mesh._nbrRankSize = domain[i].mesh._nbrRank.size();
        hipMalloc((void **) &domain_h[i].mesh._nbrRank,domain[i].mesh._nbrRank.size()*sizeof(int));
        hipMemcpy(domain_h[i].mesh._nbrRank,domain[i].mesh._nbrRank.outputPointer(),domain[i].mesh._nbrRank.size()*sizeof(int),hipMemcpyHostToDevice);

        domain_h[i].mesh._nodeSize = domain[i].mesh._node.size();
        hipMalloc((void **) &(domain_h[i].mesh._node),domain_h[i].mesh._nodeSize*sizeof(MC_Vector));
        hipMemcpy(domain_h[i].mesh._node,domain[i].mesh._node.outputPointer(),domain_h[i].mesh._nodeSize*sizeof(MC_Vector),hipMemcpyHostToDevice);

        int _cellConnectivitySize = domain[i].mesh._cellConnectivity.size(); 
        domain_h[i].mesh._cellConnectivitySize = _cellConnectivitySize;
        MC_Facet_Adjacency_Cell * cellConnectivity = (MC_Facet_Adjacency_Cell *) malloc(_cellConnectivitySize*sizeof(MC_Facet_Adjacency_Cell));
        for(int j=0;j<_cellConnectivitySize;j++)
        {
           cellConnectivity[j].num_points=domain[i].mesh._cellConnectivity[j].num_points;
           cellConnectivity[j].num_facets=domain[i].mesh._cellConnectivity[j].num_facets;
           hipMalloc((void**) &cellConnectivity[j]._point,cellConnectivity[j].num_points*sizeof(int));
           hipMemcpy(cellConnectivity[j]._point,domain[i].mesh._cellConnectivity[j]._point,cellConnectivity[j].num_points*sizeof(int),hipMemcpyHostToDevice);
           hipMalloc((void**) &cellConnectivity[j]._facet,cellConnectivity[j].num_facets*sizeof(MC_Facet_Adjacency));
           hipMemcpy(cellConnectivity[j]._facet,domain[i].mesh._cellConnectivity[j]._facet,cellConnectivity[j].num_facets*sizeof(MC_Facet_Adjacency),hipMemcpyHostToDevice);
        }
        hipMalloc((void **) &domain_h[i].mesh._cellConnectivity,_cellConnectivitySize*sizeof(MC_Facet_Adjacency_Cell));
        hipMemcpy(domain_h[i].mesh._cellConnectivity,cellConnectivity,_cellConnectivitySize*sizeof(MC_Facet_Adjacency_Cell),hipMemcpyHostToDevice);
        free(cellConnectivity);

        int _cellGeometrySize = domain[i].mesh._cellGeometry.size();
        domain_h[i].mesh._cellGeometrySize = _cellGeometrySize;
        MC_Facet_Geometry_Cell * cellGeometry = (MC_Facet_Geometry_Cell *) malloc(_cellGeometrySize*sizeof(MC_Facet_Geometry_Cell));
        for(int j=0;j<_cellGeometrySize;j++)
        {
            cellGeometry[j]._size=domain[i].mesh._cellGeometry[j]._size;
            hipMalloc((void **) &cellGeometry[j]._facet,cellGeometry[j]._size*sizeof(MC_General_Plane));
            hipMemcpy(cellGeometry[j]._facet,domain[i].mesh._cellGeometry[j]._facet,cellGeometry[j]._size*sizeof(MC_General_Plane),hipMemcpyHostToDevice);
        }
        hipMalloc((void **) &domain_h[i].mesh._cellGeometry,_cellGeometrySize*sizeof(MC_Facet_Geometry_Cell));
        hipMemcpy(domain_h[i].mesh._cellGeometry,cellGeometry,_cellGeometrySize*sizeof(MC_Facet_Geometry_Cell),hipMemcpyHostToDevice);
        free(cellGeometry);


    }
    hipMemcpy(domain_d,domain_h,domainSize*sizeof(MC_Domain_d),hipMemcpyHostToDevice);
    free(domain_h);
} 

#endif
