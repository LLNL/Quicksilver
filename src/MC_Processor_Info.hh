#ifndef MC_PROCESSOR_INFO_HH
#define MC_PROCESSOR_INFO_HH

#include "utilsMpi.hh"
#include "macros.hh"

class MC_Processor_Info
{
public:

    int rank;
    int num_processors;
    int num_tasks;
    int use_gpu;
    int gpu_id;

    MPI_Comm  comm_mc_world;
    MPI_Comm *comm_mc_world_threads;  // Communicator to allow threads to make MPI calls.

    MC_Processor_Info() : comm_mc_world(MPI_COMM_WORLD)
    {
      mpiComm_rank(comm_mc_world, &rank);
      mpiComm_size(comm_mc_world, &num_processors);
//      num_tasks = omp_get_max_threads();
      num_tasks = 1;
      comm_mc_world_threads = new MPI_Comm[num_tasks];

      for (int thread_ndx=0; thread_ndx<num_tasks; ++thread_ndx)
      {
         mpiComm_split(comm_mc_world, thread_ndx, 0, &comm_mc_world_threads[thread_ndx]);
      }

      use_gpu = 0;
      gpu_id = 0;
    }

};



#endif
