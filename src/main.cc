/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#include "hip/hip_runtime.h"
#include <iostream>
#include "utils.hh"
#include "Parameters.hh"
#include "utilsMpi.hh"
#include "MonteCarlo.hh"
#include "initMC.hh"
#include "Tallies.hh"
#include "PopulationControl.hh"
#include "ParticleVaultContainer.hh"
#include "ParticleVault.hh"
#include "MC_Particle_Buffer.hh"
#include "MC_Processor_Info.hh"
#include "MC_Time_Info.hh"
#include "macros.hh"
#include "MC_Fast_Timer.hh"
#include "MC_SourceNow.hh"
#include "SendQueue.hh"
#include "NVTX_Range.hh"
#include "hipUtils.hh"
#include "hipFunctions.hh"
#include "qs_assert.hh"
#include "CycleTracking.hh"
#include "CoralBenchmark.hh"
#include "EnergySpectrum.hh"

#include "git_hash.hh"
#include "git_vers.hh"

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "utilsMpi.hh"

void gameOver();
void cycleInit( bool loadBalance );
void cycleTracking(MonteCarlo* monteCarlo,uint64_cu*,uint64_cu*);
void cycleFinalize();

void setGPU()
{

    int rank;
    MPI_Comm  comm_mc_world(MPI_COMM_WORLD);

    int Ngpus;
    hipGetDeviceCount(&Ngpus);

    mpiComm_rank(comm_mc_world, &rank);
    int GPUID = rank%Ngpus;
    hipSetDevice(GPUID);


}

using namespace std;

MonteCarlo *mcco  = NULL;

int main(int argc, char** argv)
{
   mpiInit(&argc, &argv);
   printBanner(GIT_VERS, GIT_HASH);

   Parameters params = getParameters(argc, argv);
   printParameters(params, cout);

   setGPU();

   // mcco stores just about everything. 
   mcco = initMC(params); 

   int myRank, nRanks;
   mpiComm_rank(MPI_COMM_WORLD, &myRank);


   
   copyMaterialDatabase_device(mcco);
   copyNuclearData_device(mcco->_nuclearData,mcco->_nuclearData_d);
   copyDomainDevice(mcco->_nuclearData->_numEnergyGroups,mcco->domain,mcco->domain_d,mcco->domainSize);
   
   mpiBarrier(MPI_COMM_WORLD);
   int loadBalance = params.simulationParams.loadBalance;

   MC_FASTTIMER_START(MC_Fast_Timer::main);     // this can be done once mcco exist.

   const int nSteps = params.simulationParams.nSteps;

   //allocate arrays to hold counters in pinned memory on the host and on the device.
   int replications = mcco->_tallies->GetNumBalanceReplications();
   uint64_cu * tallies;
   hipHostMalloc( (void **) &tallies,sizeof(uint64_cu)*NUM_TALLIES*replications); 

   uint64_cu * tallies_d;
   hipMalloc( (void **) &tallies_d, sizeof(uint64_cu)*NUM_TALLIES*replications);


   for (int ii=0; ii<nSteps; ++ii)
   {
      cycleInit( bool(loadBalance) );
      cycleTracking(mcco,tallies,tallies_d);
      cycleFinalize();


      mcco->fast_timer->Last_Cycle_Report(
            params.simulationParams.cycleTimers,
            mcco->processor_info->rank,
            mcco->processor_info->num_processors,
            mcco->processor_info->comm_mc_world );
   }


   MC_FASTTIMER_STOP(MC_Fast_Timer::main);

   gameOver();

   coralBenchmarkCorrectness(mcco, params);

   hipHostFree(tallies);
   hipFree(tallies_d);

#ifdef HAVE_UVM
    mcco->~MonteCarlo();
    gpuFree( mcco );
#else
   delete mcco;
#endif

   mpiFinalize();
   
   return 0;
}

void gameOver()
{
    mcco->fast_timer->Cumulative_Report(mcco->processor_info->rank,
                                        mcco->processor_info-> num_processors,
                                        mcco->processor_info->comm_mc_world,
                                        mcco->_tallies->_balanceCumulative._numSegments);
    mcco->_tallies->_spectrum.PrintSpectrum(mcco);
}

void cycleInit( bool loadBalance )
{

    MC_FASTTIMER_START(MC_Fast_Timer::cycleInit);

    mcco->clearCrossSectionCache();

    mcco->_tallies->CycleInitialize(mcco);

    mcco->_particleVaultContainer->swapProcessingProcessedVaults();

    mcco->_particleVaultContainer->collapseProcessed();
    mcco->_particleVaultContainer->collapseProcessing();

    mcco->_tallies->_balanceTask[0]._start = mcco->_particleVaultContainer->sizeProcessing();

    mcco->particle_buffer->Initialize();

    MC_SourceNow(mcco);
   
    PopulationControl(mcco, loadBalance); // controls particle population

    RouletteLowWeightParticles(mcco); // Delete particles with low statistical weight

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleInit);
}


#if defined (HAVE_HIP)

__launch_bounds__(256) __global__ void CycleTrackingKernel(MonteCarlo* monteCarlo, int num_particles, ParticleVault* processingVault, ParticleVault* processedVault, uint64_cu * tallies )
{
   int global_index = getGlobalThreadID(); 
   int local_index = getLocalThreadID();
   int replications = monteCarlo->_tallies->GetNumBalanceReplications();

   extern __shared__ int values_l[];
   if(local_index<replications*NUM_TALLIES)
   { 
      values_l[local_index]=0;
   }
   __syncthreads();

   if (global_index<replications*NUM_TALLIES)
   {
      tallies[global_index]=0;
   }
   if( global_index < num_particles )
   {
       CycleTrackingGuts( monteCarlo, global_index, processingVault, processedVault, &values_l[0] );
   }

   __syncthreads();
   if(local_index<replications*NUM_TALLIES)
   { 
      #if defined (HAVE_CUDA)
      ATOMIC_ADD(tallies[local_index], (uint64_cu)values_l[local_index]);
      #else
      __atomic_fetch_add(&(tallies[local_index]), (uint64_t)values_l[local_index], __ATOMIC_RELAXED);
      #endif
   }
}
#endif

void cycleTracking(MonteCarlo *monteCarlo, uint64_cu* tallies, uint64_cu * tallies_d)
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking);

    bool done = false;

    //Determine whether or not to use GPUs if they are available (set for each MPI rank)
    ExecutionPolicy execPolicy = getExecutionPolicy( monteCarlo->processor_info->use_gpu );

    ParticleVaultContainer &my_particle_vault = *(monteCarlo->_particleVaultContainer);

    //Post Inital Receives for Particle Buffer
    monteCarlo->particle_buffer->Post_Receive_Particle_Buffer( my_particle_vault.getVaultSize() );

    //Get Test For Done Method (Blocking or non-blocking
    MC_New_Test_Done_Method::Enum new_test_done_method = monteCarlo->particle_buffer->new_test_done_method;
 
    int l5=0;
    
    const int replications = monteCarlo->_tallies->GetNumBalanceReplications();
 
    do
    {

        int particle_count = 0; // Initialize count of num_particles processed

        while ( !done )
        {
            uint64_t fill_vault = 0;

            for ( uint64_t processing_vault = 0; processing_vault < my_particle_vault.processingSize(); processing_vault++ )
            {
                MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Kernel);
                uint64_t processed_vault = my_particle_vault.getFirstEmptyProcessedVault();

                ParticleVault *processingVault = my_particle_vault.getTaskProcessingVault(processing_vault);
                ParticleVault *processedVault =  my_particle_vault.getTaskProcessedVault(processed_vault);
            
                int numParticles = processingVault->size();
                

                if ( numParticles != 0 )
                {
                    NVTX_Range trackingKernel("cycleTracking_TrackingKernel"); // range ends at end of scope

                    // The tracking kernel can run
                    // * As a cuda kernel
                    // * As an OpenMP 4.5 parallel loop on the GPU
                    // * As an OpenMP 3.0 parallel loop on the CPU
                    // * AS a single thread on the CPU.


                    switch (execPolicy)
                    {
                      case gpuWithHIP:
                       {
                          #if defined (HAVE_HIP)
                          dim3 grid(1,1,1);
                          dim3 block(1,1,1);
                          int runKernel = ThreadBlockLayout( grid, block, numParticles);
                          //Call Cycle Tracking Kernel

                          if( runKernel)
                          {
                              hipLaunchKernelGGL((CycleTrackingKernel), dim3(grid), dim3(block ), NUM_TALLIES*replications*sizeof(int), 0,  monteCarlo, numParticles, processingVault, processedVault,tallies_d);

                              hipError_t errorchk = hipPeekAtLastError();

                              if (errorchk != hipSuccess)
                              {
                                   std::cout << "error: #" << errorchk << " (" << hipGetErrorString(errorchk) << std::endl;
                                   abort();
                              }

                              //Synchronize the stream
                          }
                          hipMemcpy(tallies,tallies_d,NUM_TALLIES*sizeof(uint64_cu)*replications,hipMemcpyDeviceToHost);

                          #endif
                       }
                       break;
                       
                      case gpuWithOpenMP:
                       {
      
                          std::cout<<" this isn't supported with hip yet "<<std::endl;
                       }
                       break;

                      case cpu:
                       #include "mc_omp_parallel_for_schedule_static.hh"
                       for ( int particle_index = 0; particle_index < numParticles; particle_index++ )
                       {
                          CycleTrackingGuts( monteCarlo, particle_index, processingVault, processedVault, &particle_index );
                       }
                       break;
                      default:
                       qs_assert(false);

                    } // end switch

                    //Add in counters from GPU kernel
                    for (int il=0;il<replications;il++)
                    {
                     monteCarlo->_tallies->_balanceTask[il]._numSegments+=tallies[NUM_TALLIES*il+0];
                     monteCarlo->_tallies->_balanceTask[il]._escape+=tallies[NUM_TALLIES*il+1];
                     monteCarlo->_tallies->_balanceTask[il]._census+=tallies[NUM_TALLIES*il+2];
                     monteCarlo->_tallies->_balanceTask[il]._collision+=tallies[NUM_TALLIES*il+3];
                     monteCarlo->_tallies->_balanceTask[il]._scatter+=tallies[NUM_TALLIES*il+4];
                     monteCarlo->_tallies->_balanceTask[il]._absorb+=tallies[NUM_TALLIES*il+5];
                     monteCarlo->_tallies->_balanceTask[il]._fission+=tallies[NUM_TALLIES*il+6];
                     monteCarlo->_tallies->_balanceTask[il]._produce+=tallies[NUM_TALLIES*il+7];
                    }

                }

                particle_count += numParticles;
                MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Kernel);

                MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_MPI);

                // Next, communicate particles that have crossed onto
                // other MPI ranks.
                NVTX_Range cleanAndComm("cycleTracking_clean_and_comm");
                
                SendQueue &sendQueue = *(my_particle_vault.getSendQueue());
                monteCarlo->particle_buffer->Allocate_Send_Buffer( sendQueue );

                //Move particles from send queue to the send buffers
                for ( int index = 0; index < sendQueue.size(); index++ )
                {
                    sendQueueTuple& sendQueueT = sendQueue.getTuple( index );
                    MC_Base_Particle mcb_particle;

                    processingVault->getBaseParticleComm( mcb_particle, sendQueueT._particleIndex );

                    int buffer = monteCarlo->particle_buffer->Choose_Buffer(sendQueueT._neighbor );
                    monteCarlo->particle_buffer->Buffer_Particle(mcb_particle, buffer );
                }

                monteCarlo->particle_buffer->Send_Particle_Buffers(); // post MPI sends

                processingVault->clear(); //remove the invalid particles
                sendQueue.clear();

                // Move particles in "extra" vaults into the regular vaults.
                my_particle_vault.cleanExtraVaults();

                // receive any particles that have arrived from other ranks
                monteCarlo->particle_buffer->Receive_Particle_Buffers( fill_vault );

                MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_MPI);


            } // for loop on vaults


            MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_MPI);

            NVTX_Range collapseRange("cycleTracking_Collapse_ProcessingandProcessed");
            my_particle_vault.collapseProcessing();
            my_particle_vault.collapseProcessed();
            collapseRange.endRange();


            //Test for done - blocking on all MPI ranks
            NVTX_Range doneRange("cycleTracking_Test_Done_New");
            done = monteCarlo->particle_buffer->Test_Done_New( new_test_done_method );
            doneRange.endRange();

            MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_MPI);
            

        } // while not done: Test_Done_New()


        // Everything should be done normally.
        done = monteCarlo->particle_buffer->Test_Done_New( MC_New_Test_Done_Method::Blocking );

    } while ( !done );

    //Make sure to cancel all pending receive requests
    monteCarlo->particle_buffer->Cancel_Receive_Buffer_Requests();
    //Make sure Buffers Memory is Free
    monteCarlo->particle_buffer->Free_Buffers();

   MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking);
}


void cycleFinalize()
{
    MC_FASTTIMER_START(MC_Fast_Timer::cycleFinalize);

    mcco->_tallies->_balanceTask[0]._end = mcco->_particleVaultContainer->sizeProcessed();

    // Update the cumulative tally data.
    mcco->_tallies->CycleFinalize(mcco); 

    mcco->time_info->cycle++;

    mcco->particle_buffer->Free_Memory();

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleFinalize);
}

