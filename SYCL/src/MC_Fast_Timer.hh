#ifndef MC_FAST_TIMER_INCLUDE
#define MC_FAST_TIMER_INCLUDE

#include <iostream>
#ifndef CHRONO_MISSING
#include <chrono>
#endif

#include "portability.hh"   // needed for uint64_t in this file
#include "utilsMpi.hh"      // needed for MPI_Comm type in this file

class MC_Fast_Timer
{
    public:
    uint64_t numCalls;
#ifdef CHRONO_MISSING
    double startClock;                                              // from MPI
    double stopClock;
#else
    std::chrono::high_resolution_clock::time_point startClock;      // from c++11 high resolution timer calls
    std::chrono::high_resolution_clock::time_point stopClock;
#endif
    uint64_t lastCycleClock;                                        // in microseconds
    uint64_t cumulativeClock;                                       // in microseconds


  MC_Fast_Timer() : numCalls(0), startClock(), stopClock(), lastCycleClock(0), cumulativeClock(0)  {} ; // consturctor


  // 1 enumerated type for each timed section, this is hardcoded for efficiency.
  enum Enum
  {
    main = 0,
    cycleInit,
    cycleTracking,
    cycleTracking_Kernel,
    cycleTracking_MPI,
    cycleTracking_Test_Done,
    cycleFinalize,
    Num_Timers
  };
};

class MC_Fast_Timer_Container
{
public:
    MC_Fast_Timer_Container() {} ; // constructor
    void Cumulative_Report(int mpi_rank, int num_ranks, MPI_Comm comm_world, uint64_t numSegments);
    void Last_Cycle_Report(int report_time, int mpi_rank, int num_ranks, MPI_Comm comm_world);
    void Clear_Last_Cycle_Timers();
    MC_Fast_Timer  timers[MC_Fast_Timer::Num_Timers];  // timers for various routines

private:
    void Print_Cumulative_Heading(int mpi_rank);
    void Print_Last_Cycle_Heading(int mpi_rank);
};


extern const int   mc_fast_timer_enums[MC_Fast_Timer::Num_Timers];
extern const char *mc_fast_timer_names[MC_Fast_Timer::Num_Timers];

#ifdef DISABLE_TIMERS // Disable timers with empty macros -- do not make timer calls

   #define MC_FASTTIMER_START(timerIndex)
   #define MC_FASTTIMER_STOP(timerIndex)
   #define MC_FASTTIMER_GET_LASTCYCLE(timerIndex) 0.0

#else   // DISABLE_TIMERS not defined.  Set up timers 

   #ifdef CHRONO_MISSING   // compiler does not support high resolution timer, use MPI timer instead

      #define MC_FASTTIMER_START(timerIndex) \
         if (omp_get_thread_num() == 0) {				      \
            mcco->fast_timer->timers[timerIndex].startClock = mpiWtime(); \
         }

      #define MC_FASTTIMER_STOP(timerIndex) \
          if ( omp_get_thread_num() == 0 ) { \
              mcco->fast_timer->timers[timerIndex].stopClock = mpiWtime(); \
              mcco->fast_timer->timers[timerIndex].lastCycleClock   += \
		(long unsigned) ((mcco->fast_timer->timers[timerIndex].stopClock - mcco->fast_timer->timers[timerIndex].startClock) * 1000000.0); \
              mcco->fast_timer->timers[timerIndex].cumulativeClock += \
		(long unsigned) ((mcco->fast_timer->timers[timerIndex].stopClock - mcco->fast_timer->timers[timerIndex].startClock) * 1000000.0); \
              mcco->fast_timer->timers[timerIndex].numCalls++; \
          }

      #define MC_FASTTIMER_GET_LASTCYCLE(timerIndex) (float)mcco->fast_timer->timers[timerIndex].lastCycleClock / 1000000.

   #else // else CHRONO_MISSING is not defined, so high resolution clock is available

      #define MC_FASTTIMER_START(timerIndex) \
          if (omp_get_thread_num() == 0) { \
              mcco->fast_timer->timers[timerIndex].startClock = std::chrono::high_resolution_clock::now(); \
          }

      #define MC_FASTTIMER_STOP(timerIndex) \
          if ( omp_get_thread_num() == 0 ) { \
              mcco->fast_timer->timers[timerIndex].stopClock = std::chrono::high_resolution_clock::now(); \
              mcco->fast_timer->timers[timerIndex].lastCycleClock += \
                std::chrono::duration_cast<std::chrono::microseconds> \
		(mcco->fast_timer->timers[timerIndex].stopClock - mcco->fast_timer->timers[timerIndex].startClock).count(); \
              mcco->fast_timer->timers[timerIndex].cumulativeClock += \
	        std::chrono::duration_cast<std::chrono::microseconds> \
	        (mcco->fast_timer->timers[timerIndex].stopClock - mcco->fast_timer->timers[timerIndex].startClock).count(); \
              mcco->fast_timer->timers[timerIndex].numCalls++;		\
          }

      #define MC_FASTTIMER_GET_LASTCYCLE(timerIndex) (float)mcco->fast_timer->timers[timerIndex].lastCycleClock / 1000000.


   #endif // end ifdef CHRONO_MISSING else section
#endif // end if DISABLE_TIMERS

#endif // end ifdef MC_FAST_TIMER_INCLUDE
