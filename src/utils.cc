#include "utils.hh"
#include <cstdio>
#include "qs_assert.hh"
#include "utilsMpi.hh"
#include "macros.hh"
#include <vector>
#include <stdarg.h>
#include <string.h>
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Processor_Info.hh"


// Returns the number of physical cores.  Relies on the env var
// KMP_PLACE_THREADS being set to someting like 60c2t.
// Otherwise, returns omp_get_num_procs()
int mc_get_num_physical_procs(void)
{
   int num_physical_cores = omp_get_num_procs();
   #if defined(HAVE_OPENMP) && defined(HAVE_KNL)
   int num_threads_per_core = 0;
   char *env_str = getenv("KMP_PLACE_THREADS");
   if (env_str)
   {
      char *ptr = strchr(env_str, (int)'t');
      if (ptr)
      {
         int num_threads_per_core = 1;
         ptr--;
         while ((ptr > env_str) && isdigit(*ptr) )
         { num_threads_per_core = atoi(ptr); ptr--; }
         if (num_threads_per_core > 0) 
         { num_physical_cores = omp_get_num_procs() / num_threads_per_core; }
      }
   }
   #endif
   return num_physical_cores;
}


void MC_Verify_Thread_Zero(char const * const file, int line)
{
#ifdef HAVE_OPENMP
    int thread_id = omp_get_thread_num();
    if (thread_id != 0)
    {
        int mpi_rank = -1;
        mpiComm_rank(mcco->processor_info->comm_mc_world, &mpi_rank);
        fprintf(stderr,"Fatal Error: %s:%d MPI Routine called by thread other than zero."
                       "\n\tMPI Process %d, Thread %d", file, line, mpi_rank, thread_id);
        mpiAbort(MPI_COMM_WORLD, -1); abort();
    }
#endif
    return;
}

void printBanner(const char *git_version, const char *git_hash)
{
    int rank = -1, size=-1, mpi_major=0, mpi_minor=0;
    mpiComm_rank(MPI_COMM_WORLD, &rank);
    mpiComm_size(MPI_COMM_WORLD, &size);
    mpiGet_version(&mpi_major, &mpi_minor);

    if (rank == 0)
    {
        printf("Copyright (c) 2016\n");
        printf("Lawrence Livermore National Security, LLC\n");
        printf("All Rights Reserved\n");

        printf("Quicksilver Version     : %s\n",git_version);
        printf("Quicksilver Git Hash    : %s\n",git_hash);
        printf("MPI Version             : %d.%d\n",mpi_major,mpi_minor);
        printf("Number of MPI ranks     : %d\n",size);
        printf("Number of OpenMP Threads: %d\n",(int)omp_get_max_threads());
        printf("Number of OpenMP CPUs   : %d\n\n",(int)omp_get_num_procs());
    }
}

void Print0(const char *format, ...)
{
    int rank = -1;
    mpiComm_rank(MPI_COMM_WORLD, &rank);

#if 0
    printf("rank %i: ", rank);
#else
    if ( rank != 0 ) { return; }
#endif

    va_list args;
    va_start( args, format );
    vprintf(format, args);
    va_end( args );
}

//----------------------------------------------------------------------------------------------------------------------
// Converts a format string into a c++ string. Parameters are the same as printf.
//----------------------------------------------------------------------------------------------------------------------
std::string MC_String(const char fmt[], ...)
{
    va_list args;
    va_start(args, fmt);
    int chars_needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    if (chars_needed < 0)
    {
        MC_Fatal_Jump( "Output error from vsnprintf: %d", chars_needed );
    }

    // Increase one for the null terminator.
    chars_needed++;

    // Bump up chars_needed (if necessary) so that we allocate according to our byte alignment.
    // This is currently 16 bytes, so allocated 16, 32 48, etc. bytes at a time.
#define MC_BYTE_ALIGNMENT 16

    int remainder = chars_needed % MC_BYTE_ALIGNMENT;
    chars_needed += remainder > 0 ? MC_BYTE_ALIGNMENT - remainder: 0;

    std::vector<char> buffer(chars_needed);
    va_start(args, fmt);
    vsnprintf(&buffer[0], chars_needed, fmt, args);
    va_end(args);

    return std::string(&buffer[0]);
}

