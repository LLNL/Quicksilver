#ifndef UTILS_MPI_HH
#define UTILS_MPI_HH

#ifdef HAVE_MPI

#if defined (GNU_PERMISSIVE)
#pragma GCC diagnostic ignored "-fpermissive"
#endif

#include <mpi.h>

#if defined (GNU_PERMISSIVE)
#pragma GCC diagnostic ignored "-pedantic"
#endif

#ifndef MPI_INT64_T
#define MPI_INT64_T  MPI_LONG_LONG
#endif

#ifndef MPI_UINT64_T
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG
#endif

double mpiWtime        ( void );
void mpiTest_cancelled ( MPI_Status *status, int *flag );
void mpiInit           ( int * argc, char *** argv );
void mpiFinalize       ( void );
void mpiComm_rank      ( MPI_Comm comm, int *rank );
void mpiComm_size      ( MPI_Comm comm, int *size );
int  mpiComm_split     ( MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
void mpiBarrier        ( MPI_Comm comm );
void mpiGet_version    ( int *version, int *subversion );
void mpiReduce         ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm );
void mpiGather         ( void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
void mpiBcast          ( void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm);
void mpiCancel         ( MPI_Request *request );
void mpiWait           ( MPI_Request *request, MPI_Status *status );
void mpiWaitall        ( int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses );
void mpiTest           ( MPI_Request *, int *, MPI_Status * );
void mpiIrecv          ( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
void mpiRecv           ( void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status);
void mpiIsend          ( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
void mpiSend           ( void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm);
void mpiType_contiguous( int count, MPI_Datatype old_type, MPI_Datatype *newtype );
void mpiType_commit    ( MPI_Datatype *datatype ) ;
void mpiAllreduce      ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm );
void mpiIAllreduce     ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm, MPI_Request *request);
void mpiScan           ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm );
void mpiAbort          ( MPI_Comm comm, int errorcode );
void mpiRequestFree    ( MPI_Request *request );

// HAVE_MPI not defined, define a serial version of  MPI that works for us
#else

#include "qs_assert.hh"
#include <stdio.h> 
#include <stdlib.h> 

typedef struct {
    int count ;
    int MPI_SOURCE ;
    int MPI_TAG ;
    int MPI_ERROR ;
} MPI_Status;

typedef int MPI_Datatype ;
typedef int MPI_Comm ;
typedef int MPI_Request ;
typedef int MPI_Op ;

// If more datatypes are added here, they must also be added to mpi_datatype_sizes in utilsMpi.cc
#define MPI_BYTE               ((MPI_Datatype)1)   // MPI official type is 3
#define MPI_INT                ((MPI_Datatype)2)   // MPI official type is 6
#define MPI_DOUBLE             ((MPI_Datatype)3)   // MPI official type is 11
#define MPI_LONG_LONG          ((MPI_Datatype)4)   // MPI official type is 13
#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)5)   // MPI official type is 35

#define MPI_REQUEST_NULL       ((MPI_Request)0)
#define MPI_STATUS_IGNORE      ((MPI_Status *)0)
#define MPI_STATUSES_IGNORE    ((MPI_Status *)0)

#define MPI_INT64_T  MPI_LONG_LONG
#define MPI_UINT64_T MPI_UNSIGNED_LONG_LONG

#define MPI_COMM_WORLD  (1)

#define MPI_MAX         (1)
#define MPI_MIN         (2)
#define MPI_SUM         (3)

inline void mpiInit           ( int * argc, char *** argv ) { return; }
inline void mpiFinalize       ( void ) { return; }
inline void mpiCancel         ( MPI_Request *request) { return; }
inline void mpiTest_cancelled ( MPI_Status *status, int *flag ) { return; }
inline void mpiWait           ( MPI_Request *request, MPI_Status *status ) { return; }
inline void mpiWaitall        ( int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses ) { return; }
inline void mpiBarrier        ( MPI_Comm comm ) { return; }
inline void mpiType_commit    ( MPI_Datatype *datatype ) { return; }
inline void mpiType_contiguous(int count, MPI_Datatype old_type, MPI_Datatype *newtype) { return; }
inline void mpiComm_rank      ( MPI_Comm comm, int *rank ) { *rank = 0; }
inline void mpiComm_size      ( MPI_Comm comm, int *size ) { *size = 1; }
inline void mpiGet_version    ( int *version, int *subversion ) { *version = 3; *subversion = 0; }
inline void mpiAbort          ( MPI_Comm comm, int errorcode ) { fprintf(stderr,"\n\nMPI_Abort called\n\n"); exit(errorcode); }

inline void mpiTest( MPI_Request *, int *, MPI_Status * )
    { printf ("mpiTest should not be called in serial run\n"); qs_assert(false); }
inline void mpiIrecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
    { printf ("mpiIrecv should not be called in serial run\n"); qs_assert(false); }
inline void mpiIsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
    { printf ("mpiIsend should not be called in serial run\n"); qs_assert(false); }
inline void mpiSend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
    { printf ("mpiSend should not be called in serial run\n"); qs_assert(false); }

inline void mpiBcast( void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm){return;}

double mpiWtime( void );
int  mpiComm_split( MPI_Comm comm, int color, int key, MPI_Comm *newcomm);
void mpiAllreduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm );
void mpiIAllreduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm, MPI_Request *request);
void mpiScan( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm );
void mpiReduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm );
void mpiGather( void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);

#endif  // end #else HAVE_MPI
#endif  // end #ifndef UTILS_MPI_HH
