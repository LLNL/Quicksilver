#include "utilsMpi.hh"
#include <cstdio>
#include <string.h>     // needed for memcpy on some compilers
#include <time.h>       // needed for clock
#include "qs_assert.hh"
#include "macros.hh"
#include "MonteCarlo.hh"
#include "MC_Processor_Info.hh"
#include "Globals.hh"


#ifdef HAVE_MPI

void mpiInit( int *argc, char ***argv)
{

#ifdef HAVE_OPENMP
   { // limit scope
      char const* const provided_string[4] = \
         {"MPI_THREAD_SINGLE","MPI_THREAD_FUNNELED","MPI_THREAD_SERIALIZED","MPI_THREAD_MULTIPLE"};
      int provided, required = MPI_THREAD_FUNNELED;
      
      int err = MPI_Init_thread(argc, argv, required, &provided);
      qs_assert(err == MPI_SUCCESS);
      
      int rank = -1;
      mpiComm_rank(MPI_COMM_WORLD, &rank);
      if (rank == 0)
         fprintf(stdout,"MPI Initialized         : %s\n", provided_string[provided]); 

      if ((required > MPI_THREAD_SINGLE) && (required > provided))
      {
         printf("MPI-OpenMP Error.\n\tCode requires %s thread support. MPI library provides %s support.\n",
                provided_string[required],provided_string[provided]);
         qs_assert(false);
      }
   } // limit scope
   
#else
   { // limit scope
      int err = MPI_Init(argc, argv);
      qs_assert(err == MPI_SUCCESS);
   } //limit scope

#endif

}


double mpiWtime( void ) { return MPI_Wtime(); }

int  mpiComm_split ( MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{
   qs_assert(MPI_Comm_split(comm, color, key, newcomm) == MPI_SUCCESS); 
   return MPI_SUCCESS;
}

void mpiComm_rank( MPI_Comm comm, int *rank ) {   qs_assert(MPI_Comm_rank(comm, rank) == MPI_SUCCESS); }
void mpiCancel( MPI_Request *request ) { qs_assert(MPI_Cancel(request) == MPI_SUCCESS); }
void mpiTest_cancelled( MPI_Status *status, int *flag ) { qs_assert(MPI_Test_cancelled(status, flag) == MPI_SUCCESS); }
void mpiTest( MPI_Request *request, int *flag, MPI_Status * status) { qs_assert(MPI_Test(request, flag, status) == MPI_SUCCESS); }
void mpiWait( MPI_Request *request, MPI_Status *status ) { qs_assert(MPI_Wait(request, status) == MPI_SUCCESS); }
void mpiComm_size( MPI_Comm comm, int *size ) { qs_assert(MPI_Comm_size(comm, size) == MPI_SUCCESS); }
void mpiBarrier( MPI_Comm comm) { qs_assert(MPI_Barrier(comm) == MPI_SUCCESS); }
void mpiGet_version( int *version, int *subversion ) { qs_assert(MPI_Get_version(version, subversion) == MPI_SUCCESS); }
void mpiFinalize( void ) { qs_assert(MPI_Finalize() == MPI_SUCCESS); }
void mpiAbort( MPI_Comm comm, int errorcode ) { qs_assert(MPI_Abort(comm, errorcode) == MPI_SUCCESS); }
void mpiRequestFree( MPI_Request *request ){qs_assert( MPI_Request_free( request ) == MPI_SUCCESS);}

void mpiScan( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm )
   { qs_assert(MPI_Scan(sendbuf, recvbuf, count, datatype, operation, comm) == MPI_SUCCESS); }
void mpiType_commit(MPI_Datatype *datatype )
   { qs_assert(MPI_Type_commit( datatype ) == MPI_SUCCESS); }
void mpiType_contiguous(int count, MPI_Datatype old_type, MPI_Datatype *newtype)
   { qs_assert(MPI_Type_contiguous(count, old_type, newtype) == MPI_SUCCESS); }
void mpiWaitall( int count, MPI_Request *array_of_requests, MPI_Status *array_of_statuses )
   { qs_assert(MPI_Waitall(count, array_of_requests, array_of_statuses) == MPI_SUCCESS); }
void mpiAllreduce ( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm )
   { qs_assert(MPI_Allreduce(sendbuf, recvbuf, count, datatype, operation, comm) == MPI_SUCCESS); }
void mpiIAllreduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm, MPI_Request *request)
#ifdef HAVE_ASYNC_MPI
   { qs_assert(MPI_Iallreduce(sendbuf, recvbuf, count, datatype, operation, comm, request) == MPI_SUCCESS); }
#else
   { qs_assert(MPI_Allreduce(sendbuf, recvbuf, count, datatype, operation, comm ) == MPI_SUCCESS); }
#endif
void mpiReduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm )
   { qs_assert(MPI_Reduce(sendbuf, recvbuf, count, datatype, op, root, comm) == MPI_SUCCESS); }
void mpiGather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
   { qs_assert(MPI_Gather(sendbuf, sendcount, sendtype, recvbuf, recvcount, recvtype, root, comm) == MPI_SUCCESS); }
void mpiBcast( void* buf, int count, MPI_Datatype datatype, int root, MPI_Comm comm)
   { qs_assert(MPI_Bcast(buf, count, datatype, root, comm) == MPI_SUCCESS); }
void mpiIrecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
   { qs_assert(MPI_Irecv(buf, count, datatype, source, tag, comm, request) == MPI_SUCCESS); }
void mpiRecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)
   { qs_assert(MPI_Recv(buf, count, datatype, source, tag, comm, status) == MPI_SUCCESS); }
void mpiIsend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
   { qs_assert(MPI_Isend(buf, count, datatype, dest, tag, comm, request) == MPI_SUCCESS); }
void mpiSend(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)
   { qs_assert(MPI_Send(buf, count, datatype, dest, tag, comm) == MPI_SUCCESS); }
    
      // -------------------------------------------------------------------------------
      // -------------------------------------------------------------------------------
#else // HAVE_MPI is not defined : Serial (non-MPI) implementation of necessary routines
      // -------------------------------------------------------------------------------
      // -------------------------------------------------------------------------------

#include "mpi_stubs_internal.hh"   // This will be our internal C++ structs.

static Handleitem *init_block(int block, Handleitem *b);
static void init_handles();
static MPI_Comm mpi_stubs_comm_new();
static void     mpi_stubs_alloc_handle(int *handle, void **data);
static pList mpi_stubs_list_new();

static MPI_Stubs_Data_type mpi_stubs_data;


// These slot numbers must match the #define of the data type in utilsMpi.hh
static size_t mpi_datatype_sizes[MPI_UNSIGNED_LONG_LONG+1] =
{
   sizeof(char),                   // slot 0 is not used
   sizeof(unsigned char),          // slot 1 MPI_Byte
   sizeof(int),                    // slot 2 MPI_Int
   sizeof(double),                 // slot 3 MPI_Double
   sizeof(long long int),          // slot 4 MPI_Long_Long
   sizeof(unsigned long long)      // slot 5 MPI_Unsigned_Long_Long
};

void mpiReduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm )
{
   if (((sendbuf == NULL) || (recvbuf == NULL)) && (count > 0))
   { printf("%s:%d - MPI_Reduce sendbuf or recvbuf is NULL \n", __FILE__, __LINE__); qs_assert(false); }

   if (root != 0)
   { printf("%s:%d - MPI_Reduce: bad root = %d\n", __FILE__, __LINE__, root); qs_assert(false); }

   switch (datatype)
   {
      case MPI_INT:
      case MPI_LONG_LONG:
      case MPI_DOUBLE:
      case MPI_UNSIGNED_LONG_LONG:
         memcpy(recvbuf, sendbuf, count * mpi_datatype_sizes[datatype]);
         break;
      default:
         printf("%s:%d - MPI_Reduce type (%d) not implemented.", __FILE__, __LINE__,datatype); qs_assert(false);
   }
}

void mpiAllreduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm )
{
   if (((sendbuf == NULL) || (recvbuf == NULL)) && (count > 0))
   { printf("%s:%d - MPI_Allreduce sendbuf or recvbuf is NULL \n",__FILE__, __LINE__); qs_assert(false); }

   switch (datatype)
   {
      case MPI_INT:
      case MPI_LONG_LONG:
      case MPI_DOUBLE:
      case MPI_UNSIGNED_LONG_LONG:
         memcpy(recvbuf, sendbuf, count * mpi_datatype_sizes[datatype]);
         break;
      default:
         printf("%s:%d - MPI_Allreduce type (%d) not implemented.", __FILE__, __LINE__, datatype);
         qs_assert(false);
  }
}

void mpiIAllreduce( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm, MPI_Request *request)
{
   if (((sendbuf == NULL) || (recvbuf == NULL)) && (count > 0))
   { printf("%s:%d - MPI_Allreduce sendbuf or recvbuf is NULL \n",__FILE__, __LINE__); qs_assert(false); }

   switch (datatype)
   {
      case MPI_INT:
      case MPI_LONG_LONG:
      case MPI_DOUBLE:
      case MPI_UNSIGNED_LONG_LONG:
         memcpy(recvbuf, sendbuf, count * mpi_datatype_sizes[datatype]);
         break;
      default:
         printf("%s:%d - MPI_Allreduce type (%d) not implemented.", __FILE__, __LINE__, datatype);
         qs_assert(false);
  }
}

void mpiScan( void *sendbuf, void *recvbuf, int count, MPI_Datatype datatype, MPI_Op operation, MPI_Comm comm )
{
   if (((sendbuf == NULL) || (recvbuf == NULL)) && (count > 0))
   { printf("%s:%d - MPI_Scan sendbuf or recvbuf is NULL \n",__FILE__, __LINE__); qs_assert(false); }

   switch (datatype)
   {
      case MPI_INT:
      case MPI_LONG_LONG:
      case MPI_DOUBLE:
      case MPI_UNSIGNED_LONG_LONG:
         memcpy(recvbuf, sendbuf, count * mpi_datatype_sizes[datatype]);
         break;
      default:
         printf("%s:%d - MPI_Scan type (%d) not implemented.", __FILE__, __LINE__, datatype);
         qs_assert(false);
   }
}

void mpiGather(void *sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm)
{
   if (sendcount != recvcount)
   { printf("%s:%d - MPI_Gather sendcount=%d != recvcount=%d\n", __FILE__, __LINE__, sendcount, recvcount); qs_assert(false); }

   if (sendtype != recvtype)
   { printf("%s:%d - MPI_Gather sendtype=%d != recvtype=%d\n", __FILE__, __LINE__, sendtype, recvtype); qs_assert(false); }

   if (((sendbuf == NULL) || (recvbuf == NULL)) && (sendcount > 0))
   { printf("%s:%d - MPI_Gather sendbuf or recvbuf is NULL \n", __FILE__, __LINE__); qs_assert(false); }

   if (root != 0)
   { fprintf(stderr,"%s:%d - MPI_Gather bad root = %d\n", __FILE__, __LINE__,root); qs_assert(false); }

   switch (recvtype)
   {
      case MPI_INT:
      case MPI_LONG_LONG:
      case MPI_DOUBLE:
      case MPI_UNSIGNED_LONG_LONG:
         memcpy(recvbuf, sendbuf, recvcount * mpi_datatype_sizes[recvtype]);
         break;
      default:
         printf("%s:%d - MPI_Gather type (%d) not implemented.", __FILE__, __LINE__, recvtype);
         qs_assert(false);
   }
}

double mpiWtime (void)
{
   double value;
   value = ( double ) clock ( ) / ( double ) CLOCKS_PER_SEC;
   return value;
}

static Handleitem *init_block(int block, Handleitem *b)
{
    for (int i=0; i<MPI_STUBS_BLOCK_ITEMS-1; i++)
    {
        b[i].handle = MPI_STUBS_HANDLE(block,i);
        b[i].next   = &b[i+1];
    }

    b[MPI_STUBS_BLOCK_ITEMS-1].handle = MPI_STUBS_HANDLE(block,MPI_STUBS_BLOCK_ITEMS-1);
    b[MPI_STUBS_BLOCK_ITEMS-1].next   =0;

    return( &(b[0]) );
}


static void init_handles()
{
    Handleitem *newh;

    newh = init_block(0,mpi_stubs_data.block0);

    mpi_stubs_data.nextfree = newh->next;             // Skip over using item 0
    newh->next = NULL;

    mpi_stubs_data.blocks[0] = mpi_stubs_data.block0;
    mpi_stubs_data.nblocks   = 1;

    for (int i=1; i<MPI_STUBS_MAX_BLOCKS; i++) { mpi_stubs_data.blocks[i] = NULL; }

    mpi_stubs_data.need_to_init = 0;
}


static void mpi_stubs_alloc_handle(int *handle, void **data)
{
    Handleitem *newh = NULL;
    int nblocks = mpi_stubs_data.nblocks;

    if (mpi_stubs_data.need_to_init) { init_handles(); }

    if (mpi_stubs_data.nextfree)
    {
        newh = mpi_stubs_data.nextfree;
        mpi_stubs_data.nextfree = mpi_stubs_data.nextfree->next;
        newh->next = NULL;

        *handle = newh->handle;
        *data   = &(newh->data);

        return;
    }

    /* there is nothing free, so allocate a newh block and add it
    * to mpi_stubs_data.blocks[]
    */

    if (nblocks == MPI_STUBS_MAX_BLOCKS)
    {
        fprintf(stderr,"%s:%d - allocate_handle: max %d active handles exceeded\n",
                __FILE__, __LINE__, MPI_STUBS_MAX_BLOCKS*MPI_STUBS_BLOCK_ITEMS);
        abort();
    }

    MC_MALLOC(mpi_stubs_data.blocks[nblocks], MPI_STUBS_BLOCK_ITEMS, Handleitem);

    newh = init_block(nblocks, mpi_stubs_data.blocks[nblocks]);

    mpi_stubs_data.nextfree = newh->next;
    newh->next = NULL;

    *handle = newh->handle;
    *data   = &(newh->data);

    mpi_stubs_data.nblocks++;  // DON'T FORGET THIS!!!!
}

static pList mpi_stubs_list_new()
{
    pList list = NULL;

    MC_MALLOC(list, 1, List);

    list->head  = NULL;
    list->tail  = NULL;
    list->count = 0;

    mpi_stubs_data.headcount++;
    return(list);
}



static MPI_Comm mpi_stubs_comm_new()
{
    MPI_Comm chandle;
    Comm *cptr;
    static int num = 0;

    mpi_stubs_alloc_handle(&chandle,(void **) &cptr);

    cptr->sendlist = mpi_stubs_list_new();
    cptr->recvlist = mpi_stubs_list_new();

    cptr->num = num++;
    cptr->name = NULL;

    return(chandle);
}

int mpiComm_split(MPI_Comm comm, int color, int key, MPI_Comm *newcomm)
{

    if (color == MPI_UNDEFINED)
    {
        *newcomm = MPI_COMM_NULL;
    }
    else
    {
        *newcomm = mpi_stubs_comm_new();
    }

    return(MPI_SUCCESS);
}




#endif  // end #else HAVE_MPI
