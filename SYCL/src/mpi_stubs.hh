#ifndef MC_STUBS_MPI_H
#define MC_STUBS_MPI_H

#include <stdint.h>     // for uint64_t 

#define MPI_STUBS_MAX_BLOCKS (1024)
#define MPI_STUBS_BLOCK_ITEMS          (256)
#define MPI_STUBS_HANDLE_TO_BLOCK(x)   ( (x) >> 8)
#define MPI_STUBS_HANDLE_TO_INDEX(x)   ( (x) & 0xff )
#define MPI_STUBS_HANDLE(block,index)  ( (block << 8) | (index) )

//----------------------------------------------------------------------------------------------------------------------
// Defines for standard MPI definitions
//----------------------------------------------------------------------------------------------------------------------
#ifndef MPI_SUCCESS
#define MPI_SUCCESS          0      // Successful return code
#define MPI_FAILURE          1      // Generic failure for mpi stubs library
#define MPI_ERR_BUFFER       1      // Invalid buffer pointer
#define MPI_ERR_COUNT        2      // Invalid count argument
#define MPI_ERR_TYPE         3      // Invalid datatype argument
#define MPI_ERR_TAG          4      // Invalid tag argument
#define MPI_ERR_COMM         5      // Invalid communicator
#define MPI_ERR_RANK         6      // Invalid rank
#define MPI_ERR_ROOT         7      // Invalid root
#define MPI_ERR_GROUP        8      // Invalid group
#define MPI_ERR_OP           9      // Invalid operation
#define MPI_ERR_REQUEST     19      // Invalid mpi_request handle
#define MPI_ERR_TOPOLOGY    10      // Invalid topology
#define MPI_ERR_DIMS        11      // Invalid dimension argument
#define MPI_ERR_ARG         12      // Invalid argument
#define MPI_ERR_TRUNCATE    14      // Message truncated on receive
#define MPI_ERR_OTHER       15      // Other error; use Error_string
#define MPI_ERR_UNKNOWN     13      // Unknown error
#define MPI_ERR_INTERN      16      // Internal error code
#define MPI_ERR_IN_STATUS   17      // Look in status for error value
#define MPI_ERR_PENDING     18      // Pending request
#define MPI_ERR_CONVERSION  23      //
#define MPI_ERR_DUP_DATAREP 24      //
#define MPI_ERR_FILE        27      //
#define MPI_ERR_INFO        28      //
#define MPI_ERR_INFO_KEY    29      //
#define MPI_ERR_ACCESS      20      //
#define MPI_ERR_AMODE       21      //
#define MPI_ERR_BAD_FILE    22      //
#define MPI_ERR_FILE_EXISTS 25      //
#define MPI_ERR_FILE_IN_USE 26      //
#define MPI_ERR_IO          32      //
#define MPI_ERR_INFO_VALUE  30      //
#define MPI_ERR_INFO_NOKEY  31      //
#define MPI_ERR_NAME        33      //
#define MPI_ERR_NO_MEM      34      //
#define MPI_ERR_NOT_SAME    35      //
#define MPI_ERR_NO_SPACE    36      //
#define MPI_ERR_NO_SUCH_FILE 37     //
#define MPI_ERR_PORT        38      //
#define MPI_ERR_QUOTA       39      //
#define MPI_ERR_READ_ONLY   40      //
#define MPI_ERR_SERVICE     41      //
#define MPI_ERR_SPAWN       42      //
#define MPI_ERR_UNSUPPORTED_DATAREP   43 //
#define MPI_ERR_UNSUPPORTED_OPERATION 44 //
#define MPI_ERR_WIN         45      //
#define MPI_ERR_LASTCODE    0x3FFFFFFF      // Last error code
#endif // MPI_SUCCESS

#define MPI_GRAPH  1
#define MPI_CART   2

#define MPI_UNDEFINED     (-1)

#define MPI_BOTTOM      (void *)0

#define MPI_PROC_NULL   (-1)
#define MPI_ANY_SOURCE  (-2)
#define MPI_ROOT        (-3)
#define MPI_ANY_TAG     (-1)

#define MPI_COMM_WORLD (1)
#define MPI_COMM_SELF  (2)

#define MPI_STATUS_SIZE 4

//
//  For supported thread levels 
//
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3

typedef int MPI_Datatype ;
typedef int MPI_Comm ;
typedef int MPI_Request ;
typedef int MPI_Op ;
typedef int MPI_Group ;
typedef int MPI_Errhandler;

// User combination function
typedef void (MPI_User_function) ( void *, void *, int *, MPI_Datatype * );

// MPI Attribute copy and delete functions
typedef int (MPI_Copy_function) ( MPI_Comm, int, void *, void *, void *, int * );
typedef int (MPI_Delete_function) ( MPI_Comm, int, void *, void * );

#if 0
#define MPI_CHAR               ((MPI_Datatype)1)
#define MPI_UNSIGNED_CHAR      ((MPI_Datatype)2)
#define MPI_BYTE               ((MPI_Datatype)3)
#define MPI_SHORT              ((MPI_Datatype)4)
#define MPI_UNSIGNED_SHORT     ((MPI_Datatype)5)
#define MPI_INT                ((MPI_Datatype)6)
#define MPI_UNSIGNED           ((MPI_Datatype)7)
#define MPI_LONG               ((MPI_Datatype)8)
#define MPI_UNSIGNED_LONG      ((MPI_Datatype)9)
#define MPI_FLOAT              ((MPI_Datatype)10)
#define MPI_DOUBLE             ((MPI_Datatype)11)
#define MPI_LONG_DOUBLE        ((MPI_Datatype)12)
#define MPI_LONG_LONG_INT      ((MPI_Datatype)13)
#define MPI_LONG_LONG          ((MPI_Datatype)13)

#define MPI_PACKED             ((MPI_Datatype)14)
#define MPI_LB                 ((MPI_Datatype)15)
#define MPI_UB                 ((MPI_Datatype)16)
#define MPI_DOUBLE_INT         ((MPI_Datatype)18)

#define MPI_UNSIGNED_LONG_LONG ((MPI_Datatype)35)
#endif

#define MPI_MAX_ERROR_STRING   (512)
#define MPI_MAX_PROCESSOR_NAME (128)

// Define some null objects
#define MPI_COMM_NULL       ((MPI_Comm)0)
#define MPI_OP_NULL         ((MPI_Op)0)
#define MPI_DATATYPE_NULL   ((MPI_Datatype)0)
#define MPI_REQUEST_NULL    ((MPI_Request)0)
#define MPI_ERRHANDLER_NULL ((MPI_Errhandler )0)
#define MPI_GROUP_NULL      ((MPI_Group)0)

#define MPI_REQUEST_ONE     ((MPI_Request)1)

// Define some MPI groups
#define MPI_GROUP_ONE       ((MPI_Group)1)
#define MPI_GROUP_EMPTY     ((MPI_Group)-1)

#define MPI_ERRORS_ARE_FATAL ((MPI_Errhandler)119)
#define MPI_ERRORS_RETURN    ((MPI_Errhandler)120)
#define MPIR_ERRORS_WARN     ((MPI_Errhandler)121)

//----------------------------------------------------------------------------------------------------------------------
// Standard MPI prototypes.
//----------------------------------------------------------------------------------------------------------------------
int  MPI_Send(void*, int, MPI_Datatype, int, int, MPI_Comm);
int  MPI_Recv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
int  MPI_Get_count(MPI_Status *, MPI_Datatype, int *);
int  MPI_Bsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
int  MPI_Ssend(void*, int, MPI_Datatype, int, int, MPI_Comm);
int  MPI_Rsend(void*, int, MPI_Datatype, int, int, MPI_Comm);
int  MPI_Buffer_attach( void*, int);
int  MPI_Buffer_detach( void*, int*);
int  MPI_Errhandler_set(MPI_Comm, MPI_Errhandler);
int  MPI_Isend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Ibsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Issend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Irsend(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Irecv(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Wait(MPI_Request *, MPI_Status *);
int  MPI_Test(MPI_Request *, int *, MPI_Status *);
int  MPI_Request_free(MPI_Request *);
int  MPI_Waitany(int, MPI_Request *, int *, MPI_Status *);
int  MPI_Testany(int, MPI_Request *, int *, int *, MPI_Status *);
int  MPI_Waitall(int, MPI_Request *, MPI_Status *);
int  MPI_Testall(int, MPI_Request *, int *, MPI_Status *);
int  MPI_Waitsome(int, MPI_Request *, int *, int *, MPI_Status *);
int  MPI_Testsome(int, MPI_Request *, int *, int *, MPI_Status *);
int  MPI_Iprobe(int, int, MPI_Comm, int *flag, MPI_Status *);
int  MPI_Probe(int, int, MPI_Comm, MPI_Status *);
int  MPI_Cancel(MPI_Request *);
int  MPI_Test_cancelled(MPI_Status *, int *);
int  MPI_Send_init(void*, int, MPI_Datatype, int, int, MPI_Comm, MPI_Request *);
int  MPI_Bsend_init(void*, int, MPI_Datatype, int,int, MPI_Comm, MPI_Request *);
int  MPI_Ssend_init(void*, int, MPI_Datatype, int,int, MPI_Comm, MPI_Request *);
int  MPI_Rsend_init(void*, int, MPI_Datatype, int,int, MPI_Comm, MPI_Request *);
int  MPI_Recv_init(void*, int, MPI_Datatype, int,int, MPI_Comm, MPI_Request *);
int  MPI_Start(MPI_Request *);
int  MPI_Startall(int, MPI_Request *);
int  MPI_Sendrecv(void *, int, MPI_Datatype,int, int, void *, int, MPI_Datatype, int, int, MPI_Comm, MPI_Status *);
int  MPI_Sendrecv_replace(void*, int, MPI_Datatype, int, int, int, int, MPI_Comm, MPI_Status *);
int  MPI_Type_contiguous(int, MPI_Datatype, MPI_Datatype *);
int  MPI_Type_vector(int, int, int, MPI_Datatype, MPI_Datatype *);
int  MPI_Type_indexed(int, int *, int *, MPI_Datatype, MPI_Datatype *);
int  MPI_Type_size(MPI_Datatype, int *);
int  MPI_Type_commit(MPI_Datatype *);
int  MPI_Type_free(MPI_Datatype *);
int  MPI_Get_elements(MPI_Status *, MPI_Datatype, int *);
int  MPI_Pack(void*, int, MPI_Datatype, void *, int, int *,  MPI_Comm);
int  MPI_Unpack(void*, int, int *, void *, int, MPI_Datatype, MPI_Comm);
int  MPI_Pack_size(int, MPI_Datatype, MPI_Comm, int *);
int  MPI_Barrier(MPI_Comm );
int  MPI_Bcast(void*, int, MPI_Datatype, int, MPI_Comm );
int  MPI_Gather(void* , int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm); 
int  MPI_Gatherv(void* , int, MPI_Datatype, void*, int *, int *, MPI_Datatype, int, MPI_Comm); 
int  MPI_Scatter(void* , int, MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
int  MPI_Scatterv(void* , int *, int *,  MPI_Datatype, void*, int, MPI_Datatype, int, MPI_Comm);
int  MPI_Allgather(void* , int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
int  MPI_Allgatherv(void* , int, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
int  MPI_Alltoall(void* , int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
int  MPI_Alltoallv(void* , int *, int *, MPI_Datatype, void*, int *, int *, MPI_Datatype, MPI_Comm);
int  MPI_Reduce(void* , void*, int, MPI_Datatype, MPI_Op, int, MPI_Comm);
int  MPI_Op_create(MPI_User_function *, int, MPI_Op *);
int  MPI_Op_free( MPI_Op *);
int  MPI_Allreduce(void* , void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
int  MPI_Reduce_scatter(void* , void*, int *, MPI_Datatype, MPI_Op, MPI_Comm);
int  MPI_Scan(void* , void*, int, MPI_Datatype, MPI_Op, MPI_Comm );
int  MPI_Group_size(MPI_Group group, int *);
int  MPI_Group_rank(MPI_Group group, int *);
int  MPI_Group_translate_ranks (MPI_Group, int, int *, MPI_Group, int *);
int  MPI_Group_compare(MPI_Group, MPI_Group, int *);
int  MPI_Comm_group(MPI_Comm, MPI_Group *);
int  MPI_Group_union(MPI_Group, MPI_Group, MPI_Group *);
int  MPI_Group_intersection(MPI_Group, MPI_Group, MPI_Group *);
int  MPI_Group_difference(MPI_Group, MPI_Group, MPI_Group *);
int  MPI_Group_incl(MPI_Group group, int, int *, MPI_Group *);
int  MPI_Group_excl(MPI_Group group, int, int *, MPI_Group *);
int  MPI_Group_range_incl(MPI_Group group, int, int [][3], MPI_Group *);
int  MPI_Group_range_excl(MPI_Group group, int, int [][3], MPI_Group *);
int  MPI_Group_free(MPI_Group *);
int  MPI_Comm_size(MPI_Comm, int *);
int  MPI_Comm_rank(MPI_Comm, int *);
int  MPI_Comm_compare(MPI_Comm, MPI_Comm, int *);
int  MPI_Comm_dup(MPI_Comm, MPI_Comm *);
int  MPI_Comm_create(MPI_Comm, MPI_Group, MPI_Comm *);
int  MPI_Comm_split(MPI_Comm, int, int, MPI_Comm *);
int  MPI_Comm_free(MPI_Comm *);
int  MPI_Comm_test_inter(MPI_Comm, int *);
int  MPI_Comm_remote_size(MPI_Comm, int *);
int  MPI_Comm_remote_group(MPI_Comm, MPI_Group *);
int  MPI_Intercomm_create(MPI_Comm, int, MPI_Comm, int, int, MPI_Comm * );
int  MPI_Intercomm_merge(MPI_Comm, int, MPI_Comm *);
int  MPI_Keyval_create(MPI_Copy_function *, MPI_Delete_function *, int *, void*);
int  MPI_Keyval_free(int *);
int  MPI_Attr_put(MPI_Comm, int, void*);
int  MPI_Attr_get(MPI_Comm, int, void *, int *);
int  MPI_Attr_delete(MPI_Comm, int);
int  MPI_Topo_test(MPI_Comm, int *);
int  MPI_Cart_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
int  MPI_Dims_create(int, int, int *);
int  MPI_Graph_create(MPI_Comm, int, int *, int *, int, MPI_Comm *);
int  MPI_Graphdims_get(MPI_Comm, int *, int *);
int  MPI_Graph_get(MPI_Comm, int, int, int *, int *);
int  MPI_Cartdim_get(MPI_Comm, int *);
int  MPI_Cart_get(MPI_Comm, int, int *, int *, int *);
int  MPI_Cart_rank(MPI_Comm, int *, int *);
int  MPI_Cart_coords(MPI_Comm, int, int, int *);
int  MPI_Graph_neighbors_count(MPI_Comm, int, int *);
int  MPI_Graph_neighbors(MPI_Comm, int, int, int *);
int  MPI_Cart_shift(MPI_Comm, int, int, int *, int *);
int  MPI_Cart_sub(MPI_Comm, int *, MPI_Comm *);
int  MPI_Cart_map(MPI_Comm, int, int *, int *, int *);
int  MPI_Graph_map(MPI_Comm, int, int *, int *, int *);
int  MPI_Get_processor_name(char *, int *);
int  MPI_Get_version(int *, int *);
int  MPI_Error_string(int, char *, int *);
int  MPI_Error_class(int, int *);
double  MPI_Wtime();
double  MPI_Wtick();
int  MPI_Init(int *, char ***);
int  MPI_Init_thread( int *, char ***, int, int * );
int  MPI_Finalize();
int  MPI_Initialized(int *);
int  MPI_Abort(MPI_Comm, int);
int  MPI_Comm_set_name(MPI_Comm, char *);
int  MPI_Comm_get_name(MPI_Comm, char *, int *);

int  MPI_NULL_COPY_FN ( MPI_Comm, int, void *, void *, void *, int * );
int  MPI_NULL_DELETE_FN ( MPI_Comm, int, void *, void * );
int  MPI_DUP_FN ( MPI_Comm, int, void *, void *, void *, int * );


#endif      // ifndef MPI_STUBS_H
