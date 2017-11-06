#include "MC_Particle_Buffer.hh"
#include <time.h>
#include "utilsMpi.hh"
#include "ParticleVaultContainer.hh"
#include "SendQueue.hh"
#include "MCT.hh"
#include "MC_Processor_Info.hh"
#include "MC_Base_Particle.hh"
#include "Tallies.hh"
#include "MonteCarlo.hh"
#include "Globals.hh"
#include "MC_Fast_Timer.hh"
#include "macros.hh"
#include "NVTX_Range.hh"

static const int MC_Tag_Particle_Buffer = 2300;

// Static declarations
static std::map<int, int> send_count;
static std::map<int, int> recv_count;

//----------------------------------------------------------------------------------------------------------------------
//  Cancels and frees a pending request.
//----------------------------------------------------------------------------------------------------------------------
void MCP_Cancel_Request(MPI_Request *request)
{
    if (request[0] != MPI_REQUEST_NULL)
    {
        MPI_Status status;
        int flag;
        mpiCancel(request);
        mpiWait(request, &status);
        mpiTest_cancelled(&status, &flag);
        if ( !flag ) // cancel did not succeed
        {
            qs_assert(false);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Test a pending MPI request for completion.
//----------------------------------------------------------------------------------------------------------------------
int MCP_Test(MPI_Request *request)
{
    // Test that the request has completed
    int flag = 0;
    mpiTest(request, &flag, MPI_STATUS_IGNORE);
    return flag;
}

//
//  particle_buffer_base_type
//

//----------------------------------------------------------------------------------------------------------------------
//  Allocate a contiguous particle buffer, set pointers to int, float and char data.
//----------------------------------------------------------------------------------------------------------------------
void particle_buffer_base_type::Allocate(int buffer_size)
{
    // we add 2 ints: 1 for the number of particles and the second int is so the float_data
    // buffer will be 8 byte aligned

    uint64_t length_int_data   = (MC_Base_Particle::num_base_ints   * buffer_size + 2) * (int)sizeof(int);
    uint64_t length_float_data = (MC_Base_Particle::num_base_floats * buffer_size    ) * (int)sizeof(double);
    uint64_t length_char_data  = (MC_Base_Particle::num_base_chars  * buffer_size    ) * (int)sizeof(char);

    this->length = length_int_data + length_float_data + length_char_data;

    // single, contiguous allocation for all 3 int, float, char data buffers
    char *p = NULL;
    MC_MALLOC(p, this->length, char);

    if ( length_int_data % sizeof(double) != 0 )
    {
        MC_Fatal_Jump( "\nThe particle buffer for floating point data is not 8-byte alligned.\n"
                         "This means buffer_size %i is not even\n", buffer_size);
    }

    this->int_data   = (int *)p;
    this->float_data = (double *)(p + length_int_data);
    this->char_data  = p + length_int_data + length_float_data;

    // Initialize int_index to 2 so can store num_particles in index 0, nothing at index 1.

    this->int_index     = 2;
    this->float_index   = 0;
    this->char_index    = 0;
    this->num_particles = 0;
}

//----------------------------------------------------------------------------------------------------------------------
//  Initialize the data members of the particle buffer.
//----------------------------------------------------------------------------------------------------------------------
void particle_buffer_base_type::Initialize_Buffer()
{
    this->num_particles = 0;
    this->length        = 0;

    // int_index is 2 because num_particles goes in 0 position when the buffer is sent,
    // and position 1 is blank so float_data is 8 byte aligned.
    this->int_index     = 2;
    this->float_index   = 0;
    this->char_index    = 0;
    this->int_data      = NULL;   // Initialize to NULL so can be Allocated in Buffer_Particle
    this->float_data    = NULL;
    this->char_data     = NULL;
    this->request_list  = MPI_REQUEST_NULL;
}

//----------------------------------------------------------------------------------------------------------------------
//  Reset the float and char data offsets based on num_particles
//----------------------------------------------------------------------------------------------------------------------
void particle_buffer_base_type::Reset_Offsets()
{

    uint64_t length_int_data   = (MC_Base_Particle::num_base_ints   * num_particles + 2) * (int)sizeof(int);
    uint64_t length_float_data = (MC_Base_Particle::num_base_floats * num_particles    ) * (int)sizeof(double);

    char* p = (char *)int_data;

    this->float_data = (double *)(p + length_int_data);
    this->char_data  = p + length_int_data + length_float_data;
}

//----------------------------------------------------------------------------------------------------------------------
//  Free the memory for this particle buffer.
//----------------------------------------------------------------------------------------------------------------------
void particle_buffer_base_type::Free_Memory()
{
    mpiWait(&this->request_list, MPI_STATUS_IGNORE);

    MC_FREE(this->int_data);
    this->float_data = NULL;
    this->char_data  = NULL;
}

//
//  mcp_test_done_class
//

//----------------------------------------------------------------------------------------------------------------------
//  Reset to 0 and clear all vectors.
//----------------------------------------------------------------------------------------------------------------------
void mcp_test_done_class::Zero_Out()
{

    this->local_sent = 0;
    this->local_recv = 0;

    send_count.clear();
    recv_count.clear();


    this->BlockingSum = 0;

    this->non_blocking_send[0] = 0;
    this->non_blocking_send[1] = 0;
    this->non_blocking_sum[0]  = 0;
    this->non_blocking_sum[1]  = 1; // initialize these so they are not-equal, [0] != [1]

    this->IallreduceRequest = MPI_REQUEST_NULL;
}

//----------------------------------------------------------------------------------------------------------------------
//  Free the memory and cleanup communication.
//----------------------------------------------------------------------------------------------------------------------
void mcp_test_done_class::Free_Memory()
{
    this->Zero_Out();
}

//----------------------------------------------------------------------------------------------------------------------
//  Get the number of particles created and completed.
//----------------------------------------------------------------------------------------------------------------------
void mcp_test_done_class::Get_Local_Gains_And_Losses(MonteCarlo *monteCarlo, int64_t sent_recv[2])
{
    uint64_t gains = 0, losses = 0;

    Balance &bal = monteCarlo->_tallies->_balanceTask[0]; // SumTasks has been called, so just use index 0

    gains   = bal._start  + bal._source + bal._produce + bal._split;
    losses  = bal._absorb + bal._census + bal._escape  + bal._rr;
    losses += bal._fission; 

    sent_recv[0] = gains;
    sent_recv[1] = losses;
}

//
// MC_Particle_Buffer :: PRIVATE Functions
//

//----------------------------------------------------------------------------------------------------------------------
//  Initializes the particle buffers, mallocing them and assigning processors to buffers.
//
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Instantiate()
{
    this->test_done.Zero_Out();

    MC_NEW_ARRAY(this->task, 1, particle_buffer_task_class);

    this->Initialize_Map();

    this->task[0].extra_send_buffer.clear();

    MC_CALLOC(this->task[0].send_buffer, this->num_buffers, particle_buffer_base_type);
    MC_CALLOC(this->task[0].recv_buffer, this->num_buffers, particle_buffer_base_type);

    for ( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
    {
        this->task[0].send_buffer[buffer_index].Initialize_Buffer();
        this->task[0].recv_buffer[buffer_index].Initialize_Buffer();
    }

    for ( std::map<int,int>::iterator it = this->processor_buffer_map.begin();
                                      it !=this->processor_buffer_map.end(); it++ )
    {
        int buffer    = (*it).second;
        int processor = (*it).first;
        this->task[0].send_buffer[buffer].processor = processor;
        this->task[0].recv_buffer[buffer].processor = processor;
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Define this->processor_buffer_map[...].
//
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Initialize_Map()
{
    this->num_buffers = 0;

    // Determine number of buffers needed and assign processors to buffers
    for ( int domain_index = 0; domain_index < mcco->domain.size(); domain_index++ )
    {
        MC_Domain &domain = mcco->domain[domain_index];
        for ( int neighbor_index = 0; neighbor_index < domain.mesh._nbrRank.size(); neighbor_index++ )
        {
            int neighbor_rank = domain.mesh._nbrRank[neighbor_index];

            // If neighbor is not on same processor
            if ( neighbor_rank != mcco->processor_info->rank )
            {
                if ( this->Get_Processor_Buffer_Index(neighbor_rank) == -1 )
                {
                    this->processor_buffer_map[neighbor_rank] = this->num_buffers++;
                }
            }
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Unpack a particle buffer that was just received.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Unpack_Particle_Buffer(int buffer_index, uint64_t &fill_vault)
{
    MC_Base_Particle base_particle;

    int int_index   = 0;
    int float_index = 0;
    int char_index  = 0;

    particle_buffer_base_type &recv_buffer = this->task[0].recv_buffer[buffer_index];

    // Get the number of particles in the buffer
    recv_buffer.num_particles = recv_buffer.int_data[int_index++];
    int_index++; // increment past the second integer, (for 8 byte allignment of float_data)

    recv_buffer.Reset_Offsets();

    if (mcco->_params.simulationParams.debugThreads >= 2)
    {
        fprintf(stderr,"%02d-%02d <- %02d %3d particles MC_Particle_Buffer::Unpack_Particle_Buffer into vault %d\n",
                        mcco->processor_info->rank ,omp_get_thread_num(), recv_buffer.processor,
                        recv_buffer.num_particles, 0);
    }

    // Unpack each particle and place into a partivault.
    for ( int particle_index = 0; particle_index < recv_buffer.num_particles; particle_index++)
    {
        base_particle.Serialize(recv_buffer.int_data, recv_buffer.float_data, recv_buffer.char_data,
                                int_index, float_index, char_index, MC_Data_Member_Operation::Unpack);

        base_particle.last_event = MC_Tally_Event::Facet_Crossing_Communication;

        mcco->_particleVaultContainer->addProcessingParticle(base_particle, fill_vault);
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Trivially_Done
//      Do we have more than 1 processor running
//      Do we have any particles to process locally 
//----------------------------------------------------------------------------------------------------------------------
bool MC_Particle_Buffer::Trivially_Done()
{
    if (mcco->processor_info->num_processors > 1) 
    {
        return false;
    }

    uint64_t processingSize = mcco->_particleVaultContainer->sizeProcessing();
    if( processingSize == 0 )
    {
        return true;
    }

    return false;
}

//----------------------------------------------------------------------------------------------------------------------
//  Delete the extra send buffers which have completed the send.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Delete_Completed_Extra_Send_Buffers()
{
    std::list<particle_buffer_base_type>::iterator it = this->task[0].extra_send_buffer.begin();
    while ( it != this->task[0].extra_send_buffer.end() )
    {
        int flag = MCP_Test(&it->request_list);
        if ( flag )
        {
            mpiWait(&it->request_list, MPI_STATUS_IGNORE);
            it->Free_Memory();

            it = this->task[0].extra_send_buffer.erase(it);
        }
        else
        {
            ++it;
        }
    }
}

//
// MC_Particle_Buffer :: PUBLIC Functions
// 

//----------------------------------------------------------------------------------------------------------------------
//  Constructor.
//----------------------------------------------------------------------------------------------------------------------
MC_Particle_Buffer::MC_Particle_Buffer(MonteCarlo *mcco_, size_t bufferSize_)
{
    this->mcco  = mcco_;
#ifdef HAVE_ASYNC_MPI
    this->new_test_done_method = MC_New_Test_Done_Method::NonBlocking;
#else
    this->new_test_done_method = MC_New_Test_Done_Method::Blocking;
#endif

    this->test_done.Zero_Out();

    this->num_buffers = 0;
    this->task        = NULL;
    this->buffer_size = bufferSize_;
    this->processor_buffer_map.clear();
}

//----------------------------------------------------------------------------------------------------------------------
//  Initializes the particle buffers, mallocing them and assigning processors to buffers.
//
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Initialize()
{
    NVTX_Range range("MC_Particle_Buffer::Initialize");

    if (mcco->processor_info->num_processors > 1) 
    {
        this->Instantiate();

        mpiBarrier(mcco->processor_info->comm_mc_world);
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Selects a  particle buffer.
//----------------------------------------------------------------------------------------------------------------------
int MC_Particle_Buffer::Choose_Buffer(int processor)
{
    int buffer    = this->Get_Processor_Buffer_Index(processor);

    if ( buffer < 0 || buffer >= this->num_buffers )
    {
        MC_Fatal_Jump( "Bad buffer value (buffer = %i) for neighbor.next_send %i, neighbor.replication_level %i, processor %i\n",
                         buffer, neighbor.task[0].next_send, neighbor.replication_level, processor);
    }

    return buffer;
}

//----------------------------------------------------------------------------------------------------------------------
//  Given the processor rank, return the particle buffer index.
//----------------------------------------------------------------------------------------------------------------------
int MC_Particle_Buffer::Get_Processor_Buffer_Index(int processor)
{
    std::map<int,int>::iterator it = this->processor_buffer_map.find(processor);

    if ( it == this->processor_buffer_map.end() )
    {
        // return -1 if the input processor does not communicate with this processor. i.e. not found in map.
        return -1;
    }
    else
    {
        return (*it).second;
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Wrapper routine converts MC_Particle to MC_Base_Particle and calls base Particle version
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Buffer_Particle(MC_Particle *particle, int buffer)
{
    MC_Base_Particle base_particle(*particle);
    this->Buffer_Particle(base_particle, buffer);
}

//----------------------------------------------------------------------------------------------------------------------
//  Stores the particle into a particle buffer.  
//  If the buffer becomes to full error out - no buffer flush mode.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Buffer_Particle(MC_Base_Particle &particle, int buffer)
{
    particle_buffer_base_type &send_buffer = this->task[0].send_buffer[buffer];

    if (mcco->_params.simulationParams.debugThreads >= 3)
    { 
        fprintf(stderr,"%02d-%02d MC_Particle_Buffer::Buffer_Particle entered task_index=%d buffer_size=%d "
                "buffer.num_particles=%d\n",mcco->processor_info->rank ,omp_get_thread_num(),0, buffer_size, 
                send_buffer.num_particles); 
    }

    if ( send_buffer.int_data == NULL )
    {
        fprintf( stderr, "Should not reach here. This should be already preallocated\n" );
        send_buffer.Allocate(this->buffer_size);
    }

    // Put the particle into the buffer.
    particle.Serialize(send_buffer.int_data,  send_buffer.float_data,  send_buffer.char_data,
                       send_buffer.int_index, send_buffer.float_index, send_buffer.char_index,
                       MC_Data_Member_Operation::Pack);

    send_buffer.num_particles++;
}

//----------------------------------------------------------------------------------------------------------------------
//  Allocate Send Buffers given sendQueue neighbor size
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Allocate_Send_Buffer( SendQueue &sendQueue )
{
    for( int buffer = 0; buffer < this->num_buffers; buffer++ )
    {
        particle_buffer_base_type &send_buffer = this->task[0].send_buffer[buffer];
        int send_size = sendQueue.neighbor_size(send_buffer.processor); 
        send_buffer.Free_Memory();
        send_buffer.Allocate(send_size);
    }
}

//----------------------------------------------------------------------------------------------------------------------
// Wrapper to send all of the particle buffers
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Send_Particle_Buffers( )
{
    for( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
    {
        Send_Particle_Buffer( buffer_index );
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Sends a particle buffer to the appropriate processor.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Send_Particle_Buffer(int buffer)
{
    particle_buffer_base_type &send_buffer = this->task[0].send_buffer[buffer];

    if( send_buffer.num_particles > 0 )
    {
        // Fill in the number of particles being sent.
        send_buffer.int_data[0] = send_buffer.num_particles;
        send_buffer.int_data[1] = 0; //Padding

        if (mcco->_params.simulationParams.debugThreads >= 2)
        {
            fprintf(stderr,"%02d-%02d -> %02d %3d particles MC_Particle_Buffer::Send_Particle_Buffer\n",
                    mcco->processor_info->rank ,omp_get_thread_num(), send_buffer.processor,
                    send_buffer.num_particles);
        }


        mpiIsend(send_buffer.int_data, send_buffer.length, MPI_BYTE, send_buffer.processor,
                 MC_Tag_Particle_Buffer, mcco->processor_info->comm_mc_world,
                 &send_buffer.request_list);

        // non-blocking send, copy send_buffer to the extra list, so we can re-use send_buffer
        this->task[0].extra_send_buffer.push_back(send_buffer);
        send_buffer.Initialize_Buffer();
        this->Delete_Completed_Extra_Send_Buffers();
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Receive the size of the next message you are expecting
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Post_Receive_Particle_Buffer( size_t bufferSize_ )
{
    for ( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
    {
        particle_buffer_base_type &recv_buffer = this->task[0].recv_buffer[buffer_index];

        recv_buffer.Allocate(bufferSize_); 

        //Posting the Irecv Buffers
        mpiIrecv(recv_buffer.int_data, recv_buffer.length, MPI_BYTE, recv_buffer.processor,          
                 MC_Tag_Particle_Buffer, 
                 mcco->processor_info->comm_mc_world, &recv_buffer.request_list);
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Receives a particle buffer and puts the particles in it into particle vault to be processed.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Receive_Particle_Buffers(uint64_t &fill_vault)
{
    for ( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
    {
        particle_buffer_base_type &recv_buffer = this->task[0].recv_buffer[buffer_index];
        int flag = MCP_Test(&recv_buffer.request_list);
        if( flag )
        {
            this->Unpack_Particle_Buffer(buffer_index, fill_vault);

            // Reset the number of particles.
            recv_buffer.num_particles = 0;

            recv_count[MC_Tag_Particle_Buffer]++;

            mpiIrecv(recv_buffer.int_data, recv_buffer.length, MPI_BYTE, recv_buffer.processor,
                     MC_Tag_Particle_Buffer,
                     mcco->processor_info->comm_mc_world, &recv_buffer.request_list);
        }
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Cancels all pending irecv requests
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Cancel_Receive_Buffer_Requests()
{
    for ( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
    {
        particle_buffer_base_type &recv_buffer = this->task[0].recv_buffer[buffer_index];
        mpiCancel( &recv_buffer.request_list );
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Test to see if we are done with streaming communication.
//----------------------------------------------------------------------------------------------------------------------
bool MC_Particle_Buffer::Test_Done_New( MC_New_Test_Done_Method::Enum test_done_method )
{
    if ( !(mcco->processor_info->num_processors > 1 ))
    {
        return this->Trivially_Done();
    }

    MC_VERIFY_THREAD_ZERO

    MC_FASTTIMER_START(MC_Fast_Timer::cycleTracking_Test_Done);

    mcco->_tallies->SumTasks();

    if ( test_done_method == MC_New_Test_Done_Method::Blocking )
    {

        // brain dead test for done, that is synchronized, does an allreduce.
        bool answer = this->Allreduce_ParticleCounts();

        MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Test_Done);
        return answer;
    }
    else if ( test_done_method == MC_New_Test_Done_Method::NonBlocking )
    {
        bool answer = this->Iallreduce_ParticleCounts();
        MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Test_Done);
        return answer;
    }

    MC_FASTTIMER_STOP(MC_Fast_Timer::cycleTracking_Test_Done);

    return true;
}

//----------------------------------------------------------------------------------------------------------------------
//  Perform blocking allreduce on particle counts, return true if counts are equal.
//----------------------------------------------------------------------------------------------------------------------
bool MC_Particle_Buffer::Allreduce_ParticleCounts()
{
    int64_t buf[2];
    int64_t hard_blocking_sum[2] = {0, 0};

    this->test_done.Get_Local_Gains_And_Losses(mcco, buf);

    mpiAllreduce(buf, hard_blocking_sum, 2, MPI_INT64_T, MPI_SUM, mcco->processor_info->comm_mc_world);

#if 0
    if (hard_blocking_sum[0] == hard_blocking_sum[1])
        fprintf(stderr,"DEBUGT %d:%d %s:%d Allreduce_ParticleCounts gains=%d loss=%d\n",
                mcco->processor_info->rank,omp_get_thread_num(),
                __FILE__,__LINE__,hard_blocking_sum[0],hard_blocking_sum[1]);
#endif

    return ( hard_blocking_sum[0] == hard_blocking_sum[1] );
}

//----------------------------------------------------------------------------------------------------------------------
//  Perform non-blocking allreduce on particle counts, return true if local counts are equal.
//----------------------------------------------------------------------------------------------------------------------
bool MC_Particle_Buffer::Iallreduce_ParticleCounts()
{
    // non blocking allreduce request
    int flag = MCP_Test(&this->test_done.IallreduceRequest);

    if ( flag )
    {
        if( this->test_done.non_blocking_sum[0] == this->test_done.non_blocking_sum[1] )
        {
            bool answer = this->Allreduce_ParticleCounts();
            return answer;
        }
        else
        {
            this->test_done.Get_Local_Gains_And_Losses(mcco, this->test_done.non_blocking_send);
            mpiIAllreduce(this->test_done.non_blocking_send, this->test_done.non_blocking_sum, 
                          2, MPI_INT64_T, MPI_SUM, mcco->processor_info->comm_mc_world, 
                          &this->test_done.IallreduceRequest);
        }
    }
    return false;
}


//----------------------------------------------------------------------------------------------------------------------
//  Free Buffers to allow realocation
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Free_Buffers()
{
    for( int buffer = 0; buffer < this->num_buffers; buffer++ )
    {
        particle_buffer_base_type &send_buffer = this->task[0].send_buffer[buffer];
        particle_buffer_base_type &recv_buffer = this->task[0].recv_buffer[buffer];
        send_buffer.Free_Memory();
        recv_buffer.Free_Memory();
    }
}

//----------------------------------------------------------------------------------------------------------------------
//  Free the particle buffer memory.
//----------------------------------------------------------------------------------------------------------------------
void MC_Particle_Buffer::Free_Memory()
{
    MC_VERIFY_THREAD_ZERO;

    // Free the send and recv buffers for each task.
    // Cancel outstanding pre-posted receives for each task
    if ( this->task )
    {
        particle_buffer_task_class &mytask = this->task[0];
        for ( int buffer_index = 0; buffer_index < this->num_buffers; buffer_index++ )
        {
            MCP_Cancel_Request(&mytask.recv_buffer[buffer_index].request_list);
            recv_count[MC_Tag_Particle_Buffer]--;

            mytask.send_buffer[buffer_index].Free_Memory();
            mytask.recv_buffer[buffer_index].Free_Memory();
        }
    }

    if ( this->task )
    {
        particle_buffer_task_class &mytask = this->task[0];

        // Clean up the Isend requests
        for ( std::list<particle_buffer_base_type>::iterator it = mytask.extra_send_buffer.begin();
              it != mytask.extra_send_buffer.end(); ++it )
        {
            mpiWait(&it->request_list, MPI_STATUS_IGNORE);
            it->Free_Memory();
        }
        mytask.extra_send_buffer.clear();

        MC_FREE(mytask.send_buffer);
        MC_FREE(mytask.recv_buffer);
    }

    this->num_buffers = 0;      // buffers are now freed

    MC_DELETE_ARRAY(task);
    this->processor_buffer_map.clear();
    this->test_done.Free_Memory();
}

