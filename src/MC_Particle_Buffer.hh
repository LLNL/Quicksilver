#ifndef MC_PARTICLE_BUFFER_INCLUDE
#define MC_PARTICLE_BUFFER_INCLUDE

#include "MC_Processor_Info.hh"
#include "MC_Base_Particle.hh"
#include "utilsMpi.hh"
#include <map>
#include <list>


// forward declarations
class MC_Particle;
class MonteCarlo;
class SendQueue;

//
//  Type Definitions
//
struct MC_MPI_Send_Mode
{
    public:
    enum Enum
    {
    Send,
    Ssend,
    Bsend,
    Rsend,
    Isend,
    Issend,
    Ibsend,
    Irsend
    };
};

struct particle_buffer_base_type
{
    int          processor;
    int          task_num;        // Used for creating tag for messages non master threads.
    int          num_particles;   // Number of particles in buffer.
    int          int_index;       // Next free space in int_data array
    int          float_index;     // Next free space in float_data array
    int          char_index;      // Next free space in char_data array
    uint64_t     length;          // Length in bytes of buffer
    int         *int_data;        // int data for particles
    double      *float_data;      // float data for particles
    char        *char_data;       // char data for particles
    MPI_Request  request_list;    // Request for the unbuffered data

    void Allocate(int buffer_size);
    void Reset_Offsets();
    void Initialize_Buffer();
    void Free_Memory();
};


class particle_buffer_task_class
{
 public:
    particle_buffer_task_class() : send_buffer(NULL), recv_buffer(NULL), extra_send_buffer() {}
    particle_buffer_base_type *send_buffer;                   // send particle buffers
    particle_buffer_base_type *recv_buffer;                   // recv particle buffers
    std::list<particle_buffer_base_type> extra_send_buffer;   // extra send buffers for non-blocking sends
};


class mcp_test_done_class
{
 public:
    mcp_test_done_class() { Zero_Out(); }
    int local_sent;
    int local_recv;
    int BlockingSum;

    int64_t non_blocking_send[2];
    int64_t non_blocking_sum[2];
    MPI_Request IallreduceRequest;

    void Get_Local_Gains_And_Losses(MonteCarlo *mcco, int64_t sent_recv[2]);
    void Post_Recv();
    void Zero_Out();
    void Reduce_Num_Sent_And_Recv(int64_t buf_sum[2]);
    void Free_Memory();
    bool ThisProcessorCommunicates(int rank = -1);
};


    //------------------------------------------------------------------------------------------------------------------
    //
    //------------------------------------------------------------------------------------------------------------------
struct MC_New_Test_Done_Method
{
public:
    enum Enum
        {
            AllProcessorTree,
            Blocking,
            NonBlocking
        };

};

class MC_Particle_Buffer
{
 private:

    MonteCarlo *mcco;
    mcp_test_done_class          test_done;
    particle_buffer_task_class  *task;                 // buffers for each task
    std::map<int, int>    processor_buffer_map; // Map processors to buffers. buffer_index = processor_buffer_map[processor]

    bool Trivially_Done();
    void Unpack_Particle_Buffer(int particle_vault_task_num, int recv_buff_task_num, int buffer_index, uint64_t &fill_vault);
    void Unpack_Particle_Buffer_Thread_Multiple(int particle_vault_task_num, int recv_buff_task_num, int buffer_index, uint64_t &fill_vault);
    void Instantiate();
    void Initialize_Map();
    void Delete_Completed_Extra_Send_Buffers(int task_num);


 public:
    // non-master threads place full buffers here for master thread to send
    // std::list<particle_buffer_base_type> thread_send_buffer_queue;

    bool Test_Done_New( MC_New_Test_Done_Method::Enum test_done_method = MC_New_Test_Done_Method::Blocking);
    MC_New_Test_Done_Method::Enum new_test_done_method; // which algorithm to use

    int  num_buffers;         // Number of particle buffers
    int  buffer_size;         // Buffer size to be sent.

    MC_Particle_Buffer(MonteCarlo *mcco_, size_t bufferSize_);       // constructor
    void Initialize();
    int  Get_Processor_Buffer_Index(int processor);
    void Buffer_Particle(MC_Particle *particle_to_buffer, int task_index, int buffer);
    void Buffer_Particle(MC_Base_Particle &particle_to_buffer, int task_index, int buffer);
    int  Choose_Buffer(int neighbor_rank, int task_num);
    void Send_Particle_Buffer(int task_num, int buffer);
    void Send_Particle_Buffers(int task_num);
    void Allocate_Send_Buffer( int task_index, SendQueue& sendQueue);
    void Free_Send_Buffer_Requests( int task_num );
    void Receive_Particle_Buffers(int particle_vault_task_num, uint64_t &fill_vault);
    void Post_Receive_Particle_Buffer( int particle_vault_task_num, size_t batchSize_ );
    void Cancel_Receive_Buffer_Requests( int particle_vault_task_num );
    bool Allreduce_ParticleCounts();
    bool Iallreduce_ParticleCounts();

    int Num_Parts_Recv();

    void Free_Buffers( int task_index );
    void Free_Memory();
private:
    MC_Particle_Buffer( const MC_Particle_Buffer& );                    // disable copy constructor
    MC_Particle_Buffer& operator=( const MC_Particle_Buffer& tmp );     // disable assignment operator

};

/*
  mcco->particle_buffer->test_done.local_sent
                                  .local_recv
                                  .BlockingSum;



                         num_buffers
                         task[]
                               .send_buffer[].processor
                                             .num_particles
                                             .int_index, float_index, char_index, length
                                             .int_data, float_data, char_data
                                             .request_list
                               .recv_buffer[].processor
                                             .num_particles
                                             .int_index, float_index, char_index, length
                                             .int_data, float_data, char_data
                                             .request_list
                         processor_buffer_map[]

 */
#endif
