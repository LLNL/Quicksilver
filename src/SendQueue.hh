#ifndef SENDQUEUE_HH
#define SENDQUEUE_HH

#include "QS_Vector.hh"
#include "DeclareMacro.hh"

//Tuple to record which particles need to be sent to which neighbor process during tracking
struct sendQueueTuple
{
    int _neighbor;
    int _particleIndex;
};

class SendQueue
{
  public:

    SendQueue();
    SendQueue( size_t size );

    //Get the total size of the send Queue
    size_t size();

    void reserve( size_t size ){ _data.reserve(size, VAR_MEM); }

    //get the number of items in send queue going to a specific neighbor
    size_t neighbor_size( int neighbor_ );

    sendQueueTuple& getTuple( int index_ );

    //Add items to the send queue in a kernel
    HOST_DEVICE_CUDA
    void push( int neighbor_, int vault_index_ );

    //Clear send queue before after use
    void clear();

  private:    

    //The send queue - stores particle index and neighbor index for any particles that hit (TRANSIT_OFF_PROCESSOR)
    qs_vector<sendQueueTuple> _data;

};

#endif
