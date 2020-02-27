/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


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
    HOST_DEVICE_HIP
    void push( int neighbor_, int vault_index_ );

    //Clear send queue before after use
    void clear();

  private:    

    //The send queue - stores particle index and neighbor index for any particles that hit (TRANSIT_OFF_PROCESSOR)
    qs_vector<sendQueueTuple> _data;

};


inline SendQueue::SendQueue()
{}

inline SendQueue::SendQueue( size_t size )
: _data( size, VAR_MEM )
{}


// -----------------------------------------------------------------------
inline size_t SendQueue::
size()
{
    return _data.size();
}

// -----------------------------------------------------------------------
inline size_t SendQueue::
neighbor_size( int neighbor_ )
{
    size_t sum_n=0;
    for( size_t i = 0; i < _data.size(); i++ )
    {
        if( neighbor_ == _data[i]._neighbor )
            sum_n++;
    }
    return sum_n;
}

// -----------------------------------------------------------------------
inline HOST_DEVICE
void SendQueue::
push( int neighbor_, int vault_index_ )
{
    size_t indx = _data.atomic_Index_Inc(1);

    _data[indx]._neighbor    = neighbor_;
    _data[indx]._particleIndex = vault_index_;
}
HOST_DEVICE_END

// -----------------------------------------------------------------------
inline void SendQueue::
clear()
{
    _data.clear();
}

// -----------------------------------------------------------------------
inline sendQueueTuple& SendQueue::
getTuple( int index_ )
{
    qs_assert( index_ >= 0 );
    qs_assert( index_ < _data.size() );
    return _data[index_];
}



#endif
