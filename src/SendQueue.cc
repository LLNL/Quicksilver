#include "SendQueue.hh"
#include "QS_Vector.hh"

SendQueue::SendQueue()
{}

SendQueue::SendQueue( size_t size )
: _data( size, VAR_MEM )
{}


// -----------------------------------------------------------------------
size_t SendQueue::
size()
{
    return _data.size();
}
 
// -----------------------------------------------------------------------
size_t SendQueue::
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
HOST_DEVICE
void SendQueue::
push( int neighbor_, int vault_index_ )
{
    size_t indx = _data.atomic_Index_Inc(1);

    _data[indx]._neighbor    = neighbor_;
    _data[indx]._particleIndex = vault_index_;
}
HOST_DEVICE_END

// -----------------------------------------------------------------------
void SendQueue::
clear()
{
    _data.clear();
}

// -----------------------------------------------------------------------
sendQueueTuple& SendQueue::
getTuple( int index_ )
{
    qs_assert( index_ >= 0 );
    qs_assert( index_ < _data.size() );
    return _data[index_];
}

