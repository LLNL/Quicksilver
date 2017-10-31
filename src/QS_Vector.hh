#ifndef QS_VECTOR_HH
#define QS_VECTOR_HH

#include "DeclareMacro.hh"
#include "AtomicMacro.hh"
#include "qs_assert.hh"
#include "MemoryControl.hh"

#include <algorithm>

template <class T>
class qs_vector 
{
 public:

   qs_vector() : _data(0), _capacity(0), _size(0), _memPolicy(MemoryControl::AllocationPolicy::HOST_MEM), _isOpen(0) {};

   qs_vector(int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM )
   : _data(0), _capacity(size), _size(size), _memPolicy(memPolicy), _isOpen(0) 
   {
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }


   qs_vector( int size, const T& value, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM )
   : _data(0), _capacity(size), _size(size), _memPolicy(memPolicy), _isOpen(0) 
   { 
      _data = MemoryControl::allocate<T>(size, memPolicy);

      for (int ii = 0; ii < _capacity; ++ii)
         _data[ii] = value;
   }

   qs_vector(const qs_vector<T>& aa )
   : _data(0), _capacity(aa._capacity), _size(aa._size), _memPolicy(aa._memPolicy), _isOpen(aa._isOpen)
   {
      _data = MemoryControl::allocate<T>(_capacity, _memPolicy);
 
      for (int ii=0; ii<_size; ++ii)
         _data[ii] = aa._data[ii];
   }
   
   ~qs_vector()
   { 
      MemoryControl::deallocate(_data, _size, _memPolicy);
   }

   /// Needed for copy-swap idiom
   void swap(qs_vector<T>& other)
   {
      std::swap(_data,     other._data);
      std::swap(_capacity, other._capacity);
      std::swap(_size,     other._size);
      std::swap(_memPolicy, other._memPolicy);
      std::swap(_isOpen,   other._isOpen);
   }
   
   /// Implement assignment using copy-swap idiom
   qs_vector<T>& operator=(const qs_vector<T>& aa)
   {
      if (&aa != this)
      {
         qs_vector<T> temp(aa);
         this->swap(temp);
      }
      return *this;
   }
   
   HOST_DEVICE_CUDA
   int get_memPolicy()
   {
	return _memPolicy;
   }

   void push_back( const T& dataElem )
   {
      qs_assert( _isOpen );
      qs_assert( _size < _capacity );
      _data[_size] = dataElem;
      _size++;
   }

   void Open() { _isOpen = true;  }
   void Close(){ _isOpen = false; }

   HOST_DEVICE_CUDA
   const T& operator[]( int index ) const
   {
      qs_assert( index < _capacity ); 
      qs_assert( index >= 0);
      return _data[index];
   }

   HOST_DEVICE_CUDA
   T& operator[]( int index )
   {
      qs_assert( index < _capacity );
      qs_assert( index >= 0);
      return _data[index];
   }
   
   HOST_DEVICE_CUDA
   int capacity() const
   {
      return _capacity;
   }

   HOST_DEVICE_CUDA
   int size() const
   {
      return _size;
   }
   
   T& back()
   {
       qs_assert( _size > 0 );
      return _data[_size-1];
   }
   
   void reserve( int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM )
   {
      qs_assert( _capacity == 0 );
      _capacity = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }

   void resize( int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM )
   {
      qs_assert( _capacity == 0 );
      _capacity = size;
      _size = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }

   void resize( int size, const T& value, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM ) 
   { 
      qs_assert( _capacity == 0 );
      _capacity = size;
      _size = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);

      for (int ii = 0; ii < _capacity; ++ii)
         _data[ii] = value;
   }

   bool empty() const
   {
       return ( _size == 0 );
   }

   void eraseEnd( int NewEnd )
   {
       qs_assert( NewEnd <= _size );
       qs_assert( NewEnd >= 0 );
       _size = NewEnd;
   }

    void pop_back()
   {
       qs_assert(_size > 0);
       _size--;
   }

   void clear()
   {
       _size = 0;
   }

   void appendList( int listSize, T* list )
   {
       qs_assert( this->_size + listSize < this->_capacity );

       int size = _size;
       this->_size += listSize;

       for( int i = size; i < _size; i++ )
       {
           _data[i] = list[ i-size ];
       }

   }

   //Atomically retrieve an availible index then increment that index some amount
   HOST_DEVICE_CUDA
   int atomic_Index_Inc( int inc )
   {
       qs_assert(_size+inc <= _capacity);
       int pos;

       ATOMIC_CAPTURE( _size, inc, pos );

       return pos;
   }

 private:
   T* _data;
   int _capacity;
   int _size;
   bool _isOpen;
   MemoryControl::AllocationPolicy _memPolicy;

};


#endif
