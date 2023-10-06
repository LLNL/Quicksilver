/*
Modifications Copyright (C) 2023 Intel Corporation

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


SPDX-License-Identifier: BSD-3-Clause
*/

/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

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

   qs_vector() : _data(0), _capacity(0), _size(0), _memPolicy(MemoryControl::AllocationPolicy::HOST_MEM), _isOpen(0){};

   qs_vector(int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM)
   : _data(0), _capacity(size), _size(size), _memPolicy(memPolicy), _isOpen(0) 
   {
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }


   qs_vector(int size, const T& value, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM)
   : _data(0), _capacity(size), _size(size), _memPolicy(memPolicy), _isOpen(0) 
   { 
      _data = MemoryControl::allocate<T>(size, memPolicy);

      for (int ii = 0; ii < _capacity; ++ii)
         _data[ii] = value;
   }

   qs_vector(const qs_vector<T> &aa )
   : _data(0), _capacity(aa._capacity), _size(aa._size), _memPolicy(aa._memPolicy), _isOpen(aa._isOpen)
   {
      _data = MemoryControl::allocate<T>(_capacity, _memPolicy);
 
      for (int ii = 0; ii < _size; ++ii)
         _data[ii] = aa._data[ii];
   }

   ~qs_vector()
   {
      MemoryControl::deallocate(_data, _size, _memPolicy);
   }

   /// Needed for copy-swap idiom
   void swap(qs_vector<T> &other)
   {
      std::swap(_data, other._data);
      std::swap(_capacity, other._capacity);
      std::swap(_size, other._size);
      std::swap(_memPolicy, other._memPolicy);
      std::swap(_isOpen, other._isOpen);
   }
   
   /// Implement assignment using copy-swap idiom
   qs_vector<T> &operator=(const qs_vector<T> &aa)
   {
      if (&aa != this)
      {
         qs_vector<T> temp(aa);
         this->swap(temp);
      }
      return *this;
   }
   
   HOST_DEVICE_SYCL
   int get_memPolicy()
   {
	   return _memPolicy;
   }

   void push_back(const T &dataElem)
   {
      qs_assert( _isOpen );
      _data[_size] = dataElem;
      _size++;
   }

   void Open() { _isOpen = true; }
   void Close(){ _isOpen = false; }

   HOST_DEVICE_SYCL
   const T& operator[](int index) const
   {
      return _data[index];
   }

   HOST_DEVICE_SYCL
   T& operator[](int index)
   {
      return _data[index];
   }
   
   HOST_DEVICE_SYCL
   int capacity() const
   {
      return _capacity;
   }

   HOST_DEVICE_SYCL
   int size() const
   {
      return _size;
   }

   void setsize(int size)
   {
      _size = size;
   }
   
   T& back()
   {
      return _data[_size-1];
   }
   
   void reserve(int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM)
   {
      qs_assert( _capacity == 0 );
      _capacity = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }

   void resize(int size, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM)
   {
      qs_assert( _capacity == 0 );
      _capacity = size;
      _size = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);
   }

   void resize(int size, const T &value, MemoryControl::AllocationPolicy memPolicy = MemoryControl::AllocationPolicy::HOST_MEM) 
   {
      qs_assert(_capacity == 0);
      _capacity = size;
      _size = size;
      _memPolicy = memPolicy;
      _data = MemoryControl::allocate<T>(size, memPolicy);

      for (int ii = 0; ii < _capacity; ++ii)
         _data[ii] = value;
   }

   bool empty() const
   {
      return (_size == 0);
   }

   void eraseEnd(int NewEnd)
   {
      _size = NewEnd;
   }

    void pop_back()
   {
      _size--;
   }

   void clear()
   {
      _size = 0;
   }

   void appendList(int listSize, T *list )
   {
      qs_assert(this->_size + listSize < this->_capacity);

      int size = _size;
      this->_size += listSize;

      for( int i = size; i < _size; i++ )
      {
         _data[i] = list[ i-size ];
      }

   }

   const T *const outputPointer()
   {
      return _data;
   }

   //Atomically retrieve an availible index then increment that index some amount
   HOST_DEVICE_SYCL
   int atomic_Index_Inc(int inc)
   {
      int pos;

      ATOMIC_CAPTURE(_size, inc, pos);

      return pos;
   }


private:
   T *_data;
   int _capacity;
   int _size;
   bool _isOpen;
   MemoryControl::AllocationPolicy _memPolicy;
};

#endif
