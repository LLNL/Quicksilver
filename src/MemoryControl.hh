#ifndef MEMORY_CONTROL_HH
#define MEMORY_CONTROL_HH

#include "cudaUtils.hh"

#include "qs_assert.hh"

namespace MemoryControl
{
   enum AllocationPolicy {HOST_MEM, UVM_MEM, UNDEFINED_POLICY};

   template <typename T>
   T* allocate(const int size, const AllocationPolicy policy)
   {
      if (size == 0) { return NULL;}
      T* tmp = NULL;
      
      switch (policy)
      {
        case AllocationPolicy::HOST_MEM:
         tmp = new T [size];
         break;
#ifdef HAVE_UVM
        case AllocationPolicy::UVM_MEM:
         void *ptr;
         cudaMallocManaged(&ptr, size*sizeof(T), cudaMemAttachGlobal);
         tmp = new(ptr) T[size]; 
         break;
#endif
        default:
         qs_assert(false);
         break;
      }
      return tmp;
   }

   template <typename T>
   void deallocate(T* data, const int size, const AllocationPolicy policy)
   {
      switch (policy)
      {
        case MemoryControl::AllocationPolicy::HOST_MEM:
         delete[] data; 
         break;
#ifdef HAVE_UVM
        case UVM_MEM:
         for (int i=0; i < size; ++i)
            data[i].~T();
         cudaFree(data);
         break;
#endif         
        default:
         qs_assert(false);
         break;
      }
   }
}


#endif
