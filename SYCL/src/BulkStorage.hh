#ifndef BULK_STORAGE_HH
#define BULK_STORAGE_HH

#include "MemoryControl.hh"

template <typename T>
class BulkStorage
{
 public:
   BulkStorage()
   : _bulkStorage(0),
     _refCount(0),
     _size(0),
     _capacity(0),
     _memPolicy(MemoryControl::AllocationPolicy::UNDEFINED_POLICY)
   {
      _refCount = new int;
      *_refCount = 1;
   }

   BulkStorage(const BulkStorage& aa)
   : _bulkStorage(aa._bulkStorage), _refCount(aa._refCount), _size(aa._size), _capacity(aa._capacity), _memPolicy(aa._memPolicy)
   {
      ++(*_refCount);
   }
   
   ~BulkStorage()
   {
      // Check for instances that never allocated memory.
      // I'm not exactly sure how this can happen, but it does.
      if (_bulkStorage == 0) 
         return;
      
      --(*_refCount);
      if (*_refCount > 0)
         return;
            
      MemoryControl::deallocate(_bulkStorage, _capacity, _memPolicy);
      delete _refCount;
   }
   
   /// Needed for copy-swap idiom
   void swap(BulkStorage<T>& other)
   {
      std::swap(_bulkStorage, other._bulkStorage);
      std::swap(_refCount,    other._refCount);
      std::swap(_size,        other._size);
      std::swap(_capacity,    other._capacity);
      std::swap(_memPolicy,   other._memPolicy);
   }

   /// Implement assignment using copy-swap idiom
   BulkStorage& operator=(const BulkStorage& aa)
   {
      if (&aa != this)
      {
         BulkStorage<T> temp(aa);
         this->swap(temp);
      }
      return *this;
   }
   
   void setCapacity(int capacity, MemoryControl::AllocationPolicy policy)
   {
      qs_assert(_bulkStorage == 0);
      _bulkStorage = MemoryControl::allocate<T>(capacity, policy);
      _capacity = capacity;
      _memPolicy = policy;
   }
   
   T* getBlock(int nItems)
   {
      T* blockStart = _bulkStorage + _size;
      _size += nItems;
      qs_assert(_size <= _capacity);
      return blockStart;
   }
   

 private:

   // This class doesn't have well defined copy semantics.  However,
   // just disabling copy operations breaks the build since we haven't
   // been consistent about dealing with copy semantics in classes like
   // MC_Mesh_Domain.
   


   T* _bulkStorage;
   int* _refCount;
   int _size;
   int _capacity;
   MemoryControl::AllocationPolicy _memPolicy;
   
};


#endif
