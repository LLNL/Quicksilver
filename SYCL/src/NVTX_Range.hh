/// \file
/// Manage NVTX ranges.  These are used to provide extra information
/// to NVProf.  They also create regions that can be visualized in
/// NVVP.

/// The easiest way to use a range is to create a NVTX_Range instance
/// at the start of a scope (such as a function).  The range will be
/// automatically ended by the destructor when the instance goes out
/// of scope.  The endRange() method exists for situations where it
/// would be awkward or impossible to take advantage of scope to end
/// the range.

#ifndef NVTX_RANGE_HH
#define NVTX_RANGE_HH

#include <string>

#ifdef USE_NVTX
#include "nvToolsExt.h"
#endif


class NVTX_Range
{
 public:
   
   NVTX_Range(const std::string& rangeName)
   {
      #ifdef USE_NVTX
      char *result = strdup(rangeName.c_str());
      _rangeId = nvtxRangeStartA(result);
      _isOpen = true;
      #endif
   }
   
   ~NVTX_Range()
   {
      #ifdef USE_NVTX
      if (_isOpen)
	 nvtxRangeEnd(_rangeId);
      #endif
   }

  void endRange()
  {
      #ifdef USE_NVTX
      nvtxRangeEnd(_rangeId);
      _isOpen = false;
      #endif
  }
  
 private:
   #ifdef USE_NVTX
   nvtxRangeId_t _rangeId;
   bool _isOpen;
   #endif
};

#endif
