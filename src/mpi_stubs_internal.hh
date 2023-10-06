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

/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/
//
//                                      Copyright (c) 2012
//                           Lawrence Livermore National Security, LLC
//                                      All Rights Reserved
//
/*++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++*/

#ifndef MPI_STUBS_INTERNAL_H
#define MPI_STUBS_INTERNAL_H

#include "mpi_stubs.hh"

//----------------------------------------------------------------------------------------------------------------------
// MPI stubs structures to implement mpi calls
//----------------------------------------------------------------------------------------------------------------------

typedef struct _List *pList; // forward declaration for prototypes.
typedef struct _Listitem *pListitem;

typedef uint64_t MPI_Aint;

typedef struct _List
{
  pListitem head;
  pListitem tail;
  int count;
} List;

typedef struct _Listitem
{
  void *data;
  pListitem prev;
  pListitem next;

#ifdef MPI_STUBS_DEBUG_DATA
  pList list;
#endif

} Listitem;

typedef struct
{
  pList sendlist;
  pList recvlist;

  int num;
  char *name;

} Comm;

typedef struct
{
  pListitem listitem; // to allow Req to be removed from list

  int *buf;
  int tag;
  int complete;

} Req;

typedef struct _Handleitem
{
  int handle = 0;
  struct _Handleitem *next = NULL;

  union
  {
    void *anything; // At least size of void *
    Comm comm;
    Req req;

  } data;

} Handleitem;

typedef struct MPI_Stubs_Data_struct
{

  MPI_Errhandler errhandler;
  int headcount;
  int itemcount;
  int initialized;

  //
  // The first block of handle items will be statically allocated.
  // Subsequent ones will be added if necessary.
  // blocks[0..nblocks-1] are allocated at any given time.
  //
  // Increase MPI_STUBS_MAX_BLOCKS if you *really* need more active request
  // (Although probably something is wrong if you need more than 256k !!!)
  //
  Handleitem block0[MPI_STUBS_BLOCK_ITEMS];
  Handleitem *(blocks[MPI_STUBS_MAX_BLOCKS]);
  int nblocks;

  int need_to_init;
  Handleitem *nextfree;

  MPI_Stubs_Data_struct()
  {
    this->errhandler = MPI_ERRORS_ARE_FATAL;
    this->headcount = 0;
    this->itemcount = 0;
    this->initialized = 0;
    this->nblocks = 0;
    this->need_to_init = 1;
    this->nextfree = NULL;
    for (int index = 0; index < MPI_STUBS_MAX_BLOCKS; index++)
    {
      this->blocks[index] = NULL;
    }
  }

  ~MPI_Stubs_Data_struct(){};

} MPI_Stubs_Data_type;

#endif // ifndef MPI_STUBS_INTERNAL_H
