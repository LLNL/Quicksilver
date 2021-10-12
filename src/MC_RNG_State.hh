/*
Copyright 2019 Advanced Micro Devices

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/


#ifndef MC_RNG_STATE_INCLUDE
#define MC_RNG_STATE_INCLUDE

#include "portability.hh"
#include "DeclareMacro.hh"

//----------------------------------------------------------------------------------------------------------------------
//  A random number generator that implements a 64 bit linear congruential generator (lcg).
//
//  This implementation is based on the rng class from Nick Gentile.
//----------------------------------------------------------------------------------------------------------------------

// Generate a new random number seed
HOST_DEVICE
uint64_t rngSpawn_Random_Number_Seed(uint64_t *parent_seed);
HOST_DEVICE_END

//----------------------------------------------------------------------------------------------------------------------
//  Sample returns the pseudo-random number produced by a call to a random
//  number generator.
//----------------------------------------------------------------------------------------------------------------------
HOST_DEVICE
inline double rngSample(uint64_t *seed)
{
   // Reset the state from the previous value.
   *seed = 2862933555777941757ULL*(*seed) + 3037000493ULL;

   // Map the int state in (0,2**64) to double (0,1)
   // by multiplying by
   // 1/(2**64 - 1) = 1/18446744073709551615.
   return 5.4210108624275222e-20*(*seed);
}
HOST_DEVICE_END



//---------------------------------------------------------------------------//

namespace
{
inline HOST_DEVICE
   // Break a 64 bit state into 2 32 bit ints.
   void breakup_uint64( uint64_t uint64_in,
                        uint32_t& front_bits, uint32_t& back_bits )
   {
      front_bits = static_cast<uint32_t>( uint64_in >> 32 );
      back_bits = static_cast<uint32_t>( uint64_in & 0xffffffff );
   }
HOST_DEVICE_END
}

//---------------------------------------------------------------------------//

namespace
{
   // Function sed to hash a 64 bit int into another, unrelated one.  It
   // does this in two 32 bit chuncks. This function uses the algorithm
   // from Numerical Recipies in C, 2nd edition: psdes, p. 302.  This is
   // used to make 64 bit numbers for use as initial states for the 64
   // bit lcg random number generator.
inline HOST_DEVICE
   void pseudo_des( uint32_t& lword, uint32_t& irword )
   {
      // This random number generator assumes that type uint32_t is a 32 bit int
      // = 1/2 of a 64 bit int. The sizeof operator returns the size in bytes = 8 bits.
      
      const int NITER = 2;
      //const uint32_t c1[] = { 0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L };
      //const uint32_t c2[] = { 0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L};
      uint32_t c1[4];
      c1[0]=0xbaa96887L;
      c1[1]=0x1e17d32cL;
      c1[2]=0x03bcdc3cL;
      c1[3]=0x0f33d1b2L;
      uint32_t c2[4];
      c2[0]=0x4b0f3b58L;
      c2[1]=0xe874f0c3L;
      c2[2]=0x6955c5a6L;
      c2[3]=0x55a7ca46L;

      
      uint32_t ia,ib,iswap,itmph=0,itmpl=0;
      
      for( int i = 0; i < NITER; i++)
      {
         ia = ( iswap = irword ) ^ c1[i];
         itmpl = ia & 0xffff;
         itmph = ia >> 16;
         ib = itmpl*itmpl+ ~(itmph*itmph);
         
         irword = lword ^ (((ia = (ib >> 16) |
                             ((ib & 0xffff) << 16)) ^ c2[i])+itmpl*itmph);
         
         lword=iswap;
      }
   }
HOST_DEVICE_END
}

//---------------------------------------------------------------------------//

namespace
{

   inline HOST_DEVICE
   // Function used to reconstruct  a 64 bit from 2 32 bit ints.
   uint64_t reconstruct_uint64( uint32_t front_bits, uint32_t back_bits )
   {
      uint64_t reconstructed, temp;
      reconstructed = static_cast<uint64_t>( front_bits );
      temp = static_cast<uint64_t>( back_bits );
      
      // shift first bits 32 bits to left
      reconstructed = reconstructed << 32;
      
      // temp must be masked to kill leading 1's.  Then 'or' with reconstructed
      // to get the last bits in
      reconstructed |= (temp & 0x00000000ffffffff);
      
      return reconstructed;
   }
   HOST_DEVICE_END
}

//---------------------------------------------------------------------------//

namespace
{
inline HOST_DEVICE
   // Function used to hash a 64 bit int to get an initial state.
   uint64_t hash_state( uint64_t initial_number )
   {
      // break initial number apart into 2 32 bit ints
      uint32_t front_bits, back_bits;
      breakup_uint64( initial_number, front_bits, back_bits );
      
      // hash the bits
      pseudo_des( front_bits, back_bits );
      
      // put the hashed parts together into 1 64 bit int
      return reconstruct_uint64( front_bits, back_bits );
   }
HOST_DEVICE_END
}

//----------------------------------------------------------------------------------------------------------------------
//  This routine spawns a "child" random number seed from a "parent" random number seed.
//----------------------------------------------------------------------------------------------------------------------

inline HOST_DEVICE
uint64_t rngSpawn_Random_Number_Seed(uint64_t *parent_seed)
{
  uint64_t spawned_seed = hash_state(*parent_seed);
  // Bump the parent seed as that is what is expected from the interface.
  rngSample(parent_seed);
  return spawned_seed;
}

HOST_DEVICE_END

#endif
