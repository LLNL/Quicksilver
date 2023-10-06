#include "MC_Base_Particle.hh"

#define MCP_DATA_MEMBER_OLD(member, buffer, index, mode) \
   { if (     mode == MC_Data_Member_Operation::Count )  { (index)++; } \
    else if ( mode == MC_Data_Member_Operation::Pack   ) { buffer[ (index)++ ] = (member); } \
    else if ( mode == MC_Data_Member_Operation::Unpack ) { member = buffer[ (index)++ ]; }   \
    else if ( mode == MC_Data_Member_Operation::Reset )  { (index)++; member = 0; } }

#define MCP_DATA_MEMBER_CAST_OLD(member, buffer, index, mode, someType) \
   { if (     mode == MC_Data_Member_Operation::Count )  { (index)++; } \
    else if ( mode == MC_Data_Member_Operation::Pack   ) { buffer[ (index)++ ] = (member); } \
    else if ( mode == MC_Data_Member_Operation::Unpack ) { member = (someType) buffer[ (index)++ ]; } \
    else if ( mode == MC_Data_Member_Operation::Reset )  { (index)++; member = (someType) 0; } }

#define MCP_DATA_MEMBER_LONG_TO_CHAR8(member, buffer, index, mode) \
  {      if ( mode == MC_Data_Member_Operation::Count  ) { (index) += 8; } \
    else if ( mode == MC_Data_Member_Operation::Pack   ) { MC_Long_To_Char8(&member, &buffer[(index)]); (index) += 8; } \
    else if ( mode == MC_Data_Member_Operation::Unpack ) { MC_Char8_To_Long(&member, &buffer[(index)]); (index) += 8; } \
    else if ( mode == MC_Data_Member_Operation::Reset )  { (index) += 8; member = 0; }}

void MC_Char8_To_Long(uint64_t *long_out, char char_in[8])
{
    *long_out = 0 ;

    for (int char_index = 0; char_index < 8; char_index++)
    {
        *long_out = *long_out | (unsigned char) char_in[char_index]; // OR in next byte
        if (char_index < 7)
        {
            *long_out  = *long_out << 8;              // Shift Left one byte
        }
    }

}

void MC_Long_To_Char8(const uint64_t *long_in,
                      char char_out[8])
{
    uint64_t long_tmp;
    uint64_t mask = 0xffff;

    // Examine long_in from Right > Left, byte by byte.
    long_tmp = *long_in;
    for (int char_index = 7; char_index >= 0; char_index--)
    {
        char_out[char_index] = (char)(long_tmp & mask); // Get right-most byte
        long_tmp             = long_tmp >> 8; // Shift Right one byte
    }

}


//----------------------------------------------------------------------------------------------------------------------
//  Count, pack or unpack a single base particle.  This routine operates in 3
//  different modes.  This is so that the exact same code does the counting, packing and
//  unpacking so they  will always stay synchronized and the communication will happen correctly.
//  Also, when the data structure changes, you only have to change this one place.
//
//----------------------------------------------------------------------------------------------------------------------
void MC_Base_Particle::Serialize(int *int_data, double *float_data, char *char_data, int &int_index, int &float_index,
                                int &char_index, MC_Data_Member_Operation::Enum mode)
{
    MCP_DATA_MEMBER_OLD(coordinate.x, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(coordinate.y, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(coordinate.z, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(velocity.x, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(velocity.y, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(velocity.z, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(kinetic_energy, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(weight, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(time_to_census, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(age, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(num_mean_free_paths, float_data, float_index, mode);
    MCP_DATA_MEMBER_OLD(num_segments, float_data, float_index, mode);

    MCP_DATA_MEMBER_LONG_TO_CHAR8(random_number_seed, char_data, char_index, mode);
    MCP_DATA_MEMBER_LONG_TO_CHAR8(identifier, char_data, char_index, mode);

    MCP_DATA_MEMBER_CAST_OLD(last_event, int_data, int_index, mode, MC_Tally_Event::Enum);
    MCP_DATA_MEMBER_OLD(num_collisions, int_data, int_index, mode);
    MCP_DATA_MEMBER_OLD(breed, int_data, int_index, mode);
    MCP_DATA_MEMBER_OLD(species, int_data, int_index, mode);
    MCP_DATA_MEMBER_OLD(domain, int_data, int_index, mode);
    MCP_DATA_MEMBER_OLD(cell, int_data, int_index, mode);
}






int MC_Base_Particle::num_base_ints = 0;
int MC_Base_Particle::num_base_floats = 0;
int MC_Base_Particle::num_base_chars = 0;


//----------------------------------------------------------------------------------------------------------------------
//  Updates the num base counts by creating an instance and callingthe broadcast routine.
//
//----------------------------------------------------------------------------------------------------------------------
void MC_Base_Particle::Update_Counts()
{
    MC_Base_Particle base_particle;
    num_base_ints = 0;
    num_base_floats = 0;
    num_base_chars = 0;
    base_particle.Serialize(NULL, NULL, NULL, num_base_ints, num_base_floats,
                            num_base_chars, MC_Data_Member_Operation::Count);
}

