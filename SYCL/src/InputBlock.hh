#ifndef INPUT_BLOCK_HH
#define INPUT_BLOCK_HH

#include <vector>
#include <string>
#include <map>
#include <sstream>
#include "qs_assert.hh"


class InputBlock
{
 public:

   InputBlock(const std::string& blockName);
   void addPair(const std::string& keyword, const std::string& value);

   template<typename T>
   void getValue(const std::string& keyword, T& value) const;

   const std::string& name() const {return _blockName;}
   unsigned nPairs() const {return _kvPair.size();}

   void serialize(std::vector<char>& buf) const;
   void deserialize(const std::vector<char>& buf);

 private:
   void parseError(const std::string& keyword) const;

   std::string                        _blockName;
   std::map<std::string, std::string> _kvPair;
};

// If the keyword isn't found, value is unchanged.
template<typename T>
void InputBlock::getValue(const std::string& keyword, T& value) const
{
   auto here = _kvPair.find(keyword);
   if (here == _kvPair.end())
      return;

   std::istringstream tmp(here->second);
   tmp >> value;

   if (!tmp)
      parseError(keyword);
}

inline void InputBlock::parseError(const std::string& keyword) const
{
   qs_assert(false);
}

#endif
