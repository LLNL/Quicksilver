#include "parseUtils.hh"
#include <utility>
#include "InputBlock.hh"

using std::string;
using std::istream;

namespace
{
   string whitespace(" \t\f\v\n\r");
   bool isComment(std::string line);
   bool split(string line, string& keyword, string& value, int& indent);
   bool validKeyword(const string& word);
   void chop(string& line);
}

bool blockStart(const string& line, string& blockName)
{
   string keyword;
   string value;
   int indent;
   bool valid = split(line, keyword, value, indent);
   if (valid && indent == 0 && value.size() == 0)
   {
      blockName = keyword;
      return true;
   }
   return false;
}

string readBlock(InputBlock& block, istream& in)
{
   string line;
   while (!in.eof())
   {
      getline(in, line);
      if (isComment(line))
         continue;
      string keyword;
      string value;
      int indent;
      bool valid = split(line, keyword, value, indent);
      if (!valid || indent == 0)
         break;
      block.addPair(keyword, value);
   }

   return line;
}

namespace
{
   /// Returns true if line contains nothing but whitespace and
   /// comments. False otherwise.
   bool isComment(string line)
   {
      size_t here = line.find("//");
      if (here != string::npos)
         line.erase(here, string::npos);
      return (line.find_last_not_of(whitespace) == string::npos);
   }
}

namespace
{
   bool split(string line, string& keyword, string& value, int& indent)
   {
      indent = 0;
      while (indent < line.size() && isspace(line[indent]))
         ++indent;

      size_t delimPos = line.find_first_of(":=", indent);
      if (delimPos == string::npos)
         return false;
      keyword = line.substr(indent, delimPos-indent);
      chop(keyword);
      if (! validKeyword(keyword))
         return false;
      value.clear();
      if (delimPos + 1 < line.size())
      {
         value = line.substr(delimPos+1, string::npos);
         chop(value);
      }
      return true;
   }
}

namespace
{
   bool validKeyword(const string& word)
   {
      return true;
   }
}

namespace
{
   void chop(string& line)
   {
      size_t here = line.size();
      while (here > 0 && isspace(line[here-1]))
         --here;
      if (here < line.size())
          line.erase(here, string::npos);
      size_t nSpace = 0;
      while (nSpace < line.size() && isspace(line[nSpace]))
         ++nSpace;
      line.erase(0, nSpace);
   }
}
