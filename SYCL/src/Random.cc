
#include "Random.h"

namespace ts
{
thread_local std::mt19937_64 Random::_engine{std::random_device{}()};
}; // namespace ts
