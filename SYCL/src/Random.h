#include <random>

namespace ts
{
class Random
{
public:
  Random() = delete;

  static uint64_t
  random()
  {
    std::uniform_int_distribution<uint64_t> dist{0, UINT64_MAX};
    return dist(_engine);
  }

  static double
  drandom()
  {
    std::uniform_real_distribution<double> dist{0, 1};
    return dist(_engine);
  }

  static void
  seed(uint64_t s)
  {
    _engine.seed(s);
  }

private:
  thread_local static std::mt19937_64 _engine;
};
}; // namespace ts
