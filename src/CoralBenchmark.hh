#ifndef CORALBENCHMARK_HH
#define CORALBENCHMARK_HH

#ifdef CORAL_2_BENCHMARK
class MonteCarlo;
class Parameters;

void coralBenchmarkCorrectness( MonteCarlo* monteCarlo, Parameters &params );
#endif

#endif
