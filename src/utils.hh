#ifndef UTILS_HH
#define UTILS_HH

#include <string>

int mc_get_num_physical_procs(void);

void MC_Verify_Thread_Zero(char const * const file, int line);

void printBanner(const char *git_version, const char *git_hash);

#define MC_Warning printf

void Print0(const char *format, ...);

std::string MC_String(const char fmt[], ...);

#endif
