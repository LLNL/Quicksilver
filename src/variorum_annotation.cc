#include "variorum_annotation.hh"
extern "C" {
#include <variorum.h>
#include <variorum_topology.h>
}
#include <mpi.h>
#include <rankstr_mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
void write_to_file(const char *filename, const char *caller_filename, int line,
                   const char *function, const char *data) {
  if (filename == NULL || caller_filename == NULL || function == NULL ||
      data == NULL) {
    fprintf(stderr, "invalid arguments\n");
    return;
  }
  const char *filename_text = "_power_data.txt";
  int new_string_length = strlen(filename) + strlen(filename_text) + 1;
  char *new_filename = (char *)malloc(new_string_length * sizeof(char));
  if (new_filename != NULL) {
    sprintf(new_filename, "%s%s", filename, filename_text);
  }
  FILE *file;
  if (new_filename != NULL)
    file = fopen(new_filename, "a");
  else {

    file = fopen(filename, "a");
  }
  if (file == NULL) {
    fprintf(stderr, "Error opening file %s \n", filename);
    return;
  }
  if (fprintf(file, "{\"file\":\"%s\", ", caller_filename) < 0 ||
      fprintf(file, "\"line\":%d, ", line) < 0 ||
      fprintf(file, "\"caller_func_name\":\"%s\", ", function) < 0 ||
      fprintf(file, "\"power_data\": %s } \n", data) < 0)
    fprintf(stderr, "Error in writing information to file");
  if (fclose(file) != 0) {

    fprintf(stderr, "Error in closing file %s \n", filename);
  }
}
void variorum_annotate_get_node_power_json(const char *file, int line,
                                           const char *function_name) {
  int rank, numprocs;
  char host[1024];
  int num_sockets;
  int ret;
  char *s = NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  gethostname(host, 1023);
  MPI_Comm newcomm;
  int newrank, newsize;

  rankstr_mpi_comm_split(MPI_COMM_WORLD, host, rank, 1, 2, &newcomm);
  MPI_Comm_rank(newcomm, &newrank);
  MPI_Comm_size(newcomm, &newsize);
  if (newrank == 0) {
    num_sockets = variorum_get_num_sockets();
    if (num_sockets <= 0) {

      printf("hwloc returned an invalid number of sockets\n");
      return;
    }
    ret = variorum_get_node_power_json(&s);
    if (ret != 0) {
      printf("variorum:JSON get node power failed.\n");
      free(s);
      return;
    }

    write_to_file(host, file, line, function_name, s);
    free(s);
  }
}
void variorum_annotate_get_node_power_domain_info_json(
    const char *file, int line, const char *function_name) {

  int rank, numprocs;
  char host[1024];
  int num_sockets;
  int ret;
  char *s = NULL;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  gethostname(host, 1023);
  MPI_Comm newcomm;
  int newrank, newsize;

  rankstr_mpi_comm_split(MPI_COMM_WORLD, host, rank, 1, 2, &newcomm);
  MPI_Comm_rank(newcomm, &newrank);
  MPI_Comm_size(newcomm, &newsize);
  if (newrank == 0) {
    num_sockets = variorum_get_num_sockets();
    if (num_sockets <= 0) {

      printf("hwloc returned an invalid number of sockets\n");
      return;
    }
    ret = variorum_get_node_power_domain_info_json(&s);
    if (ret != 0) {
      printf("variorum:JSON get node power failed.\n");
      free(s);
      return;
    }

    write_to_file(host, file, line, function_name, s);
    free(s);
  }
}
