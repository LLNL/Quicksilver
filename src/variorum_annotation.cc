#include "variorum_annotation.hh"
extern "C" {
#include <variorum.h>
}
#include <mpi.h>
#include <rankstr_mpi.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#define FILENAME_TEXT "_power_data.txt"

#ifdef HAVE_MPI
MPI_Comm split_comm;
char host[1024];
int processor_rank,comm_size,ret;
double time_delay_due_to_variorum=0;
double time_delay_due_to_file_write=0;
#endif
void variorum_annotate_init(){
  int rank, numprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
  gethostname(host, 1023);

  rankstr_mpi_comm_split(MPI_COMM_WORLD, host, rank, 1, 2, &split_comm);
  MPI_Comm_rank(split_comm, &processor_rank);
  MPI_Comm_size(split_comm, &comm_size);
}
void variorum_annotate_finalize(){
  if(processor_rank==0){
  printf("host:%s \n",host);
  printf("Time delay due to variroum %f \n",time_delay_due_to_variorum);
  printf("Time delay due to write_to_file %f \n",time_delay_due_to_file_write);}
  MPI_Comm_free(&split_comm);
}
void write_to_file(const char *filename, const char *caller_filename, int line,
                   const char *function, const char *data) {
  if (filename == NULL || caller_filename == NULL || function == NULL ||
      data == NULL) {
    fprintf(stderr, "invalid arguments\n");
    return;
  }
  int new_string_length = strlen(filename) + strlen(FILENAME_TEXT) + 1;
  char *new_filename = (char *)malloc(new_string_length * sizeof(char));
  if (new_filename != NULL) {
    sprintf(new_filename, "%s%s", filename, FILENAME_TEXT);
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
  if (new_filename!=NULL)
  free(new_filename);

}

void variorum_annotate_get_node_power_json(const char *file, int line,
                                           const char *function_name) {
  char *s = NULL;
  clock_t start,end,start_1,end_1;
  if (processor_rank == 0) {
    // double start=MPI_Wtime();
    start=clock();
    ret = variorum_get_node_power_json(&s);
    // double end=MPI_Wtime();
    end=clock();
    time_delay_due_to_variorum+=((double)end-start)/CLOCKS_PER_SEC;
    if (ret != 0) {
      printf("variorum:JSON get node power failed.\n");
      free(s);
      return;
    }
    // double start_1=MPI_Wtime();
    start_1=clock();
    write_to_file(host, file, line, function_name, s);
    // double end_1=MPI_Wtime(); 
    end_1=clock();
    time_delay_due_to_file_write+=((double)end_1-start_1)/CLOCKS_PER_SEC; 
    free(s);
  }
}
void variorum_annotate_get_node_power_domain_info_json(
    const char *file, int line, const char *function_name) {

  char *s = NULL;
  if (processor_rank == 0) {
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
