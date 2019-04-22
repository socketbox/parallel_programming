#include <stdio.h>
#include <omp.h>

#ifndef _OPENMP
  fprintf(stderr, "OpenMP is not supported\n");
  exit(EXIT_FAILURE);
#endif

int main(int argsc, char** argsv)
{
  double start = omp_get_wtime();
  fprintf(stdout, "This is the start time: %lf\n", start);
  double prec = omp_get_wtick();
  fprintf(stdout, "This machine has a clock precision of %lf\n", prec);
  int numprocs = omp_get_num_procs();
  fprintf(stdout, "This machine has %i cores.\n", numprocs);
  double end = omp_get_wtime();
  fprintf(stdout, "This is the end time: %lf\n", end);
  return 0;
}



