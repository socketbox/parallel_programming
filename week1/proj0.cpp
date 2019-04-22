#include <vector>
#include <ctime>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <string.h>

#ifndef _OPENMP
  fprintf(stderr, "OpenMP is not supported\n");
  exit(EXIT_FAILURE);
#endif

struct Datum
{
  int threads;
  int input_size;
  double avg_mflops;
  double avg_time;
  double peak_mflops;
  double shortest_time;
};

bool doesFileExist (const std::string& name) 
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

int* gen_array(int n)
{
  int *randArr = new int[n];
  std::srand(std::time(NULL));
  for(int i = 0; i < n; i++)
  {
    randArr[i] = rand() % 10000;
  }
  return randArr;
}

void print_usage(char *arg0)
{
  fprintf(stdout, "Usage: %s  (passes)  (threads)  (input_size)  (step_size)\n", arg0);
  std::cout << "\t(passes)\t\tthe number of passes for benchmarking" << std::endl;
  std::cout << "\t(threads)\t\tthe maximum number of threads to test" << std::endl;
  std::cout << "\t(input_size)\t\tthe maximum input size to test" << std::endl;
  std::cout << "\t(step_size)\t\tthe size of the step between inputs" << std::endl;
  std::cout << "All arguments required." << std::endl;
  arg0 = nullptr;
}

struct Datum bench_calc_prod(int input_size, int *arr1, int *arr2, int threads, int passes)
{
	struct Datum datum;
	datum.threads = threads;
	datum.input_size = input_size;
	
  double maxMegaMults = 0.0;
	double megaMults = 0.0;	
  double time_delta = 0.0;
  double shortest_time = 42.0;
  double time_delta_sum = 0.0;
  double mflops_sum = 0.0;	
	
  int *devnull = new int[input_size]; 
	
  omp_set_num_threads( threads );
  fprintf(stdout, "####################\n"); 
  fprintf(stderr,"Input Size: %i\n", input_size); 
  fprintf(stdout, "Set No. Threads: %i\n", datum.threads); 
  for(int t=0; t<passes; t++ )
  {
		double time0 = omp_get_wtime( );

		#pragma omp parallel for
		for( int i = 0; i < input_size; i++ )
		{
			devnull[i] = arr1[i] * arr2[i];
			//DEBUG
			//fprintf(stdout, "This is product: %i", throw_away_product);	
		}
		double time1 = omp_get_wtime( );
		time_delta = time1-time0;	
		time_delta_sum += time_delta;
		if( time_delta < shortest_time )
			shortest_time = time_delta;
		megaMults = (double)input_size/time_delta/1000000.;
		mflops_sum += megaMults;	
		if( megaMults > maxMegaMults )
			maxMegaMults = megaMults;
  }
  //DEBUG
  //fprintf(stderr,"Delta Sum: %lf\n", mflops_sum);
  datum.avg_mflops = mflops_sum/passes;
  datum.avg_time = time_delta_sum/passes;
	printf( "Time delta: %lf\n", time_delta);
  datum.peak_mflops = maxMegaMults;
  datum.shortest_time = shortest_time;
  // note: %lf stands for "long float", which is how printf prints a "double"
  //        %d stands for "decimal integer", not "double"
 	 
  delete[] devnull;
  devnull = nullptr;

	return datum;
}

void write_avgs_to_csv_file(char *prog_name, int threads, std::vector<struct Datum> avgs)
{
  std::string filename1(prog_name);
  std::string filename2(prog_name);
  filename1 = filename1 + "_mflops_" + std::to_string(threads) + ".csv";
  filename2 = filename2 + "_times_" + std::to_string(threads) + ".csv";
  //DEBUG
  //fprintf(stderr, "filename: %s\n", filename.c_str());
	
	//build strings for writing a line at a time
	std::string header("threads,");
	std::string results_mflops(std::to_string(avgs[0].threads)+",");
	std::string results_times(std::to_string(avgs[0].threads)+",");
	for(unsigned long int j=0; j<avgs.size(); j++)
  {
		header += std::to_string(avgs[j].input_size) + std::string(",");
    results_mflops += std::to_string(avgs[j].peak_mflops) + std::string(",");
    results_times += std::to_string(avgs[j].avg_time) + std::string(",");
  }
 
  std::ofstream outFile_mflops;
  std::ofstream outFile_times;
  
  if (!doesFileExist(filename1))
    outFile_mflops.open(filename1, std::ios::out);
  else
    outFile_mflops.open(filename1, std::ios::app);
  
  if (!doesFileExist(filename2))
    outFile_times.open(filename2, std::ios::out);
  else
    outFile_times.open(filename2, std::ios::app);
 
  outFile_mflops << header << std::endl;
  outFile_mflops << results_mflops << std::endl;
  outFile_times << header << std::endl;
  outFile_times << results_times << std::endl;
  //file closed via RAII
  prog_name = nullptr;
}

int main(int argc, char *argv[])
{
  std::srand(std::time(NULL));
  
  if(argc < 4)
  {
    print_usage(argv[0]);
    exit(EXIT_FAILURE);
  }
  
  const int passes = atoi(argv[1]);
  const int threads = atoi(argv[2]);
  int n = atoi(argv[3]);
  const int step = atoi(argv[4]);
 
  //input size must be at least 1000, probably more
  if (n < 1000) 
  {
    fprintf(stderr, "Input size too small. Try again.\n");
    exit(EXIT_FAILURE);
  }
  
  std::vector<struct Datum> avgs;
	struct Datum datum;
  for(int s=step; s<=n; s+=step)
  {
      int *arr1 = gen_array(s);
      int *arr2 = gen_array(s);
      datum = bench_calc_prod(s, arr1, arr2, threads, passes); 
      fprintf(stdout, "Work peak: %lf\n", datum.peak_mflops); 
      fprintf(stdout, "Work avg: %lf\n", datum.avg_mflops); 
      fprintf(stdout, "Shortest time: %lf\n", datum.shortest_time); 
      fprintf(stdout, "Avg. time: %lf\n", datum.avg_time); 
      avgs.push_back(datum);
      delete[] arr1;
      delete[] arr2;
      arr1 = nullptr;
      arr2 = nullptr;
  }
  
 
  //write a CSV file whose name includes the program name, and number of threads
  write_avgs_to_csv_file(argv[0], threads, avgs); 

  return 0;
}

