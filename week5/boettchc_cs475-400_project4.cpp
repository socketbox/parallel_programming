#include <random>
#include <array>
#include <iostream>
#include <fstream>
#include <cfloat>
#include <limits>

#include <omp.h>

#include "simd.p4.h"

#ifndef ARRSZ
#define ARRSZ 1000
#endif

#ifndef NUMRUNS
#define NUMRUNS 10
#endif

struct Datum
{
  int arrsz;
  double t_vanilla;
  double t_simd;
  double t_vanilla_ms;
  double t_simd_ms;
  double perf_sum;
  double perf_ms;
};

void write_csv(char *prog_name, struct Datum dat);
bool does_file_exist(const std::string &filename);

int main(int argc, char *args[])
{
  if(argc > 1)
    std::cerr << "This program does not accept arguments." << std::endl;
  
  double wtick = omp_get_wtick();
  fprintf(stderr, "OMP clock precision: %g\n", wtick);

  float *rarr_1 = new float[ARRSZ];
  float *rarr_2 = new float[ARRSZ];
  float *rarr_dst_vanilla = new float[ARRSZ];
  float *rarr_dst_simd = new float[ARRSZ];
 
  //Taken from WG21 N3551: Random Number Generation in C++11, by Walter Brown
  
  //use /dev/random so we won't have to derive a seed
  std::random_device rand_dev;

  //use the mersienne twister engine
  std::mt19937 e2(rand_dev());

  //indicate distribution
  float min = 0.;
  float max = 10.;
  std::uniform_real_distribution<>  dis(min, max);

  Datum dat = {ARRSZ, 0.0};
 
  //we'll reuse the time vars
  double start_t = 0.;
  double end_t = 0.;
  double t_delta = 0.;
  //but have a unique peak for each test
  double vanilla_peak = DBL_MAX;
  double simd_peak = DBL_MAX;
  double vanilla_ms_peak = DBL_MAX;
  double simd_ms_peak = DBL_MAX;
  double van_ms_sum = 0.;
  double simd_ms_sum = 0.;

  for(int r=0; r<NUMRUNS; r++)
  {
     
    //generate array 1
    for(int i=0; i<ARRSZ; i++)
    {
      rarr_1[i] = dis(e2);
    }  

    //generate array 2
    for(int i=0; i<ARRSZ; i++)
    {
      rarr_2[i] = dis(e2); 
    }  

    //DEBUG
    std::cerr << "Starting vanilla array multiplication..." << std::endl;
    /**********************************
     start vanilla array mult. code here
     *********************************/
    start_t = omp_get_wtime(); 
    for(int v=0; v<ARRSZ; v++)
    {
      rarr_dst_vanilla[v] = rarr_1[v] * rarr_2[v];
    }
    end_t = omp_get_wtime(); 
    
    t_delta = end_t - start_t;
    //DEBUG
    //fprintf(stderr, "start, end, delta: %lf, %lf, %lf\n", start_t, end_t, t_delta);
    if(t_delta < vanilla_peak)
      vanilla_peak = t_delta;
    /**********************************
     end vanilla array mult. code here
     *********************************/

    //DEBUG
    std::cerr << "Starting simd array multiplication..." << std::endl;
    /**********************************
    start simd array mult. code here
    *********************************/
    start_t = omp_get_wtime(); 
    SimdMul( rarr_1, rarr_2, rarr_dst_simd, ARRSZ);
    end_t = omp_get_wtime(); 
   
    t_delta = end_t - start_t;
    if(t_delta < simd_peak)
      simd_peak = t_delta;
    /**********************************
    end simd array mult. code here
    *********************************/
        
    //DEBUG
    std::cerr << "Starting vanilla multsum test..." << std::endl;
    /**********************************
    start vanilla multsum code here
    *********************************/
    start_t = omp_get_wtime(); 
    for(int v=0; v<ARRSZ; v++)
    {
      van_ms_sum += rarr_1[v] * rarr_2[v];
    }
    end_t = omp_get_wtime(); 
    
    t_delta = end_t - start_t;
    //DEBUG
    //fprintf(stderr, "start, end, delta: %lf, %lf, %lf\n", start_t, end_t, t_delta);
    if(t_delta < vanilla_ms_peak)
      vanilla_ms_peak = t_delta;
    /**********************************
    end vanilla multsum code here
    *********************************/
    
    //DEBUG
    std::cerr << "Starting simd multsum test..." << std::endl;
    /**********************************
    start simd multsum code here
    *********************************/
    start_t = omp_get_wtime(); 
    SimdMulSum( rarr_1, rarr_2, ARRSZ);
    end_t = omp_get_wtime(); 
    
    t_delta = end_t - start_t;
    //DEBUG
    //fprintf(stderr, "start, end, delta: %lf, %lf, %lf\n", vanilla_start_t, vanilla_end_t, t_delta);
    if(t_delta < simd_ms_peak)
      simd_ms_peak = t_delta;
    /**********************************
    end simd multsum code here
    *********************************/
    std::cerr << "All tests finished." << std::endl; 
  
  }
  dat.t_vanilla = vanilla_peak;
  dat.t_simd = simd_peak;
  dat.perf_sum = (vanilla_peak/simd_peak);
  
  dat.t_vanilla_ms = vanilla_ms_peak;
  dat.t_simd_ms = simd_ms_peak;
  dat.perf_ms = (vanilla_ms_peak/simd_ms_peak);
   
  
  write_csv(args[0], dat); 

  delete[] rarr_1;
  rarr_1 = nullptr; 
  delete[] rarr_2;
  rarr_2 = nullptr; 
  delete[] rarr_dst_vanilla;
  rarr_dst_vanilla = nullptr;
  delete[] rarr_dst_simd;
  rarr_dst_simd = nullptr;
}

bool does_file_exist (const std::string& name) 
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
} 

void write_csv(char *prog_name, struct Datum datum)
{
  std::string filename(prog_name);
  filename += ".csv";

  //DEBUG
  std::cout << "In write_csv" << std::endl;
  fprintf(stderr, "datum peaks: %lf, %lf\n", datum.t_vanilla, datum.t_simd);
	
	//build strings for writing a line at a time
	std::string header("Array_Size,Speedup_Mult,Mult_Time_Vanilla_C,Mult_Time_SIMD,\
Speedup_SumMult,SumMult_Time_Vanilla,SumMult_Time_SIMD");
	/*std::string results_mflops(std::to_string(avgs[0].threads)+",");
	std::string results_times(std::to_string(avgs[0].threads)+",");
	for(unsigned long int j=0; j<avgs.size(); j++)
  {
		header += std::to_string(avgs[j].input_size) + std::string(",");
    results_mflops += std::to_string(avgs[j].peak_mflops) + std::string(",");
    results_times += std::to_string(avgs[j].avg_time) + std::string(",");
  }*/
 
  std::ofstream out_file;
  
  if (!does_file_exist(filename))
  {
    out_file.open(filename, std::ios::out);
    out_file << header << std::endl; 
  }
  else
    out_file.open(filename, std::ios::app);
  
  out_file.precision(std::numeric_limits<double>::max_digits10);
  
  out_file << datum.arrsz << "," << datum.perf_sum << "," << datum.t_vanilla << "," << datum.t_simd\
    << "," << datum.perf_ms << "," << datum.t_vanilla_ms << "," << datum.t_simd_ms << std::endl;
  //file closed via RAII
}


