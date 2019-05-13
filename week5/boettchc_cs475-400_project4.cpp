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
  double perf;
};

void write_csv(char *prog_name, struct Datum dat);
bool does_file_exist(const std::string &filename);

int main(int argc, char *args[])
{
  if(argc > 1)
    std::cerr << "This program does not accept rguments." << std::endl;
  
  double wtick = omp_get_wtick();
  fprintf(stderr, "OMP clock precision: %lf\n", wtick);


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
  double vanilla_start_t = 0.;
  double vanilla_end_t = 0.;
  double simd_start_t = 0.;
  double simd_end_t = 0.;
  double vanilla_peak = DBL_MAX;
  double simd_peak = DBL_MAX;
  double vanilla_msum_peak = DBL_MAX;
  double simd_msum_peak = DBL_MAX;
  double t_delta = 0.;

  for(int r=0; r<NUMRUNS; r++)
  {
     
    float rd = 0;
    
    //generate array 1
    for(int i=0; i<ARRSZ; i++)
    {
      rd = dis(e2); 
      rarr_1[i] = rd;
    }  

    //generate array 2
    for(int i=0; i<ARRSZ; i++)
    {
      rd = dis(e2); 
      rarr_2[i] = rd;
    }  
    
    //start vanilla array mult. code here
    vanilla_start_t = omp_get_wtime(); 
    for(int v=0; v<ARRSZ; v++)
    {
      rarr_dst_vanilla[v] = rarr_1[v] * rarr_2[v];
    }
    vanilla_end_t = omp_get_wtime(); 
    //end vanilla array mult. code here
    
    t_delta = vanilla_end_t - vanilla_start_t;
    //DEBUG
    //fprintf(stderr, "start, end, delta: %lf, %lf, %lf\n", vanilla_start_t, vanilla_end_t, t_delta);
    if(t_delta < vanilla_peak)
      vanilla_peak = t_delta;

    //start simd array mult. code here
    simd_start_t = omp_get_wtime(); 
    SimdMul( rarr_1, rarr_2, rarr_dst_simd, ARRSZ);
    simd_end_t = omp_get_wtime(); 
    //end vanilla array mult. code here
    t_delta = simd_end_t - simd_start_t;
    //DEBUG
    //fprintf(stderr, "start, end, delta: %lf, %lf, %lf\n", vanilla_start_t, vanilla_end_t, t_delta);
    if(t_delta < simd_peak)
      simd_peak = t_delta;


  }
  dat.t_vanilla = vanilla_peak;
  dat.t_simd = simd_peak;
  dat.perf = (vanilla_peak/simd_peak);
  
  dat.t_vanilla_msum = vanilla_msum_peak;
  dat.t_simd_msum = simd_msum_peak;
  dat.perf_msum = (vanilla_msum_peak/simd_msum_peak);
   
  
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
	std::string header("Array_Size,Speedup,Time_Vanilla_C,Time_SIMD");
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
  
  out_file << datum.arrsz << "," << datum.perf << "," << \
    datum.t_vanilla << "," << datum.t_simd << std::endl;
  //file closed via RAII
}


