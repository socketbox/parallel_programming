#include <fstream>
#include <stdio.h>
#include <omp.h>
#include "height.cpp"

#ifndef TRIALS
#define TRIALS 5
#endif 

#ifndef NUMT
#define NUMT 1
#endif 

#ifndef NUMNODES
#define NUMNODES 4
#endif 

void write_results_to_csv_file(char *prog_name, int threads, long numnodes, int trials, float vol,\
  double time, double mega_heights);

int main(int argc, char **argv)
{
  if (argc > 1)
    printf("This program does not parse arguments from the command line.\n");
  
  double clock_prec = omp_get_wtick();
  printf("Timer Precision: %lf\n", clock_prec);
  
  omp_set_num_threads(NUMT);

  //by making these constant, we need not declare them as shared in the for loop pragma
  double const full_area = (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
        ( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

  double const half_area = .5 * (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
        ( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

  double const quarter_area = .25 * (  ( ( XMAX - XMIN )/(float)(NUMNODES-1) )  *
        ( ( YMAX - YMIN )/(float)(NUMNODES-1) )  );

  double peak_perf = 0.0;
  double peak_time = 0.0;
  double performance = 0.0;
  double delta_t = 0.0;
  float avg_vol = 0.0;

  for(int j = 0; j < TRIALS; j++)
  {
    double volume = 0.0;
    double tmp_vol = 0.0;
    
    //DEBUG
    /*int corner_count = 0;
    printf("full-ta: %g \n", full_area);
    printf("half-ta: %g \n", half_area);
    printf("quarter-ta: %g \n", quarter_area);*/

    double start = omp_get_wtime();
    #pragma omp parallel for reduction(+:volume),private(tmp_vol)
    for( long i = 0; i < NUMNODES*NUMNODES; i++ )
    {
      //these vars must be kept in each thread's stack	
      long iu = i % NUMNODES;
      long iv = i / NUMNODES;
      
      //DEBUG
      //printf("iu: %d \n", iu);
      //printf("iv: %d \n", iv);

      //start with most likely case first: all interior tiles
      if( ( iu > 0 && iu < NUMNODES-1) && (iv > 0 && iv < NUMNODES-1) ) 
      {
        tmp_vol = height(iu, iv) * full_area;
      }
      //if iu and iv refer to a side
      else if( 
          (iu == 0 && (iv != 0 || iv != NUMNODES-1)) //left side, minus corners
          || (iu == NUMNODES - 1 && (iv != 0 || iv != NUMNODES-1)) //right side, minus corners
          || (iv == NUMNODES - 1 && (iu != 0 || iu != NUMNODES-1)) //top side, minus corners
          || (iv == 0 && (iu != 0 || iu != NUMNODES-1)) ) //bottom side, minus corners
      {
        tmp_vol = height(iu, iv) * half_area;
      }
      //if iu and iv refer to a corner 
      if( (iu == 0 && iv == 0) || (iu == 0 && iv == NUMNODES-1) 
          ||  (iu == NUMNODES-1 && iv == 0) || (iu == NUMNODES-1 && iv == NUMNODES-1) )
      {
        tmp_vol = height(iu, iv) * quarter_area;
        //DEBUG
        //corner_count++;
      }
      volume += tmp_vol; 
    } 
    double finish = omp_get_wtime();
    delta_t = finish - start;
    performance = (NUMNODES*NUMNODES)/delta_t;
    if( peak_perf < performance)
    {
      peak_perf = performance;
      peak_time = delta_t;
    }
    avg_vol += volume; 
  } 
  //DEBUG
  //printf("Corners: %d\n", corner_count);
  avg_vol /= TRIALS;
  printf("NUMNODES: %d, NUMT: %d, Volume: %lf, Time Delta: %lf,\
      MegaHeights/sec: %lf\n", NUMNODES, NUMT, avg_vol, peak_time,\
      peak_perf);

  write_results_to_csv_file(argv[0], NUMT, NUMNODES, TRIALS, avg_vol, peak_time, peak_perf);

}

bool does_file_exist (const std::string &name) 
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
}

void write_results_to_csv_file(char *prog_name, int threads, long numnodes, int trials, float vol, double time, double mega_heights)
{
  std::string filename1(prog_name);
  filename1 = filename1 + ".csv";
	
	//build strings for writing a line at a time
	std::string header("threads,nodes,trials,volume,time,mega_heights_sec");
	std::string results_mflops(std::to_string(threads) + "," + std::to_string(numnodes) + ","\
      + std::to_string(trials) + "," + std::to_string(vol) + "," +\
      std::to_string(time) + "," + std::to_string(mega_heights));

  std::ofstream outFile_mflops;
 
  //if file doesn't exist, open it and write the header line, otherwise open it in append mode
  if (!does_file_exist(filename1))
  {
    outFile_mflops.open(filename1, std::ios::out);
    outFile_mflops << header << std::endl;
  } 
  else
    outFile_mflops.open(filename1, std::ios::app);
  
  outFile_mflops << results_mflops << std::endl;
  //file closed via RAII
}


