#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <ctime>
#include <cstdio>
#include <string>
#include <fstream>
#include <iostream>

/*#ifndef _OPENMP
  fprintf(stderr, "OpenMP is not supported\n");
  exit(EXIT_FAILURE);
#endif 
*/

const int NUMT = 4;
const std::string PROGNAME = "project_3";

int	NowYear;		// 2019 - 2024
int	NowMonth;		// 0 - 11

float	NowPrecip;		// inches of rain per month
float	NowTemp;		// temperature this month
float	NowHeight;		// grain height in inches
int	NowNumDeer;		// number of deer in the current population
int NowNumHumans; // number of humans in the vicinity

const float GRAIN_GROWS_PER_MONTH =		8.0;
const float ONE_DEER_EATS_PER_MONTH =		0.5;

const float AVG_PRECIP_PER_MONTH =		6.0;	// average
const float AMP_PRECIP_PER_MONTH =		6.0;	// plus or minus
const float RANDOM_PRECIP =			2.0;	// plus or minus noise

const float AVG_TEMP =				50.0;	// average
const float AMP_TEMP =				20.0;	// plus or minus
const float RANDOM_TEMP =			10.0;	// plus or minus noise

const float MIDTEMP =				40.0;
const float MIDPRECIP =				10.0;

//OpenMP
int NumGone = 0;
int NumAtBarrier = 0;
int NumInThreadTeam;
omp_lock_t Lock;

//rand_r
unsigned int seed = 0;

//prototypes
float sqr(float x);
std::string OpenCsv(const std::string PROGNAME);
void WriteResults(std::string FileName, const int NowYear, const int NowMonth, const float\
    Grain, const int GrainDeer, const int Humans, const float Temp, const float Precip); 
int Ranf( unsigned int *seedp, int ilow, int ihigh );
float Ranf( unsigned int *seedp,  float low, float high );
void Watcher(std::string filename);
void Grain();
void GrainDeer();
void Humans();
void WaitBarrier();
void InitBarrier(int t);
float GetTemp(const int Month, unsigned int seed);
float GetPrecip(const int Month, unsigned int seed);

int main()
{
	// starting date and time:
	NowMonth =    0;
	NowYear  = 2019;

	// starting state (feel free to change this if you want):
	NowNumDeer = 1;
	NowHeight =  1.;

  seed = (unsigned int)std::time(nullptr);  
  NowTemp = GetTemp(NowMonth, seed);
  NowPrecip = GetPrecip(NowMonth, seed);
  //chb
  NowNumHumans = 1;

	omp_init_lock( &Lock );
	InitBarrier( NUMT );

  //DEBUG
  //std::cout << "This is filename outside of pragma: " << CsvFileName << std::endl;

	omp_set_num_threads( 4 );	// same as # of sections
  #pragma omp parallel sections 
  {
		//sets and prints out state
    #pragma omp section
    {
      std::string CsvFileName = OpenCsv(PROGNAME);
      Watcher(CsvFileName);
    }

    #pragma omp section
    {
      GrainDeer( );
    }

    #pragma omp section
    {
      Grain( );
    }

    #pragma omp section
    {
      Humans( );	// your own
    }

  } // implied barrier 
}

void Watcher(std::string filename)
{
  //unsigned int seed = (unsigned int)std::time(nullptr);  // a thread-private variable
  //unsigned int seed = 0;

  while( NowYear < 2025 )
	{
		// compute a temporary next-value for this quantity
		// based on the current state of the simulation:
    //int nextXXX = calcNextFoo();
    fprintf(stderr, "Watcher waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "Watcher resuming at Barrier #1.\n");
    
    //chb: write next to now
    //NowXXX = nextXXX;
    fprintf(stderr, "Watcher waiting at Barrier #2.\n");
    WaitBarrier(); 
    fprintf(stderr, "Watcher resuming at Barrier #2.\n");

    fprintf(stdout, "%s\n", filename.c_str());
    //write out "now " state of data
    WriteResults(filename, NowYear, NowMonth, NowHeight, NowNumDeer, NowNumHumans,\
        NowTemp, NowPrecip); 
    //advance time
    if(NowMonth < 11)
      NowMonth++;
    else
    {
      NowYear++; 
      NowMonth = 0;
    }
    //TODO: re-compute all environmental variables 
    NowTemp = GetTemp(NowMonth, seed);
    NowPrecip = GetPrecip(NowMonth, seed);
    
    fprintf(stderr, "Watcher waiting at Barrier #3.\n");
    WaitBarrier(); 
    fprintf(stderr, "Watcher resuming at Barrier #3.\n");
  }
}

void GrainDeer()
{
	while( NowYear < 2025 )
	{
    /*   
   The Carrying Capacity of the graindeer is the number of inches of height of the grain. If the number of graindeer exceeds this value at the end of a month, decrease the number of graindeer by one. If the number of graindeer is less than this value at the end of a month, increase the number of graindeer by one. */
		int NextNumDeer;	
		if(NowHeight > NowNumDeer)
			NextNumDeer++;
		else
			NextNumDeer--;
		
		if(NextNumDeer < 1)
			NextNumDeer = 1;
	 
    fprintf(stderr, "GrainDeer waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "GrainDeer resuming at Barrier #1.\n");
    
    NowNumDeer = NextNumDeer;
    fprintf(stderr, "GrainDeer waiting at Barrier #2.\n");
    WaitBarrier(); 
    fprintf(stderr, "GrainDeer resuming at Barrier #2.\n");
    
    fprintf(stderr, "GrainDeer waiting at Barrier #3.\n");
    WaitBarrier(); 
    fprintf(stderr, "GrainDeer resuming at Barrier #3.\n");
	}

}

void Grain()
{
	while( NowYear < 2025 )
	{
	  float tempFactor = exp(   -sqr(  ( NowTemp - MIDTEMP ) / 10.  )   );
    float precipFactor = exp(   -sqr(  ( NowPrecip - MIDPRECIP ) / 10.  )   );
    float NextHeight; 
    std::cout << "This is tempFactor: " << std::to_string(tempFactor) << std::endl; 
    std::cout << "This is precipFactor: " << std::to_string(precipFactor) << std::endl; 
    NextHeight += tempFactor * precipFactor * GRAIN_GROWS_PER_MONTH;
    NextHeight -= (float)NowNumDeer * ONE_DEER_EATS_PER_MONTH;
    if(NextHeight < 0.)
      NextHeight = 0.;   

    fprintf(stderr, "Grain waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "Grain resuming at Barrier #1.\n");
    
    NowHeight = NextHeight;
    fprintf(stderr, "Grain waiting at Barrier #2.\n");
    WaitBarrier(); 
    fprintf(stderr, "Grain resuming at Barrier #2.\n");
    
    fprintf(stderr, "Grain waiting at Barrier #3.\n");
    WaitBarrier(); 
    fprintf(stderr, "Grain resuming at Barrier #3.\n");
	}

}

void Humans()
{
	float NextNumHumans;

	while( NowYear < 2025 )
	{
		
		if(NowNumHumans > NowNumDeer)    
		{
			NextNumHumans--; 
		}
		else
		{
			NextNumHumans++;
		}
		if(NextNumHumans < 1)
			NextNumHumans = 1;
	
		fprintf(stderr, "Humans waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "Humans resuming at Barrier #1.\n");
    
   	NowNumHumans = NextNumHumans; 

		fprintf(stderr, "Humans waiting at Barrier #2.\n");
    WaitBarrier(); 
    fprintf(stderr, "Humans resuming at Barrier #2.\n");
    
    fprintf(stderr, "Humans waiting at Barrier #3.\n");
    WaitBarrier(); 
    fprintf(stderr, "Humans resuming at Barrier #3.\n");

	}

}

// specify how many threads will be in the barrier:
//	(also init's the Lock)

void InitBarrier( int n )
{
	NumInThreadTeam = n;
  NumAtBarrier = 0;
	omp_init_lock( &Lock );
}

// have the calling thread wait here until all the other threads catch up:
void WaitBarrier( )
{
  omp_set_lock( &Lock );
	{
    NumAtBarrier++;
		if( NumAtBarrier == NumInThreadTeam )
		{
      NumGone = 0;
      NumAtBarrier = 0;
      // let all other threads get back to what they were doing
      // before this one unlocks, knowing that they might immediately
      // call WaitBarrier( ) again:
      while( NumGone != NumInThreadTeam-1 );
      omp_unset_lock( &Lock );
							return;
    }
  }
	omp_unset_lock( &Lock );

	while( NumAtBarrier != 0 )
  ;	// this waits for the nth thread to arrive

	#pragma omp atomic
    NumGone++;			// this flags how many threads have returned
}

float Ranf( unsigned int *seedp,  float low, float high )
{
  // 0 - RAND_MAX
  float r = (float) rand_r( seedp ); 
  return(   low  +  r * ( high - low ) / (float)RAND_MAX   );
}


int Ranf( unsigned int *seedp, int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = (float)ihigh + 0.9999f;

        return (int)(  Ranf(seedp, low,high) );
}

float sqr(float x)
{
  return x*x;
}

float GetTemp(const int Month, unsigned int Seed)
{
  float ang = (  30.*(float)Month + 15.  ) * ( M_PI / 180. );
  float temp = AVG_TEMP - AMP_TEMP * cos( ang );
  temp += Ranf( &Seed, -RANDOM_TEMP, RANDOM_TEMP );
  return temp;
}

float GetPrecip(const int Month, unsigned int Seed)
{
  float ang = (  30.*(float)Month + 15.  ) * ( M_PI / 180. );
  float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
  precip += Ranf( &Seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
  if( precip < 0. )
    precip = 0.;
  return precip; 
}

std::string OpenCsv(const std::string prog_name)
{
  std::string filename(prog_name);
  time_t now = time(nullptr);
  filename = filename + "_" + std::to_string(now) + ".csv";
  
  std::ofstream proj3_csv;
  proj3_csv.open(filename, std::ios::out);
  std::string header = "Month/Year,Grain,GrainDeer,Humans,Temp(F),Rain(in.)";
  proj3_csv << header << std::endl; 
  proj3_csv.close(); 
  return filename;
}

void WriteResults(std::string File, const int NowYear, const int NowMonth, const float Grain, const int GrainDeer, const int Humans, const float Temp, const float Precip) 
{
  std::cout << "This is filename: " << File << std::endl;
	const char DELIM = ',';	
	//build strings for writing a line at a time
	std::string state = std::to_string(NowMonth) + "/" + std::to_string(NowYear) + DELIM +\
		std::to_string(Grain) + DELIM + std::to_string(GrainDeer) + DELIM + std::to_string(Humans) + DELIM\
		+ std::to_string(Temp) + DELIM + std::to_string(Precip);
 	
	std::ofstream csvout; 
 
  if (!fopen(File.c_str(), "a"))
  	fprintf(stderr, "Cannot write to nonexistant file. Exiting.\n"); 
  else
    csvout.open(File, std::ios::app);
  
	csvout << state << std::endl;
  //file closed via RAII
}

