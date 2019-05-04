#include <stdio.h>
#include <omp.h>
#include <math.h>
#include <ctime>

#ifndef _OPENMP
  fprintf(stderr, "OpenMP is not supported\n");
  exit(EXIT_FAILURE);
#endif 

const int NUMT = 4;

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


/*prototypes
char* OpenAndInitCsv(const char *PROGNAME);
void WriteResultsToCsvFile(const char* FileName, const int NowYear, const int NowMonth, const float\
    Grain, const int GrainDeer, const int Humans, const float Temp, const float Precip); 
*/
int Ranf( unsigned int *seedp, int ilow, int ihigh );
float Ranf( unsigned int *seedp,  float low, float high );
//void Watcher(const char *FileName);
void Watcher();
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

  unsigned int seed = 0;
  NowTemp = GetTemp(NowMonth, seed);
  NowPrecip = GetPrecip(NowMonth, seed);
  /*8float precip = AVG_PRECIP_PER_MONTH + AMP_PRECIP_PER_MONTH * sin( ang );
  NowPrecip = precip + Ranf( &seed,  -RANDOM_PRECIP, RANDOM_PRECIP );
  if( NowPrecip < 0. )
    NowPrecip = 0.;
  */
  //chb
  NowNumHumans = 1;

	//chb: ??
	omp_init_lock( &Lock );
	InitBarrier( NUMT );
  
	omp_set_num_threads( 4 );	// same as # of sections
  #pragma omp parallel sections
  {
		//sets and prints out state
    #pragma omp section
    {
      //Watcher(CsvFileName);
      Watcher();
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

//void Watcher(const char *CsvFileName)
void Watcher()
{
  //unsigned int seed = (unsigned int)std::time(nullptr);  // a thread-private variable

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

    //write out "now " state of data
    /*WriteResultsToCsvFile(CsvFileName, NowYear, NowMonth, NowHeight, NowNumDeer, NowNumHumans,\
        NowTemp, NowPrecip); */
    //advance time
    if(NowMonth < 12)
      NowMonth++;
    else
    {
      NowYear++; 
      NowMonth = 0;
    }
    //TODO: re-compute all environmental variables 
    unsigned int seed = 0;
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
    // compute a temporary next-value for this quantity
		// based on the current state of the simulation:
    //int nextXXX = calcNextFoo();
    fprintf(stderr, "GrainDeer waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "GrainDeer resuming at Barrier #1.\n");
    
    //chb: write next to now
    //NowXXX = nextXXX;
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
	  // compute a temporary next-value for this quantity
		// based on the current state of the simulation:
    //int nextXXX = calcNextFoo();
    fprintf(stderr, "Grain waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "Grain resuming at Barrier #1.\n");
    
    //chb: write next to now
    //NowXXX = nextXXX;
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
	while( NowYear < 2025 )
	{
	  // compute a temporary next-value for this quantity
		// based on the current state of the simulation:
    //int nextXXX = calcNextFoo();
    fprintf(stderr, "Humans waiting at Barrier #1.\n");
    WaitBarrier(); 
    fprintf(stderr, "Humans resuming at Barrier #1.\n");
    
    //chb: write next to now
    //NowXXX = nextXXX;
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
  float temp = AVG_TEMP - AMP_TEMP * cos( ang );
  temp += Ranf( &Seed, -RANDOM_TEMP, RANDOM_TEMP );
  return temp;
}

/*void OpenAndInitCsvFile(const char *prog_name)
{
  std::string filename(prog_name);
  time_t now = time(nullptr);
  filename = filename + "_" + std::to_string(now) + ".csv";
  
  std::ofstream proj3_csv;
  proj3_csv.open(filename, std::ios::out);
  std::string header = "Month/Year,Grain,GrainDeer,Hunters,Temp(F),Rain(in.)";
  proj3_csv << header << std::endl; 
}
  
void WriteResultsToCsvFile(const int NowYear, const int NowMonth, const float Grain, const int GrainDeer, const int Hunters, const float Temp, const float Precip) 
{
  std::string filename1(prog_name);
  filename1 = filename1 + "_mflops_" + std::to_string(threads) + "_" + std::to_string(trials) + ".csv";
  //DEBUG
  //fprintf(stderr, "filename: %s\n", filename.c_str());
	
	//build strings for writing a line at a time
	std::string header("threads,trials,hit_prob,mega_trials");
	std::string results_mflops(std::to_string(threads) + "," + \
      std::to_string(trials) + "," + std::to_string(hit_prob) + "," + std::to_string(mega_trials));

  
  if (!does_file_exist(filename1))
    outFile_mflops.open(filename1, std::ios::out);
  else
    outFile_mflops.open(filename1, std::ios::app);
  
  outFile_mflops << results_mflops << std::endl;
  //file closed via RAII
}

*/
