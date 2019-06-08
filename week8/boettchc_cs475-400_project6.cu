#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cstring>
#include <sstream>

// CUDA runtime
#include <cuda_runtime.h>

// Helper functions and utilities to work with CUDA
#include "helper_functions.h"
#include "helper_cuda.h"

#ifndef BLOCKSIZE
#define BLOCKSIZE		32		// number of threads per block
#endif

// setting the number of trials in the monte carlo simulation:
#ifndef NUMTRIALS
#define NUMTRIALS  1000000
#endif

// how many tries to discover the maximum performance:
#ifndef NUMTRIES
#define NUMTRIES  10
#endif

// ranges for the random numbers (changed for proj6):
const float XCMIN =   0.0;
const float XCMAX =   2.0;
const float YCMIN =   0.0;
const float YCMAX =   2.0;
const float RMIN  =   0.5;
const float RMAX  =   2.0;

// function prototypes:
float Ranf( float, float );
int Ranf( int, int );
void TimeOfDaySeed( );
bool does_file_exist(const char *name);
void write_results_to_csv_file(char *prog_name, int threads, int trials, float mega_trials);


// laser tag  (CUDA Kernel) on the device
__global__  void ShootLaser( float *A, float *B, float *C, int *D )
{
	
	unsigned int gid = blockIdx.x*blockDim.x + threadIdx.x;

	// randomize the location and radius of the circle:
	float xc = A[gid];
	float yc = B[gid];
	float  r =  C[gid];

	/*chb: DEBUG:
	printf("This is xc in kernel: %lf\n", xc);
	printf("This is yc in kernel: %lf\n", yc);
	printf("This is r in kernel: %lf\n", r);*/

	// solve for the intersection using the quadratic formula:
	float a = 2.;
	float b = -2.*( xc + yc );
	float c = xc*xc + yc*yc - r*r;
	float d = b*b - 4.*a*c;
	
	//If d is less than 0, then the circle was completely missed. (Case A) 
	//Continue on to the next trial in the for-loop.
	if(d < 0)
		D[gid] = 0;
  else
	{
		// hits the circle:
		// get the first intersection:
		d = sqrt( d );
		float t1 = (-b + d ) / ( 2.*a );  // time to intersect the circle
		float t2 = (-b - d ) / ( 2.*a );  // time to intersect the circle
		float tmin = t1 < t2 ? t1 : t2;    // only care about the first intersection

		//If tmin is less than 0., then the circle completely engulfs the laser pointer. (Case B) Continue on to the next 
		//trial in the for-loop.
		if(tmin < 0)
			D[gid] = 0;
		else
		{
			// where does it intersect the circle?
			float xcir = tmin;
			float ycir = tmin;

			// get the unitized normal vector at the point of intersection:
			float nx = xcir - xc;
			float ny = ycir - yc;
			float x = sqrt( nx*nx + ny*ny );
			nx /= x;  // unit vector
			ny /= x;  // unit vector

			// get the unitized incoming vector:
			float inx = xcir - 0.;
			float iny = ycir - 0.;
			float in = sqrt( inx*inx + iny*iny );
			inx /= in;  // unit vector
			iny /= in;  // unit vector

			// get the outgoing (bounced) vector:
			float dot = inx*nx + iny*ny;
			//float outx = inx - 2.*nx*dot;  // angle of reflection = angle of incidence`
			float outy = iny - 2.*ny*dot;  // angle of reflection = angle of incidence`

			// find out if it hits the infinite plate:
			float t = ( 0. - ycir ) / outy;
			
			//If t is less than 0., then the reflected beam went up instead of down. 
			//Continue on to the next trial in the for-loop.
			if(t < 0)
				D[gid] = 0;
			else
			{
				D[gid] = 1;
			}
		}
	}
}

// main program:
int main( int argc, char* argv[ ] )
{
  TimeOfDaySeed( ); // seed the random number generator
	
	//int dev = findCudaDevice(argc, (const char **)argv);
	
	// allocate host memory for...
  // xc	
	float * hA = new float [ NUMTRIALS ];
	// yc	
	float * hB = new float [ NUMTRIALS ];
	// radius	
	float * hC = new float [ NUMTRIALS ];
	// hit/miss	
	int * hD = new int[ NUMTRIALS ];

	// fill the random-value arrays:
	for( int n = 0; n < NUMTRIALS; n++ )
	{       
		hA[n] = Ranf( XCMIN, XCMAX );
		hB[n] = Ranf( YCMIN, YCMAX );
		hC[n] = Ranf(  RMIN,  RMAX ); 
		hD[n] = 0;
	}      


	// allocate device memory:
	float *dA, *dB, *dC; 
	int	*dD;

	dim3 dimsA( NUMTRIALS, 1, 1 );
	dim3 dimsB( NUMTRIALS, 1, 1 );
	dim3 dimsC( NUMTRIALS, 1, 1 );
	dim3 dimsD( NUMTRIALS, 1, 1 );

	cudaError_t status;
	status = cudaMalloc( reinterpret_cast<void **>(&dA), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dB), NUMTRIALS*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dC), (NUMTRIALS)*sizeof(float) );
		checkCudaErrors( status );
	status = cudaMalloc( reinterpret_cast<void **>(&dD), (NUMTRIALS)*sizeof(int) );
		checkCudaErrors( status );

	// copy host memory to the device:
	status = cudaMemcpy( dA, hA, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dB, hB, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );
	status = cudaMemcpy( dC, hC, NUMTRIALS*sizeof(float), cudaMemcpyHostToDevice );
		checkCudaErrors( status );

	// setup the execution parameters:
	dim3 threads(BLOCKSIZE, 1, 1 );
	dim3 grid( NUMTRIALS / threads.x, 1, 1 );

	// Create and start timer
	cudaDeviceSynchronize( );

	// allocate CUDA events that we'll use for timing:
	cudaEvent_t start, stop;
	status = cudaEventCreate( &start );
		checkCudaErrors( status );
	status = cudaEventCreate( &stop );
		checkCudaErrors( status );

	// get ready to record the maximum performance and the probability:
	float maxPerformance = 0.;      
	float currentProb = 0.;              
	float maxCurrentProb = 0.;           
	double megaTrialsPerSecond = 0.;
	float msecTotal = 0.;
	double secondsTotal = 0.;
	int numHits = 0;
	
		// looking for the maximum performance:
	for( int t = 0; t < NUMTRIES; t++ )
	{
		// record the start event:
		status = cudaEventRecord( start, NULL );
		checkCudaErrors( status );
		
		// execute the kernel:
		ShootLaser<<< grid, threads >>>( dA, dB, dC, dD );
	
		//record the stop event:
		status = cudaEventRecord( stop, NULL );
		checkCudaErrors( status );

		// wait for the stop event to complete:
		status = cudaEventSynchronize( stop );
			checkCudaErrors( status );

		status = cudaEventElapsedTime( &msecTotal, start, stop );
		checkCudaErrors( status );

		// compute and print the performance
		secondsTotal = 0.001 * (double)msecTotal;
		//double multsPerSecond = (float)SIZE * (float)NUMTRIALS / secondsTotal;
		//double megaMultsPerSecond = multsPerSecond / 1000000.;

		//copy result from the device to the host:
		status = cudaMemcpy( hD, dD, (NUMTRIALS)*sizeof(int), cudaMemcpyDeviceToHost );
		checkCudaErrors( status );

		numHits = 0;
		// check the sum :
		for(int i = 0; i < NUMTRIALS; i++ )
		{
			if(hD[i] == 1)
			numHits++;
		}
		megaTrialsPerSecond = (double)NUMTRIALS / secondsTotal / 1000000.;
		if( megaTrialsPerSecond > maxPerformance )
			maxPerformance = megaTrialsPerSecond;
		currentProb = (float)numHits/(float)NUMTRIALS;
		if(currentProb > maxCurrentProb)
			maxCurrentProb = currentProb;
	}
	
	printf("Block Size: %i\tTrials: %i\tHit Probability: %lf\tTime Delta: %lf\tMegaTrials/sec: %lf\n",\
			BLOCKSIZE, NUMTRIALS, maxCurrentProb, secondsTotal, maxPerformance);

	write_results_to_csv_file(argv[0], BLOCKSIZE, NUMTRIALS, maxPerformance); 

	// clean up memory:
	delete [ ] hA;
	delete [ ] hB;
	delete [ ] hC;
	delete [ ] hD;

	status = cudaFree( dA );
		checkCudaErrors( status );
	status = cudaFree( dB );
		checkCudaErrors( status );
	status = cudaFree( dC );
		checkCudaErrors( status );
	status = cudaFree( dD );
		checkCudaErrors( status );

	return 0;
		
}

float Ranf( float low, float high )
{
        float r = (float) rand();               // 0 - RAND_MAX
        float t = r  /  (float) RAND_MAX;       // 0. - 1.

        return   low  +  t * ( high - low );
}

int Ranf( int ilow, int ihigh )
{
        float low = (float)ilow;
        float high = ceil( (float)ihigh );

        return (int) Ranf(low,high);
}

void TimeOfDaySeed( )
{
  struct tm y2k = { 0 };
  y2k.tm_hour = 0;   y2k.tm_min = 0; y2k.tm_sec = 0;
  y2k.tm_year = 100; y2k.tm_mon = 0; y2k.tm_mday = 1;

  time_t  timer;
  time( &timer );
  double seconds = difftime( timer, mktime(&y2k) );
  unsigned int seed = (unsigned int)( 1000.*seconds );    // milliseconds
  srand( seed );
}

bool does_file_exist(const char *name)
{
    if (FILE *file = fopen(name, "r")) 
		{
        fclose(file);
        return true;
    } 
		else 
        return false;
}

void write_results_to_csv_file(char *prog_name, int blocksize, int numtrials, float perf)
{
	//program name + .csv extension + null terminator
	char ext[5] = ".csv";	
	char *filename = std::strcat(prog_name, ext);

	std::ostringstream converter;
	//build strings for writing a line at a time
	std::string results;
	converter << blocksize  << "," << numtrials << "," << perf << std::endl;
	results = converter.str();	
	//std::string results_mflops(std::to_string(threads) + "," + \
      //std::to_string(trials) + "," + std::to_string(hit_prob) + "," + std::to_string(mega_trials));

  std::ofstream outFile_mflops;
  
  if (!does_file_exist(filename))
	{
    outFile_mflops.open(filename, std::ios::out);
		std::string header("blocksize,numtrials,perf");
		outFile_mflops << header << std::endl;
	}
	else
    outFile_mflops.open(filename, std::ios::app);
  
  outFile_mflops << results << std::endl;
  //file closed via RAII
}


