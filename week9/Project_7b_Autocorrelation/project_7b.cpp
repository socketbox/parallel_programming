#include <cstdlib>
#include <iostream>
#include <cstdio>

#include <omp.h>
#include <cl.h>
#include <cl_platform.h>

#include "simd.p7b.h"

struct Timing
{
  char *name; 
  double time;
  int threads;
  int dataSize;
};

void getDataSize(FILE *fp, int *sz);
struct Timing timeOmp(int threads, int size, float *sigArr, float *sums); 
void printTimingStruct(struct Timing);
//void printArray(int sz, float *array);
template <typename numeric> void printArray(int sz, numeric *array);

int main(int argc, char *argv[])
{
	if(argc != 3)
	{
		fprintf(stderr, "This program takes two file arguments: the CL kernel file\
				and the signal file, in that order. Exiting.\n.");
		std::exit(EXIT_FAILURE);
	}

  #ifndef _OPENMP
    fprintf(stderr, "No OpenMP support.\n");
		std::exit(EXIT_FAILURE);
  #endif

	#ifndef CL_FILE_NAME
	#define CL_FILE_NAME	"project7b.cl"
	#endif 
  
	std::FILE *fp = std::fopen( argv[2], "r");
  
  int sigSize;
  getDataSize(fp, &sigSize);
  
  float *sumsNp  = new float[ sigSize ];
  float *sigArray = new float[ 2*sigSize ];

  for( int i = 0; i < sigSize; i++ )
  {
    fscanf( fp, "%f", &sigArray[i] );
    sigArray[i+sigSize] = sigArray[i];		// duplicate the array
  }

  //done with the file
  std::fclose(fp);

  /****************************************************************************
   * No Parallelism
  ****************************************************************************/
  double nonp0 = omp_get_wtime();
  //not paralleled
  for( int shift = 0; shift < sigSize; shift++ )
  {
    float sum = 0.;
    for( int i = 0; i < sigSize; i++ )
    {
      sum += sigArray[i] * sigArray[i + shift];
    }
    sumsNp[shift] = sum;
  }
  double nonp1 = omp_get_wtime();
  
  printf("Non-parallel time: %fs\n", nonp1-nonp0);

  /****************************************************************************
  * OMP One Thread
  ****************************************************************************/
  int threads = 1; 
  
  Timing ots1;
  ots1.threads = threads;
  ots1.dataSize = sigSize;

  float *sumsOmp1  = new float[ sigSize ];
  
  omp_set_num_threads(threads);

  double omp1_s = omp_get_wtime();
 
  #pragma omp parallel for shared(sigArray, sumsOmp1) //reduction(+:sum) 
  for( int shift = 0; shift < sigSize; shift++ )
  {
    float sum = 0.;
    for( int i = 0; i < sigSize; i++ )
    {
      sum += sigArray[i] * sigArray[i + shift];
    }
    sumsOmp1[shift] = sum;	
  }
  double omp1_f = omp_get_wtime();
  
  ots1.time = omp1_f - omp1_s;
  
  printf("OMP 1 Thread: %f\n", ots1.time);


  /****************************************************************************
   * OMP Sixteen Threads
  ****************************************************************************/
  float *sumsOmp16  = new float[ sigSize ];
  
  threads = 16; 
  
  Timing ots16;
  ots16.threads = threads;
  ots16.dataSize = sigSize;
  
  omp_set_num_threads(threads);

  double omp16_s = omp_get_wtime();
 
  #pragma omp parallel for shared(sigArray, sumsOmp16) //reduction(+:sum) 
  for( int shift = 0; shift < sigSize; shift++ )
  {
    float sum = 0.;
    for( int i = 0; i < sigSize; i++ )
    {
      sum += sigArray[i] * sigArray[i + shift];
    }
    sumsOmp16[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
  }
  double omp16_f = omp_get_wtime();
  
  ots16.time = omp16_f - omp16_s;
  
  printf("OMP 16 Thread: %f\n", ots16.time);

  /****************************************************************************
   * SIMD
  ****************************************************************************/
  float *sumsSimd  = new float[ sigSize ];
  Timing otsSimd;
  otsSimd.threads = threads;
  otsSimd.dataSize = sigSize;

  double ompSimd_s = omp_get_wtime();
  for( int shift = 0; shift < sigSize; shift++ )
  {
     sumsSimd[shift] = SimdMulSum(&sigArray[0], &sigArray[0 + shift], sigSize);
  }
  double ompSimd_f = omp_get_wtime();
  
  otsSimd.time = ompSimd_f - ompSimd_s;
  
  printf("SIMD: %f\n", otsSimd.time);

  /****************************************************************************
   * OpenCL
  ****************************************************************************/
  /*
  Timing otsOpenCl;
  otsOpenCl.threads = threads;
  otsOpenCl.dataSize = sigSize;

  double ompSimd_s = omp_get_wtime();
  for( int shift = 0; shift < sigSize; shift++ )
  {
     sumsSimd[shift] = SimdMulSum(&sigArray[0], &sigArray[0 + shift], sigSize);
  }
	double ompSimd_f = omp_get_wtime();
	*/
	FILE *clfp;
	clfp = fopen( argv[1], "r" );
	if( fp == NULL )
	{
		fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
		return 1;
	}

	cl_int status;		// returned status from opencl calls
				// test against CL_SUCCESS

	// get the platform id:

	cl_platform_id platform;
	status = clGetPlatformIDs( 1, &platform, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetPlatformIDs failed (2)\n" );
	
	// get the device id:

	cl_device_id device;
	status = clGetDeviceIDs( platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clGetDeviceIDs failed (2)\n" );


 	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	//chb: already done, no?
	//float *hArray = new float[ 2*Size ];
	//float *hSums  = new float[ 1*sigSize ];
	float *sumsOpenCl  = new float[ sigSize ];

	cl_mem dArray = clCreateBuffer( context, CL_MEM_READ_ONLY,  2*sigSize*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );
	cl_mem dSums  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, 1*sigSize*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

  status = clEnqueueWriteBuffer( cmdQueue, dArray, CL_FALSE, 0, sigSize, sigArray, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );


	//////////// Taken stragith from proj5
	
	fseek( fp, 0, SEEK_END );
	size_t fileSize = ftell( fp );
	fseek( fp, 0, SEEK_SET );
	char *clProgramText = new char[ fileSize+1 ];		// leave room for '\0'
	size_t n = fread( clProgramText, 1, fileSize, fp );
	clProgramText[fileSize] = '\0';
	fclose( fp );
	if( n != fileSize )
		fprintf( stderr, "Expected to read %d bytes read from '%s' -- actually read %d.\n", fileSize, CL_FILE_NAME, n );

	// create the text for the kernel program:

	char *strings[1];
	strings[0] = clProgramText;
	cl_program program = clCreateProgramWithSource( context, 1, (const char **)strings, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateProgramWithSource failed\n" );
	delete [ ] clProgramText;

	// 8. compile and link the kernel code:

	char *options = { "" };
	status = clBuildProgram( program, 1, &device, options, NULL, NULL );
	if( status != CL_SUCCESS )
	{
		size_t size;
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &size );
		cl_char *log = new cl_char[ size ];
		clGetProgramBuildInfo( program, device, CL_PROGRAM_BUILD_LOG, size, log, NULL );
		fprintf( stderr, "clBuildProgram failed:\n%s\n", log );
		delete [ ] log;
	}

	// 9. create the kernel object:

	cl_kernel kernel = clCreateKernel( program, "ArrayMult", &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateKernel failed\n" );


	///////////////// end proj5 cp
	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dArray );
	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dSums  );
	





	
  
///////////////////////////////////////////////////////////////////////////////////////////////////////
//and make the global and local work sizes:

	size_t globalWorkSize[3] = { Size,         1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

///////////////////////////////////////////////////////////////////////////////////////////////////////
otsOpenCl.time = ompSimd_f - ompSimd_s;
  
  printf("OpenCL: %f\n", otsOpenCl.time);

  delete[] sumsNp; 
  delete[] sumsOmp1; 
  delete[] sumsOmp16; 
  delete[] sumsSimd; 
  delete[] sumsOpenCl;
  delete[] sigArray; 

  return 0;
}

void getDataSize(FILE *fp, int *size)
{
  if( fp == NULL )
  {
    fprintf( stderr, "Cannot open file 'signal.txt'\n" );
    exit( 1 );
  }
  
  //first line indicates number of values in file
  fscanf( fp, "%d", size);
}

struct Timing timeOmp(int threads, int size, float *sigArray, float *sums)
{
  printf("before decl. of Timing struct");
  Timing ots;
  ots.threads = threads;
  ots.dataSize = size;

  omp_set_num_threads(threads);

  double omp0 = omp_get_wtime();
 
  //chb: debug
  printf("before omp for");
  #pragma omp parallel for shared(sigArray, sums) //reduction(+:sum) 
  for( int shift = 0; shift < size; shift++ )
  {
    float sum = 0.;
    for( int i = 0; i < size; i++ )
    {
      sum += sigArray[i] * sigArray[i + shift];
    }
    sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
  }
  printf("after omp for");
  double omp1 = omp_get_wtime();
  
  ots.time = omp1-omp0;

  return ots;
}

//void printFloatArray(int size, float *array)
template <typename numeric> void printArray(int size, numeric *array)
{
  for(int i = 0; i < size; i++)
  {
    //printf("%d: %f\n", i, array[i]);
    std::cout << i << ": " << array[i] << std::endl; 
  }
}

void printTimingStruct(struct Timing ts)
{
  printf("Results for: %s\n", ts.name); 
}
/*
//CPU SIMD
for( int shift = 0; shift < Size; shift++ )
{
	float sum = 0.;
	for( int i = 0; i < Size; i++ )
	{
		sum += Array[i] * Array[i + shift];
	}
	Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
}

//OpenCL
for( int shift = 0; shift < Size; shift++ )
{
	float sum = 0.;
	for( int i = 0; i < Size; i++ )
	{
		sum += Array[i] * Array[i + shift];
	}
	Sums[shift] = sum;	// note the "fix #2" from false sharing if you are using OpenMP
}

//The outer for-loop is the collection of gids that OpenCL/CUDA is creating for us, so we don't need to implement that outer for-loop ourselves. Do the inner for-loop for each gid. The value of shift is just the value of gid.

kernel void AutoCorrelate( global const float *dArray, global float *dSums )
{
	int Size = get_global_size( 0 );	// the dArray size is actually twice this big
        int gid  = get_global_id( 0 );
	int shift = gid;

	float sum = 0.;
	for( int i = 0; i < Size; i++ )
	{
		sum += dArray[i] * dArray[i + shift];
	}
	dSums[shift] = sum;

which would make the creation of the host data structures:

float *hArray = new float[ 2*Size ];
float *hSums  = new float[ 1*Size ];

which would make the device data structures:

cl_mem dArray = clCreateBuffer( context, CL_MEM_READ_ONLY,  2*Size*sizeof(cl_float), NULL, &status );
cl_mem dSums  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, 1*Size*sizeof(cl_float), NULL, &status );

and make the kernel arguments:

status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dArray );
status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dSums  );

and make the global and local work sizes:

size_t globalWorkSize[3] = { Size,         1, 1 };
size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

Note that it is really important that the global data set size be set to Size, not 2*Size. The amount of data we are producing is Size. The fact that we are computing with an Array that is 2*Size long is just a computational convenience. 
*/





