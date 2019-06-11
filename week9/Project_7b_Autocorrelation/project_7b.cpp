#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cmath>

#include <omp.h>
#include <cl.h>
#include <cl_platform.h>

#include "simd.p7b.h"

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 32
#endif

#ifndef X_AXIS_SZ
#define X_AXIS_SZ	512 
#endif

#ifndef PERF_FILE
#define PERF_FILE "perfs.csv"
#endif

#ifndef NUM_TESTS
#define NUM_TESTS 	5 
#endif

struct Timing
{
  const char *name; 
  double time;
  int threads;
  int dataSize;
	double perf;
};

void getDataSize(FILE *fp, int *sz);
void Wait( cl_command_queue );
bool does_file_exist(const char *name);
void write_sums_to_csv_file(const char *prog_name, int x_size, float *sums);
void write_timings_to_csv_file(const char *file_name, int arr_sz, struct Timing *times);
template <typename numeric> void printArray(int sz, numeric *array);

int main(int argc, char *argv[])
{
	struct Timing times[5];

	if(argc != 3)
	{
		fprintf(stderr, "This program takes two file arguments: the CL kernel file "
				"and the signal file, in that order. Exiting.\n.");
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
	struct Timing npts;
	npts.name = "Non-parallel";	
	npts.threads = 0;
	npts.dataSize = sigSize;
	
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
 
	npts.time = nonp1 - nonp0;
	npts.perf = pow(npts.dataSize, 2)/npts.time/1000000000;
	times[0] = npts;
  printf("Non-parallel time: %fs\n", nonp1-nonp0);

  /****************************************************************************
  * OMP One Thread
  ****************************************************************************/
  int threads = 1; 
  
  Timing ots1;
	ots1.name = "OMP 1 Thread"; 
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
	ots1.perf = pow(ots1.dataSize, 2)/ots1.time/1000000000;
	times[1] = ots1; 
  printf("OMP 1 Thread: %f\n", ots1.time);

	//picking OMP1 as source of sum data
	const char *out_file = "sums.csv";
	write_sums_to_csv_file(out_file, X_AXIS_SZ, sumsOmp1);

  /****************************************************************************
   * OMP Sixteen Threads
  ****************************************************************************/
  float *sumsOmp16  = new float[ sigSize ];
  
  threads = 16; 
  
  Timing ots16;
	ots16.name = "OMP 16 Threads";
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
	ots16.perf = pow(ots16.dataSize, 2)/ots16.time/1000000000;
	times[2] = ots16; 
  printf("OMP 16 Thread: %f\n", ots16.time);

  /****************************************************************************
   * SIMD
  ****************************************************************************/
  float *sumsSimd  = new float[ sigSize ];
  Timing otsSimd;
  otsSimd.name = "SIMD";
  otsSimd.threads = threads;
  otsSimd.dataSize = sigSize;

  double ompSimd_s = omp_get_wtime();
  for( int shift = 0; shift < sigSize; shift++ )
  {
     sumsSimd[shift] = SimdMulSum(&sigArray[0], &sigArray[0 + shift], sigSize);
  }
  double ompSimd_f = omp_get_wtime();
  
  otsSimd.time = ompSimd_f - ompSimd_s;
	otsSimd.perf = pow(otsSimd.dataSize, 2)/otsSimd.time/1000000000;
	times[3] = otsSimd; 
  printf("SIMD: %f\n", otsSimd.time);

  /****************************************************************************
   * OpenCL
  ****************************************************************************/
  Timing otsOpenCl;
  otsOpenCl.name = "OpenCL";
  otsOpenCl.threads = 0;
  otsOpenCl.dataSize = sigSize;
	
	FILE *clfp;
	clfp = fopen( argv[1], "r" );
	if( fp == NULL )
	{
		fprintf( stderr, "Cannot open OpenCL source file '%s'\n", CL_FILE_NAME );
		return 1;
	}
	
	// returned status from opencl calls
	cl_int status;

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

	// 3. context
 	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:
	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	//4.5 Create host destination buffer	
	float *sumsOpenCl  = new float[ sigSize ];

	// 5. buffers on device
	cl_mem dArray = clCreateBuffer( context, CL_MEM_READ_ONLY,  2*sigSize*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );
	cl_mem dSums  = clCreateBuffer( context, CL_MEM_WRITE_ONLY, 1*sigSize*sizeof(cl_float), NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

	//6. enqueue write buffer cmds
  status = clEnqueueWriteBuffer( cmdQueue, dArray, CL_FALSE, 0, sigSize, sigArray, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

	Wait(cmdQueue);

	//////////// Taken straight from proj5
	//7. read the kernel code from a file:
	
	fseek( clfp, 0, SEEK_END );
	size_t fileSize = ftell( clfp );
	fseek( clfp, 0, SEEK_SET );
	char *clProgramText = new char[ fileSize+1 ];		// leave room for '\0'
	size_t n = fread( clProgramText, 1, fileSize, clfp );
	clProgramText[fileSize] = '\0';
	fclose( clfp );
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
	const char *options = { "" };
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
	cl_kernel kernel = clCreateKernel( program, "AutoCorrelate", &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateKernel failed\n" );

	//10. args to kernel obj
	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dArray );
	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dSums  );
	
	//11. Engueue the kernel object for execution
	size_t globalWorkSize[3] = { sigSize,      1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

	Wait(cmdQueue);
  double ompOpenCl_s = omp_get_wtime();
	
	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait( cmdQueue );
	
	double ompOpenCl_f = omp_get_wtime();
	otsOpenCl.time = ompOpenCl_f - ompOpenCl_s;
	otsOpenCl.perf = pow(otsOpenCl.dataSize, 2)/otsOpenCl.time/1000000000;
	times[4] = otsOpenCl; 
	printf("OpenCL: %f\n", otsOpenCl.time);

	write_timings_to_csv_file(PERF_FILE, NUM_TESTS, times);

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

// wait until all queued tasks have taken place:

void
Wait( cl_command_queue queue )
{
      cl_event wait;
      cl_int      status;

      status = clEnqueueMarker( queue, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clEnqueueMarker failed\n" );

      status = clWaitForEvents( 1, &wait );
      if( status != CL_SUCCESS )
	      fprintf( stderr, "Wait: clWaitForEvents failed\n" );
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


void write_timings_to_csv_file(const char *file_name, int arr_sz, struct Timing *times)
{
  std::ofstream timeFile;
  
  if (!does_file_exist(file_name))
	{
    timeFile.open(file_name, std::ios::out);
		
		std::stringstream header;
		for(int h=0; h<arr_sz; h++)
		{
			header << times[h].name;
			if(h < arr_sz)
				header << ",";
		}
		timeFile << header.rdbuf() << std::endl;
	}
	else
    timeFile.open(file_name, std::ios::app);
 
	std::stringstream timing;
	for(int i=0; i < arr_sz; i++)
	{
		timing << times[i].perf;
		if(i < arr_sz)
				timing << ",";
	} 
	timeFile << timing.rdbuf() << std::endl;
	//file closed via RAII
}


void write_sums_to_csv_file(const char *file_name, int x_axis_sz, float *sums)//int blocksize, int numtrials, float perf)
{
  std::ofstream sumFile;
  
  if (!does_file_exist(file_name))
	{
    sumFile.open(file_name, std::ios::out);
		
		std::stringstream header;
		for(int i=1; i<=x_axis_sz;i++)
		{
			header << i;
			if(i < x_axis_sz)
				header << ",";
		}	
		sumFile << header.rdbuf() << std::endl;
	}
	else
    sumFile.open(file_name, std::ios::app);
 
	std::stringstream sum_row;
	for(int i=1; i<=x_axis_sz;i++)
	{
		sum_row << sums[i];
		if(i < x_axis_sz)
				sum_row << ",";
	} 
	sumFile << sum_row.rdbuf() << std::endl;
	//file closed via RAII
}

template <typename numeric> void printArray(int size, numeric *array)
{
  for(int i = 0; i < size; i++)
  {
    //printf("%d: %f\n", i, array[i]);
    std::cout << i << ": " << array[i] << std::endl; 
  }
}

/*attempted to put the OMP calls in a separate function, but didn't work */
/*struct Timing timeOmp(int threads, int size, float *sigArray, float *sums)
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
*/


