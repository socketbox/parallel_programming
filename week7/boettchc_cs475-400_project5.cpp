#include <fstream>
#include <string>
#include <cstdio>
#include <cmath>
#include <string>
#include <cstdlib>

#include <omp.h>
#include <cl.h>
#include <cl_platform.h>

#ifndef NMB
#define	NMB   64	
#endif

//chb: (compute units*max work group size)/1024
#define RX480   9

//chb: are we summing or not?
#ifndef SUMMING
#define SUMMING 0
#endif

#define NUM_ELEMENTS		NMB*RX480*1024

#ifndef LOCAL_SIZE
#define	LOCAL_SIZE		64
#endif

#define	NUM_WORK_GROUPS		NUM_ELEMENTS/LOCAL_SIZE

const char *			CL_FILE_NAME = { "project5.cl" };
const float			TOL = 0.0001f;

void Wait( cl_command_queue );
int	LookAtTheBits( float );
bool does_file_exist (const std::string& name);
void write_csv(char *prog_name, int n, int local_sz, int work_grps, double perf);

int
main( int argc, char *argv[ ] )
{

 //fprintf(stderr, "chb: this is NMB and LOCAL_SIZE: %i, %i\n", NMB, LOCAL_SIZE);  
  
  // see if we can even open the opencl kernel program
	// (no point going on if we can't):

	FILE *fp;
	fp = fopen( CL_FILE_NAME, "r" );
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

	// 2. allocate the host memory buffers:

	float *hA = new float[ NUM_ELEMENTS ];
	float *hB = new float[ NUM_ELEMENTS ];
	float *hC = new float[ NUM_ELEMENTS ];
  float *hD = nullptr;

  if(SUMMING)
  {
    hD = new float[ NUM_ELEMENTS ];
  }

	// fill the host memory buffers:
	for( int i = 0; i < NUM_ELEMENTS; i++ )
	{
		hA[i] = hB[i] = (float) sqrt(  (double)i  );
	}

	size_t dataSize = NUM_ELEMENTS * sizeof(float);

	// 3. create an opencl context:

	cl_context context = clCreateContext( NULL, 1, &device, NULL, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateContext failed\n" );

	// 4. create an opencl command queue:

	cl_command_queue cmdQueue = clCreateCommandQueue( context, device, 0, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateCommandQueue failed\n" );

	// 5. allocate the device memory buffers:

	cl_mem dA = clCreateBuffer( context, CL_MEM_READ_ONLY,  dataSize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (1)\n" );

	cl_mem dB = clCreateBuffer( context, CL_MEM_READ_ONLY,  dataSize, NULL, &status );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clCreateBuffer failed (2)\n" );

  cl_mem dC;
  
  cl_mem dD = nullptr;
  
  if(!SUMMING)
  {
    dC = clCreateBuffer( context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status );
    if( status != CL_SUCCESS )
      fprintf( stderr, "clCreateBuffer failed (3)\n" );
  }
  else
  {
      //chb: changing to R/W for verstatility
	    dC = clCreateBuffer( context, CL_MEM_READ_ONLY, dataSize, NULL, &status );
	    if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateBuffer failed (3)\n" );

      dD = clCreateBuffer( context, CL_MEM_WRITE_ONLY, dataSize, NULL, &status );
      if( status != CL_SUCCESS )
        fprintf( stderr, "clCreateBuffer failed (4)\n" );
  }

	// 6. enqueue the 2 commands to write the data from the host buffers to the device buffers:
	status = clEnqueueWriteBuffer( cmdQueue, dA, CL_FALSE, 0, dataSize, hA, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (1)\n" );

	status = clEnqueueWriteBuffer( cmdQueue, dB, CL_FALSE, 0, dataSize, hB, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (2)\n" );

  //chb: write to C regardless--it can always be overwritten
	status = clEnqueueWriteBuffer( cmdQueue, dC, CL_FALSE, 0, dataSize, hC, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueWriteBuffer failed (3)\n" );

Wait( cmdQueue );

	// 7. read the kernel code from a file:

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

	// 10. setup the arguments to the kernel object:

	status = clSetKernelArg( kernel, 0, sizeof(cl_mem), &dA );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (1)\n" );

	status = clSetKernelArg( kernel, 1, sizeof(cl_mem), &dB );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (2)\n" );

	status = clSetKernelArg( kernel, 2, sizeof(cl_mem), &dC );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clSetKernelArg failed (3)\n" );

  if(SUMMING)
  {
    status = clSetKernelArg( kernel, 3, sizeof(cl_mem), &dD );
    if( status != CL_SUCCESS )
      fprintf( stderr, "clSetKernelArg failed (4)\n" );
  }
  else
  {
    status = clSetKernelArg( kernel, 3, sizeof(cl_mem), NULL );
    if( status != CL_SUCCESS )
      fprintf( stderr, "clSetKernelArg failed (4)\n" );
  }

	// 11. enqueue the kernel object for execution:

	size_t globalWorkSize[3] = { NUM_ELEMENTS, 1, 1 };
	size_t localWorkSize[3]  = { LOCAL_SIZE,   1, 1 };

	Wait( cmdQueue );
	double time0 = omp_get_wtime( );

	time0 = omp_get_wtime( );

	status = clEnqueueNDRangeKernel( cmdQueue, kernel, 1, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL );
	if( status != CL_SUCCESS )
		fprintf( stderr, "clEnqueueNDRangeKernel failed: %d\n", status );

	Wait( cmdQueue );
	double time1 = omp_get_wtime( );

	// 12. read the results buffer back from the device to the host:

	status = clEnqueueReadBuffer( cmdQueue, dC, CL_TRUE, 0, dataSize, hC, 0, NULL, NULL );
	if( status != CL_SUCCESS )
			fprintf( stderr, "clEnqueueReadBuffer failed\n" );

	// did it work?

	for( int i = 0; i < NUM_ELEMENTS; i++ )
	{
		float expected = hA[i] * hB[i];
		if( fabs( hC[i] - expected ) > TOL )
		{
			//fprintf( stderr, "%4d: %13.6f * %13.6f wrongly produced %13.6f instead of %13.6f (%13.8f)\n",
				//i, hA[i], hB[i], hC[i], expected, fabs(hC[i]-expected) );
			//fprintf( stderr, "%4d:    0x%08x *    0x%08x wrongly produced    0x%08x instead of    0x%08x\n",
				//i, LookAtTheBits(hA[i]), LookAtTheBits(hB[i]), LookAtTheBits(hC[i]), LookAtTheBits(expected) );
		}
	}

	//fprintf( stderr, "%8d\t%4d\t%10d\t%10.3lf GigaMultsPerSecond\n",
		//NMB, LOCAL_SIZE, NUM_WORK_GROUPS, (double)NUM_ELEMENTS/(time1-time0)/1000000000. );
	
  write_csv(argv[0], NUM_ELEMENTS, LOCAL_SIZE, NUM_WORK_GROUPS, \
      (double)NUM_ELEMENTS/(time1-time0)/1000000000. );

	// 13. clean everything up:

	clReleaseKernel(        kernel   );
	clReleaseProgram(       program  );
	clReleaseCommandQueue(  cmdQueue );
	clReleaseMemObject(     dA  );
	clReleaseMemObject(     dB  );
	clReleaseMemObject(     dC  );

	delete [ ] hA;
	delete [ ] hB;
	delete [ ] hC;
	
  if(SUMMING)
  {
    clReleaseMemObject( dD );
    delete [ ] hD;
  }

	return 0;
}


int
LookAtTheBits( float fp )
{
	int *ip = (int *)&fp;
	return *ip;
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


bool does_file_exist (const std::string& name) 
{
    if (FILE *file = fopen(name.c_str(), "r")) {
        fclose(file);
        return true;
    } else {
        return false;
    }   
} 

void write_csv(char *prog_name, int n, int local_sz, int work_grps, double perf) 
{
  std::string filename(prog_name);
  filename += ".csv";

	char *header = "Elements,Local Size,Work Groups,Performance\n";

  //std::ofstream out_file;
  FILE *out_file;
  
  if (!does_file_exist(filename))
  {
    //out_file.open(filename, std::ios::out);
    out_file = std::fopen(filename.c_str(), "w"); 
    fprintf(out_file, header);
  }
  else
    out_file = std::fopen(filename.c_str(), "a");
  
  fprintf( out_file, "%8d,%4d,%10d,%10.3lf\n",
		n, local_sz, work_grps, perf);
  //out_file << datum.arrsz << "," << datum.perf_sum << "," << datum.t_vanilla << "," << datum.t_simd\
   // << "," << datum.perf_ms << "," << datum.t_vanilla_ms << "," << datum.t_simd_ms << std::endl;
  //file closed via RAII
}


