#!/bin/bash
GPP4=/usr/bin/g++
GPP8=/usr/local/common/gcc-8.2.0/bin/g++
$GPP4 -c -o simd.p7b.o simd.p7b.cpp
$GPP4 -fopenmp -I/scratch/cuda-7.0/include/ -I/scratch/cuda-7.0/include/CL/ /scratch/cuda-7.0/lib64/libOpenCL.so simd.p7b.o -o proj7 project_7b.cpp
