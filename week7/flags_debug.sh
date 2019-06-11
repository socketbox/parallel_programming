#!/bin/sh
#export CXXFLAGS="-std=c++11 -g -Wall -pedantic -Wextra -Wno-c++11-extensions" 
#export CXXFLAGS="-std=c++11 -g -Wall -pedantic -Wextra" 
#export CXXFLAGS="-lm -fopenmp -O3 -std=c++11 -g -fstack-protector-all -Werror=format-security -Wall -pedantic -Wextra" 
export CXXFLAGS="-std=c++11 -g -Wall -pedantic -Wextra -lm -fopenmp -L/usr/lib/x86_64-linux-gnu
-lOpenCL -I/usr/include/CL" 