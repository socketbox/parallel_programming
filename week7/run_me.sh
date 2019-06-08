#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

GXX="/usr/bin/g++-8"
FLAGS=(-w -std=c++11 -lm -fopenmp -L/usr/lib/x86_64-linux-gnu -lOpenCL -I/usr/include/CL)
declare -a NMB=(1 4 8 16 32 64 128 256 1024 4096 8192 12288)
declare -a LOCAL_SIZE=(8 16 32 64 128 192 256)
declare -a REDUCE_LOCAL_SIZE=(32 64 128 256)
for t in "${NMB[@]}"; do
  for n in "${LOCAL_SIZE[@]}"; do
    $GXX -o project5 boettchc_cs475-400_project5.cpp "${FLAGS[@]}" -DNMB=$t -DLOCAL_SIZE=$n && ./project5
    rm -f project5
  done
done
for t in "${NMB[@]}"; do
  for n in "${LOCAL_SIZE[@]}"; do
    $GXX -o project5 boettchc_cs475-400_project5.cpp "${FLAGS[@]}" -DSUMMING=1 -DNMB=$t -DLOCAL_SIZE=$n && ./project5
    rm -f project5
  done
done
for t in "${NMB[@]}"; do
  for n in "${REDUCE_LOCAL_SIZE[@]}"; do
    $GXX -o project5_reduce boettchc_cs475-400_project5_reduce.cpp "${FLAGS[@]}" -DREDUCE_LOCAL_SIZE=$n -DNMB=$t && ./project5_reduce
    rm -f project5_reduce
  done
done
exit
