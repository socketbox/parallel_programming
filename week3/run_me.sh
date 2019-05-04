#!/usr/local/bin/bash
set -euo pipefail
IFS=$'\n\t'

GXX="/usr/local/bin/g++-8"
declare -a THREADS=(1 2 4 8 16 32)
declare -a NUMNODES=(4 16 64 256 1024 4096 8192 16384)

for t in "${THREADS[@]}"; do
  for n in "${NUMNODES[@]}"; do
    $GXX -lm -fopenmp -std=c++11 -g -DNUMT=$t -DNUMNODES=$n -o project2 boettchc_cs475-400_project_2.cpp &&
        ./project2
    rm -f project2
  done
done
echo
exit
