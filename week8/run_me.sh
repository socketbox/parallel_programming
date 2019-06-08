#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

NVCC="/usr/local/apps/cuda/cuda-9.2/bin/nvcc"
GCC="/usr/local/common/gcc-6.3.0/bin/"
#NUMTRIALS
for i in 16000 32000 64000 128000 256000 512000 
  do
    echo "$i"
		#BLOCKSIZEs 
		for b in 16 32 64
      do
        $NVCC -o proj6 -DNUMTRIALS=$i -DBLOCKSIZE=$b boettchc_cs475-400_project6.cu &&
        ./proj6
        rm -f proj6
    done
done
