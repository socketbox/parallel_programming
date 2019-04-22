#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

GXX="/usr/local/bin/g++-8"
for i in 10000 100000 500000 1000000 5000000
  do
    echo "$i"
    for t in 1 2 4 6 8 16
      do
        echo "$t"
        $GXX -lm -fopenmp -std=c++11 -g -fstack-protector-all -Werror=format-security -Wall\
        -pedantic -Wextra -DNUMT=$t -DNUMTRIALS=$i -o project1 boettchc_cs475-400_project1.cpp &&
        ./project1
        rm -f project1
        echo $?
    done
done
