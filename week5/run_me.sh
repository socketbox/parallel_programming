#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

GXX="/usr/bin/g++-8"
declare -a ARRSZ=(1000 10000 100000 500000 1000000)

for n in "${ARRSZ[@]}"; do
  make CPPFLAGS=-DARRSZ=$n proj4 && ./proj4
  rm -f proj4
done
exit
