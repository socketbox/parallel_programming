#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

GXX="/usr/bin/g++-8"
declare -a ARRSZ=(1000 10000 100000 1000000 10000000 100000000)

for n in "${ARRSZ[@]}"; do
  make CPPFLAGS=-DARRSZ=$n arraymult && ./arraymult
  rm -f arraymult
done
exit
