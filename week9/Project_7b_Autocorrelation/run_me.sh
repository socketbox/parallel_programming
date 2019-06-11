#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

for i in {1..12}
	do 
		./proj7 project7b.cl signal.txt
	done

exit
