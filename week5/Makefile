simd.p4.o:	simd.p4.h  simd.p4.cpp
		g++  -c  simd.p4.cpp -o simd.p4.o

proj4:	boettchc_cs475-400_project4.cpp simd.p4.o
		g++ $(CPPFLAGS) -std=c++11 -lm -fopenmp simd.p4.o -o proj4  boettchc_cs475-400_project4.cpp 

clean:
		rm -f proj4
		rm -f proj4.csv
		rm -f proj4.tgz

dist:
		tar zcvf proj4.tgz *.cpp *.h Makefile run_me.sh
		scp proj4.tgz flip\:./cs475/week5/

