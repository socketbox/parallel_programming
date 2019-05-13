simd.p4.o:	simd.p4.h  simd.p4.cpp
		g++  -c  simd.p4.cpp -o simd.p4.o

arraymult:	boettchc_cs475-400_project4.cpp simd.p4.o
		g++ $(CPPFLAGS) -lm -fopenmp simd.p4.o -o arraymult  boettchc_cs475-400_project4.cpp 

clean:
		rm -f arraymult
		rm -f arraymult.csv
		rm -f arraymult.tgz

dist:
		tar zcvf arraymult.tgz *.cpp *.h
		scp arraymult.tgz flip\:./cs475/week5/

