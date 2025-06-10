EIGEN_HOME = ../../../eigen/
LAPACK_HOME = /opt/homebrew/Cellar/lapack/3.12.0/
executable = boussinesq.exe

all: exec

exec:
	g++ -O3 -std=c++20 -I eigen_lapack_interf.h -I Boussinesq.h -I $(EIGEN_HOME) -L $(LAPACK_HOME)/lib/ -llapack Boussinesq.cpp -o $(executable)

debug:
	g++ -O0 -g -Wall -std=c++20 -I eigen_lapack_interf.h -I Boussinesq.h -I $(EIGEN_HOME) -L $(LAPACK_HOME)/lib/ -llapack Boussinesq.cpp -o $(executable)

clean:
	rm -f $(executable)
