EIGEN_HOME = ../../../Eigen/eigen/
LAPACK_HOME = /opt/homebrew/Cellar/lapack/3.12.1/
executable = boussinesq.exe

all: exec

INC_ALL = -I eigen_lapack_interf.h -I Messaging.h -I Boussinesq.h -I $(EIGEN_HOME)

exec:
	g++ -O3 -std=c++20 $(INC_ALL) -L $(LAPACK_HOME)/lib/ -llapack Boussinesq.cpp -o $(executable)

debug:
	g++ -O0 -g -Wall -std=c++20 $(INC_ALL) -L $(LAPACK_HOME)/lib/ -llapack Boussinesq.cpp -o $(executable)

clean:
	rm -f $(executable)
