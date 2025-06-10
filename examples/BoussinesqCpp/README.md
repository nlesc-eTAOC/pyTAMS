# Boussinesq C++ example

This folder contains a C++ implementation of the Python
Boussinesq example. It allows to demonstrate how to couple
pyTAMS with an external C++ code.

The C++ implementation relies on Eigen for describing the
problem matrices and uses LAPACK for solving the linear systems.
Eigen headers and LAPACK libraries need to be specified in the
Makefile.

The C++ code can be compiled with `make`.
