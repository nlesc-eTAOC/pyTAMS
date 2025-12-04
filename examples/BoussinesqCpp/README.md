# Boussinesq C++ example

This folder contains a C++ implementation of the Python
Boussinesq example (see examples/Boussinesq). It allows to demonstrate how to couple
pyTAMS with an external C++ code.

The C++ implementation relies on Eigen for describing the
problem matrices and uses LAPACK for solving the linear systems.
Eigen headers and LAPACK libraries need to be specified in the
Makefile.

To get the Eigen library, simply clone the repository:

```
git clone https://gitlab.com/libeigen/eigen.git
```

To get LAPACK, it will depends on your operating system. On MacOS, we recommend using homebrew:

```
brew install openblas
brew install lapack
```

The exact path of the installed LAPACK library will depend on the version and your OS version, but
it should be close to:

```
/opt/homebrew/Cellar/lapack/3.12.1/
```

A simple inter-processor communication is implemented using two-way named pipes, to
enable running the C++ implementation in the background while
the Python code runs TAMS.

The C++ code can be compiled with `make` (provided that you update the path
to the Eigen and LAPACK libraries !).
