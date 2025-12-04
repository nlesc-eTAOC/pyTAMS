# pyTAMS

[![RSD](https://img.shields.io/badge/rsd-pyTAMS-222c71.svg)](https://research-software-directory.org/software/pytams)
[![DOI](https://img.shields.io/badge/DOI-10.5281/15349506-222c71.svg)](https://zenodo.org/doi/10.5281/zenodo.15349506)
[![build](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/build.yml/badge.svg)](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/build.yml)
[![PyPI version](https://badge.fury.io/py/pytams.svg)](https://badge.fury.io/py/pytams)
[![sonarcloud](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/sonarcloud.yml/badge.svg)](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/sonarcloud.yml)
[![workflow scc badge](https://sonarcloud.io/api/project_badges/measure?project=nlesc-eTAOC_pyTAMS&metric=coverage)](https://sonarcloud.io/dashboard?id=nlesc-eTAOC_pyTAMS)
[![github license badge](https://img.shields.io/github/license/nlesc-eTAOC/pyTAMS)](https://github.com/nlesc-eTAOC/pyTAMS)


## Overview

Rare events algorithms are powerful techniques allowing to sample rare occurrences of a computational model
at a much lower cost than brute force Monte-Carlo. However, running such algorithms on models featuring more
than a handull of dimensions become cumbersome as both compute and memory requirements increase.
*pyTAMS* is a modular implementation of the trajectory-adaptive multilevel splitting (TAMS) rare event method
introduced by [Lestang et al.](https://doi.org/10.1088/1742-5468/aab856), aiming at alleviating the difficulty
of performing rare event algorithms for to high-dimensional systems such as the ones encountered in geophysical
or engineering applications.


## Installation

To install *pyTAMS* from GitHub repository, do:

```console
git clone git@github.com:nlesc-eTAOC/pyTAMS.git
cd pyTAMS
python -m pip install .
```

Note that the latest version of *pyTAMS* is available on PyPI [here](https://pypi.org/project/pytams/)
and can be installed with `pip install pytams`, but built-in examples are not readily available using
the PyPI version.

To run the example cases shipped with *pyTAMS*, additional dependencies are required.
To install the examples dependencies, run:

```console
python -m pip install .[exec]
```

## Quick start

To get started with *pyTAMS*, let's have a look at the classical double-well potential case.
Although it is not a high-dimensional system, it provides a good overview of *pyTAMS* capabilities.
A 2D version of the double-well is available in the [examples](examples) folder. To run the case,
simply do:

```console
cd examples/DoubleWell2D
python tams_dw2dim.py
```

This minimal example runs TAMS 10 times in order to get an estimate of the transition probability
as well as the corresponding relative error. For a more in-depth explanation about this case, setting up the
model and running the simulations, have a look at the tutorial [here](https://nlesc-eTAOC.github.io/pyTAMS/Tutorials.html).

## Documentation

[![doc](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/documentation.yml/badge.svg)](https://github.com/nlesc-eTAOC/pyTAMS/actions/workflows/documentation.yml)

*pyTAMS* documentation is hosted on GitHub [here](https://nlesc-etaoc.github.io/pyTAMS/)

## Contributing

If you want to contribute to the development of *pyTAMS*,
have a look at the [contribution guidelines](CONTRIBUTING.md).

## Acknowledgements

The development of *pyTAMS* was supported by the Netherlands eScience Center
in collaboration with the Institute for Marine and Atmospheric research Utrecht [IMAU](https://www.uu.nl/onderzoek/imau).

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
