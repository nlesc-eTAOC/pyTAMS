# Boussinesq

This an implementation of the 2D Boussinesq model described in
[J. Soons et al.](https://doi.org/10.1017/jfm.2025.248).

Note that most of the model parameters are not explicitly exposed to
the user as input parameters, but can be easily changed in the sources. 

## Model parameters:
 - `size_M`: number of cells in latitude
 - `size_N`: number of cells in depth
 - `epsilon`: the noise amplitude
 - `K`: number of Fourier modes in the freshwater forcing


## Score function:

The default score function is a measure of the locally averaged
stream function in the southern region. Alternative score function
can be used by setting the `score_method` parameters.
