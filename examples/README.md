# pyTAMS examples

*pyTAMS* comes with a suite of examples referred to throughout the documentation
and useful to initiate your own cases.

## BiChannel2D
This is a 2D overdamped diffusion process, used in the Tutorials.

## Boussinesq
A high-dimensional SPDE model (~10^4 DoFs), describing collapseof
the AMOC. It provides a complete example of a complex phyical model
implementation in *pyTAMS* (albeit fully in Python).

## BoussinesqCpp
A C++ version of the Boussinesq model, showing how to couple
*pyTAMS* with an external software.

## DoubleWell2D
A 2D double well overdamped diffusion process, used for validation of
the core algorithm.

## MOC
A box model of the Atlantic ocean, providing an example of somewhat
more complex models than the overdamped diffusion processes.

## OrnsteinUhlenbeck
A 1D Ornstein-Uhlenbeck process, used in validating *pyTAMS*.
