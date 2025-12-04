# Boussinesq

This an implementation of the 2D Boussinesq model described in
[J. Soons et al.](https://doi.org/10.1017/jfm.2025.248).

Note that most of the model parameters are not explicitly exposed to
the user as input parameters, but can be easily changed in the sources. 

## Model parameters:
 - `size_M`: [DEF = 40] number of cells in latitude
 - `size_N`: [DEF = 80] number of cells in depth
 - `epsilon`: [DEF = 0.01] the noise amplitude
 - `K`: [DEF = 7] number of Fourier modes in the freshwater forcing
 - `delta_stoch`: [DEF = 0.05] thickness (depth) of the surface noise signal 


## Score function:

The default score function is a measure of the locally averaged
stream function in the southern region. Alternative score function formulation
can be used by setting the `score_method` parameters.

 - `score_method`: [DEF = "default"] Beside the default method, the score function can also
 be computed using a POD projection (`score_method = "PODDecomp"`) in which case a file containing
 POD data must be provided using `pod_data_file`. When using the edge tracking algorithm, setting
 `score_method = "EdgeTracking"` is necessary.

In addition to the static score formulation above, when using TAMS it is important to introduce
a time dependence to the score function to avoid *late extinction*, where the maximum of the score
function is reached near the end of the time horizon $`T_a`$ and the remaining time is not sufficient to
allow the model to complete its transition.
To account for this, set `score_time_dep = true` in the input TOML file and provide a typical
time scale $`\tau_t`$ of the system using `score_time_scale = <some_value>`. The static score function
described above is then multiplied by a time factor $`t_f`$:

```math
t_f = 1.0 - \exp((t - T_a) / \tau_t)
```

## Hosing

The model also allows to perform *hosing* experiments to explore bifurcation-induced transition.
The hosing parameters are as follows:
 - `hosing_rate` : the rate of change of the hosing parameter
 - `hosing_start` : the time $`t_0`$ when the rate of change starts to apply
 - `hosing_start_val` : the value of the hosing parameter at $`t_0`$ or below

 Note that the hosing parameter value remains at its `hosing_start_val` value at model time below $`t_0`$.
