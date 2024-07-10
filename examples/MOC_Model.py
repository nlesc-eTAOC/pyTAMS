from typing import Any
import netCDF4 as netcdf
import numpy as np
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS


class MOCModel(ForwardModelBaseClass):
    """Meridional overturning circulation box model."""

    def _init_model(self,
                    params: dict = None,
                    ioprefix: str = None):
        """E_A is the forcing amplitude.

        opt_params: "temp_params" to give lambda_a
                    "ice_params" to give [f_ice, q_factor_min]
        """
        self.E_A = 0.25
        self._noise_level = 0.1
        # self.forcing = None
        # seed = 0
        # self.rng = np.random.Generator(seed)

        datafolder="./MOC_files/"

        self.convention = ['S_t', 'S_ts', 'S_n', 'S_s', 'S_d', 'D',
                           'T_t', 'T_ts', 'T_n', 'T_s', 'T_d']

    # Read optional temp and ice parameters
        self.temp, self.ice = False, False
        # for key, params in opt_params.items():
        #     if key == "temp_params":
        #         self.lambda_a = params
        #         self.temp = True
        #     elif key == "ice_params":
        #         self.f_ice, self.q_factor_min = params
        #         self.ice = True

        # Load additional parameters and prepare the steady states files
        base_name = datafolder + "MOC_box_model"
        if not self.temp:
            list_name_init = [base_name, "equilibrium_states"]
        else:
            list_name_params = [base_name, "temp", "lambda_a", self.str_param(self.lambda_a)]
            if self.ice:
                list_name_init = [base_name, "ice", "equilibrium_states",
                                  "lambda_a", self.str_param(self.lambda_a),
                                  "q_min", self.str_param(self.q_factor_min)]
            else:
                list_name_init = [base_name, "temp", "equilibrium_states",
                                  "lambda_a", self.str_param(self.lambda_a)]

        if self.temp or self.ice:
            filename_parameter = "_".join(list_name_params) + ".txt"
            self.T_ts_a, self.T_n_a, self.S_t, self.S_n, self.S_ts, \
                self.S_s, self.T_t, self.T_n, self.T_ts, self.T_s, self.T_d, self.D = np.loadtxt(filename_parameter)
            self.T_t_a, self.T_s_a = np.copy(self.T_ts_a), np.copy(self.T_n_a)
            self.T_t_a, self.T_ts_a, self.T_n_a, self.T_s_a = self.T_t_a+273.15, \
                self.T_ts_a+273.15, self.T_n_a+273.15, self.T_s_a+273.15

        self.filename_init = "_".join(list_name_init) + ".nc"

        # Basic initializations
        self.init_constant_params()
        self.load_init_state()

        self._state = self.initCondition()
        self.init_var_params()

    @classmethod
    def name(self):
        """Return the model name."""
        return "MOCModel"

    def initCondition(self):
        """Return the initial conditions."""
        return self.on

    def getCurState(self):
        """Override the template."""
        return self._state

    def setCurState(self, state):
        """Override the template."""
        self._state = state

    # def reset_seed(self, seed):
    #     self.rng.bit_generator.state = np.random.PCG64(seed).state

    def set_E_A(self, E_A):
        """Sets the forcing amplitude and reinitializes the model."""
        self.E_A = E_A
        self.load_init_state()

    def set_forcing(self, forcing_function, forcing_params, *fixed_args, **fixed_kwargs):
        """Sets the forcing function."""
        self.forcing_params = forcing_params
        self.forcing = lambda x, t, fp: forcing_function(x, t, fp, *fixed_args, **fixed_kwargs)

    def str_param(self, p):
        """Returns a string representation of a parameter."""
        return f'{p:.{int(str(p)[-1])+1}f}'.replace(".", "_")

    def load_init_state(self):
        """Loads St, Sts, Sn, Ss, D."""
        #Get the steady state and use these as input parameters, but these are freely allowed to evolve
        fh         = netcdf.Dataset(self.filename_init, 'r')
        E_A_values = fh.variables['E_A'][:]
        q_N        = fh.variables['q_n'][:]

        #Select all the indices for the AMOC on state
        end_branch_on  = np.where(q_N == 0)[0][0]
        self.E_A_index_on    = (np.abs(E_A_values[:end_branch_on] - self.E_A)).argmin()

        #Select all the indices for the AMOC off state
        branch_off_index   = np.where(q_N[np.argmax(E_A_values):] >= 5)[0][0] + np.argmax(E_A_values)
        self.E_A_index_off = (np.abs(E_A_values[np.argmax(E_A_values):branch_off_index] - self.E_A)).argmin() \
            + np.argmax(E_A_values)

        self.E_A *= 1e6

        self.on = np.array([fh.variables['S_t'][self.E_A_index_on],
                            fh.variables['S_ts'][self.E_A_index_on],
                            fh.variables['S_n'][self.E_A_index_on],
                            fh.variables['S_s'][self.E_A_index_on],
                            fh.variables['S_d'][self.E_A_index_on],
                            fh.variables['D'][self.E_A_index_on]])

        self.off = np.array([fh.variables['S_t'][self.E_A_index_off],
                            fh.variables['S_ts'][self.E_A_index_off],
                            fh.variables['S_n'][self.E_A_index_off],
                            fh.variables['S_s'][self.E_A_index_off],
                            fh.variables['S_d'][self.E_A_index_off],
                            fh.variables['D'][self.E_A_index_off]])

        if self.temp or self.ice:
            self.on = np.concatenate((self.on,
                                      np.array([fh.variables['T_t'][self.E_A_index_on],
                                                fh.variables['T_ts'][self.E_A_index_on],
                                                fh.variables['T_n'][self.E_A_index_on],
                                                fh.variables['T_s'][self.E_A_index_on],
                                                fh.variables['T_d'][self.E_A_index_on]])))
            self.off = np.concatenate((self.off,
                                       np.array([fh.variables['T_t'][self.E_A_index_off],
                                                 fh.variables['T_ts'][self.E_A_index_off],
                                                 fh.variables['T_n'][self.E_A_index_off],
                                                 fh.variables['T_s'][self.E_A_index_off],
                                                 fh.variables['T_d'][self.E_A_index_off]])))

        fh.close()

        self.q_N_on, self.q_N_off = q_N[self.E_A_index_on]*1e6, q_N[self.E_A_index_off]*1e6
        self.q_N_off_full = self.get_q_N(self.off)

    def init_constant_params(self):
        """Set model constants."""
        self.V_0    = 3.0 * 10**17.0    #Total volume of basin
        self.V_n    = 3.0 * 10**15.0    #Volume of northern box
        self.V_s    = 9.0 * 10**15.0    #Volume of southern box
        self.A      = 1.0 * 10**14.0    #Area of Atlantic pycnocline
        self.L_xA   = 1.0 * 10**7.0     #Zonal extent of Atlantic Ocean at southern end
        self.L_y    = 1.0 * 10**6.0     #Meridional extent of frontal region
        self.L_xS   = 3.0 * 10**7.0     #Zonal extent of Southern Ocean
        self.tau    = 0.1               #Wind stress
        self.A_GM   = 1700.0            #Eddy diffusivity
        self.f_s    = -10**(-4.0)       #Coriolis parameter
        self.rho_0  = 1027.5            #Reference density
        self.kappa  = 10**(-5.0)        #Vertical diffusivity
        self.S_0    = 35.0              #Reference salinity
        self.T_0    = 5.0               #Reference temperature

        if not self.temp and not self.ice:
            self.T_n    = 5.0               #Temperature of northern box
            self.T_ts   = 10.0              #Temperature of TS-box

        self.eta    = 3.0 * 10**4.0     #Hydraulic constant
        self.alpha  = 2.0 * 10**(-4.0)  #Thermal expansion coefficient
        self.beta   = 8.0 * 10**(-4.0)  #Haline contraction coefficient
        self.r_S    = 1.0 * 10**7.0     #Transport by the southern subtropical gyre
        self.r_N    = 5.0 * 10**6.0     #Transport by the northern subtropical gyre
        self.E_S    = 0.17 * 10**6.0    #Symmetric freshwater flux

        self.q_Ek   = (self.tau * self.L_xS) / (self.rho_0 * np.fabs(self.f_s))    #Ekman transport

        if self.temp:
            self.A_t  = np.copy(self.A)
            self.A_ts = self.L_y * self.L_xA
            self.A_s  = self.L_y * self.L_xS
            self.A_n  = np.copy(self.A_ts)

    def init_var_params(self):
        """Only called at the initialization of the model."""
        if not self.temp and not self.ice:
            S_t, S_ts, S_n, S_s, S_d, self.D = self._state.T
        else:
            S_t, S_ts, S_n, S_s, S_d, self.D, T_t, T_ts, T_n, T_s, T_d = self._state.T

        self.update_fluxes(self._state)
        self.update_volumes()

        self.VS_0   = self.V_0 * self.S_0
        self.VS_t   = self.V_t * S_t
        self.VS_ts  = self.V_ts * S_ts
        self.VS_n   = self.V_n * S_n
        self.VS_s   = self.V_s * S_s
        self.VS_d   = self.VS_0 - self.VS_n - self.VS_t - self.VS_ts - self.VS_s   # or self.VS_d = self.V_d * S_d

        if self.temp:
            #Ocean Heat content (assume constant density)
            self.OHC_t  = self.V_t * (T_t+273.15)
            self.OHC_ts = self.V_ts * (T_ts+273.15)
            self.OHC_n  = self.V_n * (T_n+273.15)
            self.OHC_s  = self.V_s * (T_s+273.15)
            self.OHC_d  = self.V_d * (T_d+273.15)

    def _dW(self, dt, noise):
        """Stochastic forcing."""
        return self._noise_level * self.E_A * self.S_0 * noise * np.sqrt(dt * (100*365*86400))

    def sea_ice(self, temp):
        """Returns the AMOC reduction factor for a given response function."""
        #Determine the sea-ice fraction over box n
        ice_n = (5 - temp) * self.f_ice
        ice_n = np.clip(ice_n, 0, 100)

        #Determine the AMOC reduction factor
        q_factor = ((1 - self.q_factor_min)/2.0) * np.tanh((50 - ice_n) * 0.1) + (1 + self.q_factor_min)/2.0

        return q_factor

    def Heaviside(self, x):
        """Returns 1 for positive values, 0 otherwise."""
        return x > 0

    def system(self, state, dw, dt, forcing=0):
        """Computes the differential equations at each timestep."""
        if not self.temp:
            S_t, S_ts, S_n, S_s, S_d, self.D = state
        else:
            S_t, S_ts, S_n, S_s, S_d, self.D, T_t, T_ts, T_n, T_s, T_d = state

        deltas = np.zeros(len(state))

        hqs, hqn = self.Heaviside(self.q_S), self.Heaviside(self.q_N)

        # T-box
        deltas[0] = self.q_S * (hqs * S_ts + (1-hqs) * S_t) + \
            self.q_U * S_d - hqn * self.q_N * S_t + self.r_S * (S_ts - S_t) + \
            self.r_N * (S_n - S_t) + 2.0 * self.E_S * self.S_0
        # TS-box
        deltas[1] = self.q_Ek * S_s - self.q_e * S_ts - self.q_S * (hqs * S_ts + (1-hqs) * S_t) + \
            self.r_S * (S_t - S_ts)
        # N-box
        deltas[2] = hqn * self.q_N * (S_t - S_n) + self.r_N * (S_t - S_n) - \
            (self.E_S + self.E_A) * self.S_0
        # S-box
        deltas[3] = self.q_S * (hqs * S_d + (1-hqs) * S_s) + self.q_e * S_ts - self.q_Ek * S_s - \
            (self.E_S - self.E_A) * self.S_0
        # Depth pycnocline
        deltas[4] = self.q_U + self.q_Ek - self.q_e - hqn * self.q_N
        deltas[4] /= (self.A + 0.5 * self.L_xA * self.L_y)

        self.VS_t, self.VS_ts, self.VS_n, \
            self.VS_s, self.D = np.array([self.VS_t, self.VS_ts, self.VS_n, self.VS_s, self.D]) \
            + deltas[:5] * dt
        self.VS_n -= dw
        self.VS_s += dw

        # D-box, conservation of salt
        self.VS_d = self.VS_0 - self.VS_n - self.VS_t - self.VS_ts - self.VS_s

        self.VS_n -= forcing
        self.VS_0 -= forcing

        if self.temp:
            T_t, T_ts, T_n, T_s, T_d = T_t+273.15, T_ts+273.15, T_n+273.15, T_s+273.15, T_d+273.15
            deltas[5] = self.q_S * (hqs * T_ts + (1-hqs) * T_t) + self.q_U * T_d - \
                hqn * self.q_N * T_t + self.r_S * (T_ts - T_t) + self.r_N * (T_n - T_t) - \
                self.A_t * self.lambda_a * (T_t - self.T_t_a)
            deltas[6] = self.q_Ek * T_s - self.q_e * T_ts - self.q_S * (hqs * T_ts + (1-hqs) * T_t) + \
                self.r_S * (T_t - T_ts) - self.A_ts * self.lambda_a * (T_ts - self.T_ts_a)
            deltas[7] = hqn * self.q_N * (T_t - T_n) + self.r_N * (T_t - T_n) - \
                self.A_n * self.lambda_a * (T_n - self.T_n_a)
            deltas[8] = self.q_S * (hqs * T_d + (1-hqs) * T_s) + self.q_e * T_ts - self.q_Ek * T_s - \
                self.A_s * self.lambda_a * (T_s - self.T_s_a)
            deltas[9] = hqn * self.q_N * T_n - self.q_S * (hqs * T_d + (1-hqs) * T_s) - self.q_U * T_d

            self.OHC_t, self.OHC_ts, self.OHC_n, \
                self.OHC_s, self.OHC_d = np.array([self.OHC_t, self.OHC_ts, self.OHC_n, self.OHC_s, self.OHC_d]) \
                + deltas[5:] * dt

    def update_volumes(self):
        """Update the volume of each box."""
        self.V_t  = self.A * self.D
        self.V_ts = self.L_xA * self.L_y * self.D / 2.0
        self.V_d  = self.V_0 - self.V_n - self.V_s - self.V_t - self.V_ts

    def update_vars(self):
        """Update the variables and return the state vector."""
        self.update_volumes()

        S_t  = self.VS_t / self.V_t
        S_ts = self.VS_ts / self.V_ts
        S_n  = self.VS_n / self.V_n
        S_s  = self.VS_s / self.V_s
        S_d  = self.VS_d / self.V_d

        if self.temp:
            T_t  = self.OHC_t / self.V_t - 273.15
            T_ts = self.OHC_ts / self.V_ts - 273.15
            T_n  = self.OHC_n / self.V_n - 273.15
            T_s  = self.OHC_s / self.V_s - 273.15
            T_d  = self.OHC_d / self.V_d - 273.15
            return np.array([S_t, S_ts, S_n, S_s, S_d, self.D, T_t, T_ts, T_n, T_s, T_d]).T

        return np.array([S_t, S_ts, S_n, S_s, S_d, self.D]).T

    def get_q_N(self, traj, density_diff=False):
        """Re-compute q_N from a set of states.

        If density_diff is True, returns the density difference as well
        """
        S_ts, S_n, D = traj[...,1], traj[...,2], traj[...,5]
        if self.temp:
            T_ts, T_n = traj[...,7], traj[...,8]
        else:
            T_ts, T_n = self.T_ts, self.T_n
        q_N_factor = self.sea_ice(T_n) if self.ice else 1.0

        rho_n  = self.rho_0 * (1.0 - self.alpha * (T_n - self.T_0) + self.beta * (S_n - self.S_0))
        rho_ts = self.rho_0 * (1.0 - self.alpha * (T_ts - self.T_0) + self.beta * (S_ts - self.S_0))
        q_N = q_N_factor * self.eta * (rho_n - rho_ts) * D**2.0 / self.rho_0

        if density_diff:
            return q_N, rho_n - rho_ts
        return q_N

    def update_fluxes(self, state):
        """Update fluxes after each timestep."""
        self.q_N = self.get_q_N(state)

        D = state[...,5]
        self.q_U = self.kappa * self.A / D
        self.q_e = self.A_GM * self.L_xA * D / self.L_y
        self.q_S = self.q_Ek - self.q_e

    def _advance(self,
                 step: int,
                 time: float,
                 dt: float,
                 noise: Any,
                 forcingAmpl: float) -> float:
        """Override the template."""
        self.system(self._state, self._dW(dt, noise), dt)
        self._state = self.update_vars()
        self.update_fluxes(self._state)
        return dt

    # def run(self, N, T, init_state, stop_transi=True, stop_on=False, stop_off=False):
    #     """
    #     N is the number of independent trajectories to compute.
    #     T is the length of the trajectories in years.
    #     init is the initialization state. Defaults to the ON-state.
    #     """

    #     #Time parameters
    #     self.time = np.arange(T * 365 * 86400 / self.delta_t) / (365 * 86400 / self.delta_t)

    #     #Initialize the N trajectories
    #     if init_state.ndim == 1:
    #         if stop_transi:
    #             if np.all(np.isclose(init_state,self.on)):
    #                 stop_on, stop_off = False, True
    #             elif np.all(np.isclose(init_state,self.off)):
    #                 stop_on, stop_off = True, False
    #         init_state = np.repeat(init_state[np.newaxis],N,axis=0)

    #     #Basic initializations
    #     self.init_var_params(init_state)

    #     # Initialize the trajectories, the noise level is computed in advance
    #     dW = self._noise_level * self.E_A * self.S_0 * \
    #         self.rng.normal(scale=np.sqrt(self.delta_t * (100*365*86400)), size=(len(self.time)-1,N))
    #     traj = np.ma.masked_all((N, len(self.time), init_state.shape[-1]))
    #     traj[:,0] = init_state

    #     if stop_transi:
    #         idx_traj = np.arange(N)
    #         self.VS_mem = np.zeros((N,4))
    #         if self.temp:
    #             self.OHC_mem = np.zeros((N,5))

    #         i = 1
    #         while len(idx_traj) > 0 and i < len(self.time):
    #             self.system(traj[idx_traj,i-1], dW[i-1,idx_traj])
    #             self.VS_mem[idx_traj,0], self.VS_mem[idx_traj,1], self.VS_mem[idx_traj,2], \
    #                 self.VS_mem[idx_traj,3] = self.VS_t, self.VS_ts, self.VS_n, self.VS_s
    #             if self.temp:
    #                 self.OHC_mem[idx_traj,0], self.OHC_mem[idx_traj,1], self.OHC_mem[idx_traj,2], \
    #                     self.OHC_mem[idx_traj,3], self.OHC_mem[idx_traj,4] = self.OHC_t, self.OHC_ts, \
    #                     self.OHC_n, self.OHC_s, self.OHC_d

    #             traj[idx_traj,i] = self.update_vars()

    #             if stop_on:
    #                 idx_traj = np.flatnonzero(1 - self.is_on(traj[:,i]))
    #             else:
    #                 idx_traj = np.flatnonzero(1 - self.check_F_collapse(traj[:,i]))

    #             self.VS_t, self.VS_ts, self.VS_n, self.VS_s = self.VS_mem[idx_traj].T
    #             if self.temp:
    #                 self.OHC_t, self.OHC_ts, self.OHC_n, self.OHC_s, self.OHC_d = self.OHC_mem[idx_traj].T

    #             self.update_fluxes(traj[idx_traj,i])
    #             i += 1

    #     else:
    #         for i in trange(1, len(self.time)):
    #             self.system(traj[:,i-1], dW[i-1])
    #             traj[:,i] = self.update_vars()
    #             self.update_fluxes(traj[:,i])

    #     return traj

    # def run_with_forcing(self, N, start_idx, init_state, idx_forcing, Tmax=100, stop_transi=True, \
    #     stop_on=False, stop_off=False):
    #     """
    #     N is the number of independent trajectories to compute.
    #     T is the length of the trajectories in years.
    #     init is the initialization state. Defaults to the ON-state.
    #     """

    #     T = Tmax*365*86400/self.delta_t - start_idx
    #     t_end = np.max(T).astype(int)

    #     f_p = np.copy(self.forcing_params)[idx_forcing]

    #     #Initialize the N trajectories
    #     if init_state.ndim == 1:
    #         if stop_transi:
    #             if np.all(np.isclose(init_state,self.on)):
    #                 stop_on, stop_off = False, True
    #             elif np.all(np.isclose(init_state,self.off)):
    #                 stop_on, stop_off = True, False
    #         init_state = np.repeat(init_state[np.newaxis],N,axis=0)

    #     #Basic initializations
    #     self.init_var_params(init_state)

    #     # Initialize the trajectories, the noise_level is computed in advance
    #     dW = self._noise_level * self.E_A * self.S_0 * \
    #         self.rng.normal(scale=np.sqrt(self.delta_t * (100*365*86400)), size=(t_end-1,N))
    #     traj = np.ma.masked_all((N, t_end, init_state.shape[-1]))
    #     traj[:,0] = init_state

    #     self.VS_0 = np.repeat(self.VS_0, N)

    #     if stop_transi:
    #         idx_traj = np.arange(N)
    #         self.VS_mem = np.zeros((N,6))
    #         if self.temp:
    #             self.OHC_mem = np.zeros((N,5))

    #         i = 1
    #         while len(idx_traj) > 0 and i < t_end:
    #             current_time = (start_idx[idx_traj]+i-1) * self.delta_t / (365*86400)
    #             self.system(traj[idx_traj,i-1], dW[i-1,idx_traj],
    #                         forcing = self.forcing(traj[idx_traj,i-1], current_time, f_p[idx_traj]))
    #             self.VS_mem[idx_traj,0], self.VS_mem[idx_traj,1], self.VS_mem[idx_traj,2], \
    #                 self.VS_mem[idx_traj,3], self.VS_mem[idx_traj,4] = self.VS_t, self.VS_ts, \
    #                 self.VS_n, self.VS_s, self.VS_d
    #             self.VS_mem[idx_traj,4], self.VS_mem[idx_traj,5] = self.VS_d, self.VS_0
    #             if self.temp:
    #                 self.OHC_mem[idx_traj,0], self.OHC_mem[idx_traj,1], self.OHC_mem[idx_traj,2], \
    #                     self.OHC_mem[idx_traj,3], self.OHC_mem[idx_traj,4] = self.OHC_t, self.OHC_ts, \
    #                     self.OHC_n, self.OHC_s, self.OHC_d

    #             traj[idx_traj,i] = self.update_vars()

    #             if stop_on:
    #                 idx_traj = np.flatnonzero(1 - self.is_on(traj[:,i]))
    #             else:
    #                 idx_traj = np.flatnonzero(1 - self.check_F_collapse(traj[:,i]))

    #             self.VS_t, self.VS_ts, self.VS_n, self.VS_s, self.VS_d, self.VS_0 = self.VS_mem[idx_traj].T
    #             if self.temp:
    #                 self.OHC_t, self.OHC_ts, self.OHC_n, self.OHC_s, self.OHC_d = self.OHC_mem[idx_traj].T

    #             self.update_fluxes(traj[idx_traj,i])
    #             i += 1

    #     else:
    #         for i in trange(1, len(self.time)):
    #             current_time = (start_idx[idx_traj]+i-1) * self.delta_t / (365*86400)
    #             self.system(traj[:,i-1], dW[i-1], forcing=self.forcing(traj[:,i-1], current_time, f_p))
    #             traj[:,i] = self.update_vars()
    #             self.update_fluxes(traj[:,i])

    #     return traj

    # def get_var(self, var, traj):
    #     """
    #     Get the variable var from the trajectories traj.
    #     """
    #     idx = self.convention.index(var)
    #     return traj[...,idx]

    # def is_on(self, traj):
    #     """
    #     Checks every on-state in any set of trajectories.
    #     traj must contain all variables of the model
    #     """
    #     q_N = self.get_q_N(traj)
    #     return q_N >= self.q_N_on

    # def check_F_collapse(self, traj):
    #     """
    #     Checks F-collapses in any set of trajectories.
    #     If there is ice, we check only if the AMOC is weaker than the weak state threshold
    #     """
    #     q_N, rho_diff = self.get_q_N(traj, density_diff=True)
    #     if not self.ice:
    #         return rho_diff <= 0
    #     return q_N <= 2e6

    # def check_S_collapse(self, traj):
    #     """
    #     Checks S-collapses in any set of trajectories.
    #     """
    #     q_N = self.get_q_N(traj)
    #     return q_N <= self.q_N_off_full

    def score(self, z=0, on=True):
        """Compute the score of a set of trajectories, only based on q_N."""
        q_N = self.get_q_N(self._state)
        if on:
            return (self.q_N_on - q_N) / (self.q_N_on - z)
        return (self.q_N_off_full - q_N) / (self.q_N_off_full - self.q_N_on)

    def _make_noise(self):
        """Return a random noise."""
        return np.random.normal()

if __name__ == "__main__":
    fmodel = MOCModel
    tams = TAMS(fmodel_t=fmodel)
    transition_proba = tams.compute_probability()
    print("Transition probability: {}".format(transition_proba))

