# %% Stable Equilibria of the Stochastic Model

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy as sp


class Boussinesq:
    """Class for the 2D Boussinesq model of the AMOC.

    The model includes stochastic forcing on freshwater.
    The model is described in J. Soons et al. doi:10.1017/jfm.2025.248
    and encapsulated/optimzed by V. Jacques-Dumas.

    Attributes:
        m, n: number of grid points in x and z
        dt: time step
        Pr, Le, A, Ra: physical parameters
        delta, tauT, tauS: numerical parameters
    """

    def __init__(
        self,
        m: int,
        n: int,
        dt: float,
        pr: float = 1.0,
        le: float = 1.0,
        a: float = 5.0,
        ra: float = 4.0e4,
        delta: float = 0.05,
        tau_t: float = 0.1,
        tau_s: float = 1.0,
    ):
        # Pass in model paramaters
        self._pr = pr
        self._le = le
        self._a = a
        self._ra = ra
        self._delta = delta
        self._tau_t = tau_t
        self._tau_s = tau_s

        # Set up grid
        self._m, self._n = m, n
        self.xx = np.linspace(0, self._a, self._m + 1)
        self.zz = self.z_j(np.arange(self._n + 1))
        self.dx = self.xx[1] - self.xx[0]

        # Time stepping
        self._dt = dt
        self.dtsq = np.sqrt(self._dt)

        # Hosing parameters
        self.hosing_t0 = 0.0  # Start of rate increase
        self.hosing_sval = 0.0  # Initial hosing value at t0
        self.hosing_rate = 0.0  # Rate of change of hosing

        self.make_x_derivatives()
        self.make_z_derivatives()
        self.boundary_layer_operator()
        self.correction_operator()
        self.integration_matrices()

    def h(self, z: npt.NDArray[np.number], delta_loc: float | None = None) -> npt.NDArray[np.number]:
        """The surface noise degeneration function.

        Args:
            z: the depth coordinate
            delta_loc: a delta value, locally different from the class one
        """
        if delta_loc:
            return np.exp(-(1 - z) / delta_loc)

        return np.exp(-(1 - z) / self._delta)

    def temp_s(self, x: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Forcing on surface temperature.

        Args:
            x: latitude

        Returns:
            The static surface temperature forcing value
        """
        return 0.5 * (np.cos(2 * np.pi * (x / self._a - 1 / 2)) + 1)

    def salt_s(self, x: npt.NDArray[np.number], beta: float) -> npt.NDArray[np.number]:
        """Forcing on surface salinity.

        Args:
            x: latitude
            beta: asymmetry parameter

        Returns:
            The static surface salinity forcing value
        """
        return 3.5 * np.cos(2.0 * np.pi * (x / self._a - 1.0 / 2.0)) - beta * np.sin(np.pi * (x / self._a - 1 / 2))

    def hosing(self, x: npt.NDArray[np.number], ampl: float) -> npt.NDArray[np.number]:
        """Define hosing spatial profile.

        We use a tanh function in latitute, adding freshwater to
        the northern part if the amplitude is positive (while
        removing freshwater in the southern part to conserve salinity).

        Args:
            x: latitude
            ampl: hosing amplitude factor

        Returns:
            The local hosing value
        """
        return -ampl * np.tanh(20.0 * (x / self._a - 1 / 2))

    def init_hosing(self, t0: float, h0: float, rate: float) -> None:
        """Initialize hosing parameters.

        Args:
            t0 : hosing start time
            h0 : initial hosing value amplitude (at t0)
            rate : rate of change of hosing amplitude
        """
        self.hosing_t0 = t0
        self.hosing_sval = h0
        self.hosing_rate = rate

    def init_salt_stoch_noise(self, z: npt.NDArray[np.number], nk: int, eps: float, delta_stoch: float) -> None:
        """Initialize stochastic salinity noise.

        Args:
            z: depth coordinate
            nk: number of Fourier modes
            eps: stochastic noise amplitude factor
            delta_stoch: the depth thickness parameter
        """
        cos_term = np.cos(2 * np.pi * np.arange(1, nk + 1)[:, np.newaxis] * self.xx[np.newaxis] / self._a)[
            ..., np.newaxis
        ]
        sin_term = np.sin(2 * np.pi * np.arange(1, nk + 1)[:, np.newaxis] * self.xx[np.newaxis] / self._a)[
            ..., np.newaxis
        ]
        z_term = np.sqrt(eps / nk) * np.expand_dims(self.h(z, delta_stoch), axis=(0, 1))
        self.cos_term, self.sin_term = cos_term * z_term, sin_term * z_term

    def salt_stoch_noise(self, normalrange: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Assemble the salinity stochastic forcing.

        Args:
            normalrange : A numpy array of size 2*nk, containing random numbers for each mode

        Returns:
            Numpy of the salinity forcing over the entire domain
        """
        normalrange = np.expand_dims(normalrange, axis=(1, 2))
        return np.sum(normalrange[::2] * self.cos_term + normalrange[1::2] * self.sin_term, axis=0)

    def z_j(self, j: npt.NDArray[np.number]) -> npt.NDArray[np.number]:
        """Return the depth for index range."""
        q = 3
        return 0.5 + np.tanh(q * (j / self._n - 0.5)) / (2 * np.tanh(q / 2))

    def dz_j(self, j: int) -> float:
        """Return the depth width of cells."""
        z_j = self.z_j(np.array([j, j + 1], dtype="int"))
        return z_j[1] - z_j[0]

    def make_x_derivatives(self) -> None:
        """Construct latitude derivatives."""
        self.Dxx = np.diag(-2 * np.ones(self._m + 1)) + np.diag(np.ones(self._m), 1) + np.diag(np.ones(self._m), -1)
        self.Fxx = np.diag(-2 * np.ones(self._m + 1)) + np.diag(np.ones(self._m), 1) + np.diag(np.ones(self._m), -1)
        self.Dxx[0, 1], self.Dxx[-1, -2] = 2, 2
        self.Fxx[0, 0], self.Fxx[0, 1], self.Fxx[-1, -1], self.Fxx[-1, -2] = 0, 0, 0, 0
        self.Dxx /= self.dx**2
        self.Fxx /= self.dx**2

        self.Dx = np.diag(np.ones(self._m), 1) + np.diag(-np.ones(self._m), -1)
        self.Fx = np.diag(np.ones(self._m), 1) + np.diag(-np.ones(self._m), -1)
        self.Dx[0, 1], self.Dx[-1, -2] = 0, 0
        self.Fx[0, 1], self.Fx[-1, -2] = 2, -2
        self.Dx /= 2 * self.dx
        self.Fx /= 2 * self.dx

    def make_z_derivatives(self) -> None:
        """Constrcut and define depth derivative functions."""

        def p_j(j: npt.NDArray[np.number] | int) -> npt.NDArray[np.number]:
            return 1 / (self.z_j(j + 1) - self.z_j(j - 1))

        def alpha_j(j: npt.NDArray[np.number] | int) -> float:
            return 2 / (self.dz_j(j - 1) * (self.z_j(j + 1) - self.z_j(j - 1)))

        def beta_j(j: npt.NDArray[np.number] | int) -> float:
            return 2 / (self.dz_j(j) * self.dz_j(j - 1))

        def gamma_j(j: npt.NDArray[np.number] | int) -> float:
            return 2 / (self.dz_j(j) * (self.z_j(j + 1) - self.z_j(j - 1)))

        self.Dzz = (
            np.diag(-beta_j(np.arange(self._n + 1)), k=0)
            + np.diag(gamma_j(np.arange(self._n)), k=1)
            + np.diag(alpha_j(np.arange(1, self._n + 1)), k=-1)
        )
        self.Fzz = (
            np.diag(-beta_j(np.arange(self._n + 1)), k=0)
            + np.diag(gamma_j(np.arange(self._n)), k=1)
            + np.diag(alpha_j(np.arange(1, self._n + 1)), k=-1)
        )
        self.Dzz[0, 1], self.Dzz[-1, -2] = alpha_j(0) + gamma_j(0), alpha_j(self._n) + gamma_j(self._n)
        self.Fzz[0, 0], self.Fzz[0, 1], self.Fzz[-1, -1], self.Fzz[-1, -2] = 0, 0, 0, 0

        self.Dz = np.diag(p_j(np.arange(self._n)), k=1) + np.diag(-p_j(np.arange(1, self._n + 1)), k=-1)
        self.Fz = np.diag(p_j(np.arange(self._n)), k=1) + np.diag(-p_j(np.arange(1, self._n + 1)), k=-1)
        self.Dz[0, 1], self.Dz[-1, -2] = 0, 0
        self.Fz[0, 1], self.Fz[-1, -2] = 1 / self.dz_j(-1), -1 / self.dz_j(self._n - 1)

        self.DzT, self.FzT = self.Dz.T, self.Fz.T
        self.DzzT, self.FzzT = self.Dzz.T, self.Fzz.T

    def boundary_layer_operator(self) -> None:
        """Define (surface) boundary layer operator."""
        self.Hz = np.diag(self.h(self.z_j(np.arange(self._n + 1))))
        self.HzT = self.Hz.T

    def correction_operator(self) -> None:
        """Define the salinity correction operator."""
        self.S_corr = np.identity(self._n + 1)
        self.S_corr[0, 0], self.S_corr[-1, -1] = 0, 0

    def make_salinity_forcing(self, beta: float) -> None:
        """Make the salinity forcing term.

        Args:
            beta : the asymetry parameter
        """
        self.FS = (
            self.h(self.z_j(np.arange(self._n + 1)))[np.newaxis]
            * self.salt_s(self.xx, beta)[:, np.newaxis]
            / self._tau_s
        )

    def get_hosing(self, time: float) -> npt.NDArray[np.number]:
        """Get the hosing surface term.

        Args:
            time : model time

        Returns:
            A numpy array with the hosing degenerated
        """
        ampl = self.hosing_sval + max(0, (time - self.hosing_t0)) * self.hosing_rate
        return (
            self.h(self.z_j(np.arange(self._n + 1)))[np.newaxis]
            * self.hosing(self.xx, ampl)[:, np.newaxis]
            / self._tau_s
        )

    def integration_matrices(self) -> None:
        """Build the integration matrices."""
        self.AT = np.identity(self._m + 1) - self._dt * self.Dxx
        self.BT = self._dt / self._tau_t * self.HzT - self._dt * self.DzzT
        self.AS = np.identity(self._m + 1) - self._dt / self._le * self.Dxx
        self.BS = -self._dt / self._le * self.DzzT
        self.Aw = np.identity(self._m + 1) - self._dt * self._pr * self.Fxx
        self.Bw = -self._dt * self._pr * self.FzzT

        self.FT = (
            self.h(self.z_j(np.arange(self._n + 1)))[np.newaxis] * self.temp_s(self.xx)[:, np.newaxis] / self._tau_t
        )

    def trajectory(
        self, nt_max: int, phi_start: npt.NDArray[np.number], beta: float, nk: int, eps: float
    ) -> npt.NDArray[np.number]:
        """Run the model for a number of steps, forming a trajectory.

        Args:
            nt_max: number of time steps
            phi_start: initial solution
            beta: asymmetry parameter
            nk: number of forcing mode
            eps: stochastic noise amplitude factor

        Returns:
            The model state along the trajectory as a 4D numpy array
        """
        s0 = 1
        self.make_salinity_forcing(beta)

        # Initilize the state vector
        phi_traj = np.zeros((nt_max + 1, 4, self._m + 1, self._n + 1))
        phi_traj[0] = phi_start

        # Initialize random noise for all steps
        rng = np.random.default_rng()
        normalrange = rng.normal(0, 1, size=(nt_max, 2 * nk))

        self.init_salt_stoch_noise(self.zz, nk, eps, 0.05)

        # Time stepping loop
        for t in range(1, nt_max + 1):
            # Unpack state data
            w_old, s_old, t_old, psi_old = phi_traj[t - 1]

            fx_psi = self.Fx @ psi_old
            psi_fz = psi_old @ self.FzT
            dx_st = self.Dx @ phi_traj[t - 1, 1:3]
            st_dz = phi_traj[t - 1, 1:3] @ self.DzT

            # temperature, salinity updates
            q_s, q_t = fx_psi[np.newaxis] * st_dz - psi_fz[np.newaxis] * dx_st

            c_t = t_old + self._dt * (q_t + self.FT)
            c_s = s_old + self._dt * (q_s + self.FS) + self.dtsq * self.salt_stoch_noise(normalrange[t - 1])

            t_new = sp.linalg.solve_sylvester(self.AT, self.BT, c_t)
            s_new = sp.linalg.solve_sylvester(self.AS, self.BS, c_s)
            s_new *= s0 / np.mean(s_new)  # correction to conserve salinity

            # vorticity update
            f_w = self._pr * self._ra * self.Dx @ (t_new - s_new) @ self.S_corr
            q_w = fx_psi * (w_old @ self.FzT) - psi_fz * (self.Fx @ w_old)
            c_w = w_old + self._dt * (q_w + f_w)
            w_new = sp.linalg.solve_sylvester(self.Aw, self.Bw, c_w)

            # streamfunction update with the poisson operator
            psi_new = sp.linalg.solve_sylvester(self.Fxx, self.FzzT, -w_new)

            phi_traj[t] = np.array([w_new, s_new, t_new, psi_new])

        return phi_traj

    def contourplot(self, xzmatrix: npt.NDArray[np.number], name: str | None = None) -> None:
        """Plot the 2D contour of a variable.

        Args:
            xzmatrix: the variable of interest on the domain grid
            name: an optional title name for the plot
        """
        x, z = np.meshgrid(self.xx, self.zz)
        plt.contourf(x, z, np.transpose(xzmatrix))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        if name:
            plt.title(name)
        plt.show()

    def stateplot(self, state: npt.NDArray[np.number], addition: str | None = None) -> None:
        """Plot all 4 components of the state vector.

        Args:
            state: the state numpy array
            addition: an optional string for plot title
        """
        for i in range(4):
            self.contourplot(state[i], ["T", "S", r"$\psi$", r"$\omega$"][i] + str(addition))
