# %% Stable Equilibria of the Stochastic Model

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp


class Boussinesq:
    """Class for the 2D Boussinesq model of the AMOC with
    stochastic forcing.

    The model is described in J. Soons et al. doi:10.1017/jfm.2025.248
    and encapsulated/optimzed by V. Jacques-Dumas.

    Attributes:
        M, N: number of grid points in x and z
        dt: time step
        Pr, Le, A, Ra: physical parameters
        delta, tauT, tauS: numerical parameters
    """

    def __init__(self, M, N, dt, Pr=1, Le=1, A=5, Ra=4e4, delta=0.05, tauT=0.1, tauS=1, seed=None):
        self.Pr, self.Le, self.A, self.Ra = Pr, Le, A, Ra
        self.delta, self.tauT, self.tauS = delta, tauT, tauS

        self.M, self.N = M, N
        self.xx = np.linspace(0, self.A, self.M + 1)
        self.dx = self.xx[1] - self.xx[0]

        self.dt = dt
        self.dtsq = np.sqrt(self.dt)

        self.zz = self.z_j(np.arange(self.N + 1))

        self.make_x_derivatives()
        self.make_z_derivatives()
        self.boundary_layer_operator()
        self.correction_operator()
        self.integration_matrices()

    def h(self, z):
        return np.exp(-(1 - z) / self.delta)

    def Ts(self, x):
        return 0.5 * (np.cos(2 * np.pi * (x / self.A - 1 / 2)) + 1)

    def Ss(self, x, beta):
        return 3.5 * np.cos(2 * np.pi * (x / self.A - 1 / 2)) - beta * np.sin(np.pi * (x / self.A - 1 / 2))

    def init_Snoise(self, z, K, eps):
        cos_term = np.cos(2 * np.pi * np.arange(1, K + 1)[:, np.newaxis] * self.xx[np.newaxis] / self.A)[
            ..., np.newaxis
        ]
        sin_term = np.sin(2 * np.pi * np.arange(1, K + 1)[:, np.newaxis] * self.xx[np.newaxis] / self.A)[
            ..., np.newaxis
        ]
        z_term = np.sqrt(eps / K) * np.expand_dims(self.h(z), axis=(0, 1))
        self.cos_term, self.sin_term = cos_term * z_term, sin_term * z_term

    def Snoise(self, normalrange):
        normalrange = np.expand_dims(normalrange, axis=(1, 2))
        return np.sum(normalrange[::2] * self.cos_term + normalrange[1::2] * self.sin_term, axis=0)

    def z_j(self, j):
        q = 3
        return 0.5 + np.tanh(q * (j / self.N - 0.5)) / (2 * np.tanh(q / 2))

    def dz_j(self, j):
        return self.z_j(j + 1) - self.z_j(j)

    def make_x_derivatives(self):
        self.Dxx = np.diag(-2 * np.ones(self.M + 1)) + np.diag(np.ones(self.M), 1) + np.diag(np.ones(self.M), -1)
        self.Fxx = np.diag(-2 * np.ones(self.M + 1)) + np.diag(np.ones(self.M), 1) + np.diag(np.ones(self.M), -1)
        self.Dxx[0, 1], self.Dxx[-1, -2] = 2, 2
        self.Fxx[0, 0], self.Fxx[0, 1], self.Fxx[-1, -1], self.Fxx[-1, -2] = 0, 0, 0, 0
        self.Dxx /= self.dx**2
        self.Fxx /= self.dx**2

        self.Dx = np.diag(np.ones(self.M), 1) + np.diag(-np.ones(self.M), -1)
        self.Fx = np.diag(np.ones(self.M), 1) + np.diag(-np.ones(self.M), -1)
        self.Dx[0, 1], self.Dx[-1, -2] = 0, 0
        self.Fx[0, 1], self.Fx[-1, -2] = 2, -2
        self.Dx /= 2 * self.dx
        self.Fx /= 2 * self.dx

    def make_z_derivatives(self):
        def p_j(j):
            return 1 / (self.z_j(j + 1) - self.z_j(j - 1))

        def alpha_j(j):
            return 2 / (self.dz_j(j - 1) * (self.z_j(j + 1) - self.z_j(j - 1)))

        def beta_j(j):
            return 2 / (self.dz_j(j) * self.dz_j(j - 1))

        def gamma_j(j):
            return 2 / (self.dz_j(j) * (self.z_j(j + 1) - self.z_j(j - 1)))

        self.Dzz = (
            np.diag(-beta_j(np.arange(self.N + 1)))
            + np.diag(gamma_j(np.arange(self.N)), 1)
            + np.diag(alpha_j(np.arange(1, self.N + 1)), -1)
        )
        self.Fzz = (
            np.diag(-beta_j(np.arange(self.N + 1)))
            + np.diag(gamma_j(np.arange(self.N)), 1)
            + np.diag(alpha_j(np.arange(1, self.N + 1)), -1)
        )
        self.Dzz[0, 1], self.Dzz[-1, -2] = alpha_j(0) + gamma_j(0), alpha_j(self.N) + gamma_j(self.N)
        self.Fzz[0, 0], self.Fzz[0, 1], self.Fzz[-1, -1], self.Fzz[-1, -2] = 0, 0, 0, 0

        self.Dz = np.diag(p_j(np.arange(self.N)), 1) + np.diag(-p_j(np.arange(1, self.N + 1)), -1)
        self.Fz = np.diag(p_j(np.arange(self.N)), 1) + np.diag(-p_j(np.arange(1, self.N + 1)), -1)
        self.Dz[0, 1], self.Dz[-1, -2] = 0, 0
        self.Fz[0, 1], self.Fz[-1, -2] = 1 / self.dz_j(-1), -1 / self.dz_j(self.N - 1)

        self.DzT, self.FzT = self.Dz.T, self.Fz.T
        self.DzzT, self.FzzT = self.Dzz.T, self.Fzz.T

    def boundary_layer_operator(self):
        self.Hz = np.diag(self.h(self.z_j(np.arange(self.N + 1))))
        self.HzT = self.Hz.T

    def correction_operator(self):
        self.S_corr = np.identity(self.N + 1)
        self.S_corr[0, 0], self.S_corr[-1, -1] = 0, 0

    def make_FS(self, beta):
        self.FS = (
            self.h(self.z_j(np.arange(self.N + 1)))[np.newaxis] * self.Ss(self.xx, beta)[:, np.newaxis] / self.tauS
        )

    def integration_matrices(self):
        self.AT = np.identity(self.M + 1) - self.dt * self.Dxx
        self.BT = self.dt / self.tauT * self.HzT - self.dt * self.DzzT
        self.AS = np.identity(self.M + 1) - self.dt / self.Le * self.Dxx
        self.BS = -self.dt / self.Le * self.DzzT
        self.Aw = np.identity(self.M + 1) - self.dt * self.Pr * self.Fxx
        self.Bw = -self.dt * self.Pr * self.FzzT

        self.FT = self.h(self.z_j(np.arange(self.N + 1)))[np.newaxis] * self.Ts(self.xx)[:, np.newaxis] / self.tauT

    def parallel_T(self, CT):
        return sp.linalg.solve_sylvester(self.AT, self.BT, CT)

    def parallel_S(self, CS):
        return sp.linalg.solve_sylvester(self.AS, self.BS, CS)

    def parallel_w(self, Cw):
        return sp.linalg.solve_sylvester(self.Aw, self.Bw, Cw)

    def parallel_psi(self, w):
        return sp.linalg.solve_sylvester(self.Fxx, self.FzzT, -w)

    def trajectory(self, Nt_max, phi_start, beta, K, eps):
        S0 = 1
        self.make_FS(beta)

        phi_traj = np.zeros((Nt_max + 1, 4, self.M + 1, self.N + 1))
        phi_traj[0] = phi_start

        rng = np.random.default_rng(seed=0)
        normalrange = rng.normal(0, 1, size=(Nt_max, 2 * K))
        zjn = self.z_j(np.arange(self.N + 1))
        self.init_Snoise(zjn, K, eps)

        for t in range(1, Nt_max + 1):
            w_old, S_old, T_old, psi_old = phi_traj[t - 1]

            Fxpsi = self.Fx @ psi_old
            psiFz = psi_old @ self.FzT
            DxST = self.Dx @ phi_traj[t - 1, 1:3]
            STDz = phi_traj[t - 1, 1:3] @ self.DzT

            # temperature, salinity updates
            QS, QT = Fxpsi[np.newaxis] * STDz - psiFz[np.newaxis] * DxST

            CT = T_old + self.dt * (QT + self.FT)
            CS = S_old + self.dt * (QS + self.FS) + self.dtsq * self.Snoise(normalrange[t - 1])

            T_new = sp.linalg.solve_sylvester(self.AT, self.BT, CT)
            S_new = sp.linalg.solve_sylvester(self.AS, self.BS, CS)
            S_new *= S0 / np.mean(S_new)  # correction to conserve salinity

            # vorticity update
            Fw = self.Pr * self.Ra * self.Dx @ (T_new - S_new) @ self.S_corr
            Qw = Fxpsi * (w_old @ self.FzT) - psiFz * (self.Fx @ w_old)
            Cw = w_old + self.dt * (Qw + Fw)
            w_new = sp.linalg.solve_sylvester(self.Aw, self.Bw, Cw)

            # streamfunction update with the poisson operator
            psi_new = sp.linalg.solve_sylvester(self.Fxx, self.FzzT, -w_new)

            phi_traj[t] = np.array([w_new, S_new, T_new, psi_new])

        return phi_traj

    def worker_trajectory(self, args):
        return self.trajectory(*args)

    def contourplot(self, xzmatrix, name=None):
        X, Z = np.meshgrid(self.xx, self.zz)
        plt.contourf(X, Z, np.transpose(xzmatrix))
        plt.colorbar()
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title(name)
        plt.show()

    def stateplot(self, state, addition=" "):
        for i in range(4):
            self.contourplot(state[i], ["T", "S", r"$\psi$", r"$\omega$"][i] + str(addition))
