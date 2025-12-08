"""Evaluate the stationary distribution of the OU process."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import toml
from ornsteinuhlenbeck import OrnsteinUhlenbeck
from pytams.trajectory import Trajectory

if __name__ == "__main__":
    fmodel = OrnsteinUhlenbeck
    with Path("./input_stationary.toml").open("r") as f:
        input_params = toml.load(f)

    traj = Trajectory(0, 1.0, fmodel, input_params)
    traj.advance()

    # Stationary distribution has standard deviation:
    theta = input_params["model"]["theta"]
    epsilon = input_params["model"]["epsilon"]
    sigma = np.sqrt(epsilon / theta)

    # Compute stationary P_s
    nbin = 200
    xmin = -5 * sigma
    xmax = 5 * sigma
    x = np.linspace(-5 * sigma, 5 * sigma, nbin)
    Ps = np.sqrt(1.0 / (2.0 * np.pi * sigma**2)) * np.exp(-0.5 * x * x / (sigma**2))

    # Binning pyTAMS trajectory
    dx = (xmax - xmin) / nbin
    cbins = np.linspace(-5 * sigma - dx / 2, 5 * sigma + dx / 2, nbin + 1)
    counts, bins = np.histogram(traj.get_score_array(), bins=cbins, density=True)

    plt.figure(figsize=(6, 4))
    plt.plot(x / sigma, Ps, linewidth=0.8, color="k", label="Theory")
    plt.plot(x / sigma, counts, linewidth=0.8, color="r", label="pyTAMS")
    plt.grid(linestyle="dotted")
    plt.legend(fontsize="x-large")
    plt.xlabel(r"$x/\sigma$", fontsize="large")
    plt.ylabel(r"$P_s(x)$", fontsize="large")
    plt.tight_layout()
    plt.savefig("OU_StationaryDistribution.png", dpi=1200)
