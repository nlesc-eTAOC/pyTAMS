"""Compute the theoretical return time from the OrnsteinUlhenbeck process."""

import toml
from pathlib import Path
import numpy as np
import numpy.typing as npt
import scipy.integrate
from scipy.interpolate import interp1d
from scipy.special import erfi, erfcx
import matplotlib.pyplot as plt


def stat_mean_first_passage_time_theory(a: float, x0: float, theta: float, epsilon: float) -> float:
    """Compute the first passage time of the OU process.

    Starting at x0 and targeting x(t) > a. This formula is
    described in Lestang et. al. 2018, A6

    Args:
        a: target value of the process
        x0: initial value of the process
        theta: OU process inverse time
        epsilon: OU noise amplitude

    Return:
        the return time of x(t) > a
    """
    k = np.sqrt(theta / (2.0 * epsilon))

    def exppot_in(z: npt.NDArray[np.number], sign: float = -1.0) -> npt.NDArray[np.number]:
        return np.exp(sign * k * k * z * z)

    def exppot_out(
        y: npt.NDArray[np.number], sign: float = -1.0, fun: callable = lambda _: 1.0
    ) -> npt.NDArray[np.number]:
        return np.exp(sign * (k * y) * (k * y)) * fun(y) * fun(y)

    # The return time expression includes nested integrals
    # numerically integrated here
    # Inner integral
    z = np.linspace(-10, 10, 200)
    iarr = scipy.integrate.cumulative_trapezoid(exppot_in(z), z, initial=0)
    ifun = interp1d(z, iarr)

    # Outer integral
    y = np.linspace(-10, a, 200)
    oarr = scipy.integrate.cumulative_trapezoid(exppot_out(y, sign=1, fun=ifun), y, initial=0)
    ofun = interp1d(y, oarr)
    return ofun(a)


if __name__ == "__main__":
    with Path("./input.toml").open("r") as f:
        params = toml.load(f)["model"]

    # OU parameters
    theta = params.get("theta",1.0)
    epsilon = params.get("epsilon",0.5)
    sigma = np.sqrt(epsilon / theta)

    # Integration parameters
    npts = 100
    x0 = 0.0
    a_range = np.linspace(2.0 * sigma, 8.0 * sigma, npts)
    ra = np.zeros(npts)
    scaling = np.sqrt(theta / (2 * np.pi * epsilon**3))
    for i in range(len(a_range)):
        ra[i] = scaling * stat_mean_first_passage_time_theory(a_range[i], x0, theta, epsilon)

    plt.figure(figsize=(6, 4))
    plt.plot(ra, a_range / sigma, linestyle="--", linewidth=0.8, color="k", label="Lestang2018 - A6")
    plt.gcf().axes[0].set_xscale("log")
    plt.legend(fontsize="x-large")
    plt.xlim(left=1.0, right=1e15)
    plt.ylim(bottom=2.0, top=8)
    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.xlabel(r"$\hat{r}(a)$", fontsize="x-large")
    plt.ylabel(r"$a/\sigma$", fontsize="x-large")
    plt.grid(linestyle="dotted", color="silver")
    plt.tight_layout()
    plt.show()
