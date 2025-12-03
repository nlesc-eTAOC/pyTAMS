import contextlib
import matplotlib.pyplot as plt
import numpy as np
from bichannel2d import BiChannel2D
from pytams.tams import TAMS


def plot_in_landscape(fmodel, tdb, idx) -> None: # noqa: ANN001
    """Wrapper function to plot TAMS ensemble."""
    # Get a potential map
    x_range = np.linspace(-1.6, 1.6, 101)
    y_range = np.linspace(-1.6, 2.6, 132)
    x, y = np.meshgrid(x_range, y_range)
    potential = np.zeros((101, 132))
    for i in range(len(x_range)):
        for j in range(len(y_range)):
            potential[i, j] = fmodel.potential(np.array([x_range[i], y_range[j]]))

    with contextlib.suppress(Exception):
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})

    plt.figure(figsize=(6, 4))
    ctr = plt.contourf(x, y, np.transpose(potential), levels=25, cmap="viridis_r")

    for t in tdb.traj_list():
        state_x = np.array([s[1][0] for s in t.get_state_list()])
        state_y = np.array([s[1][1] for s in t.get_state_list()])
        plt.plot(state_x, state_y, color="k", alpha=0.6)

    plt.xticks(fontsize="x-large")
    plt.yticks(fontsize="x-large")
    plt.xlabel("x", fontsize="x-large")
    plt.ylabel("y", fontsize="x-large")
    plt.colorbar(ctr, label="V(x,y)")
    plt.grid(linestyle=":", color="silver")
    plt.tight_layout()
    plt.savefig(f"./Path_run_{idx:04d}.png", dpi=1200)
    plt.close()


if __name__ == "__main__":
    # For convenience
    fmodel = BiChannel2D

    # Enable TAMS trajectory plots
    plot_ensemble = True

    # Number of consecutive TAMS runs
    K = 10

    probabilities = np.zeros(K)

    # Run the model several times
    for i in range(K):
        # Initialize the algorithm object
        tams = TAMS(fmodel_t=fmodel)

        # Run TAMS and report
        try:
            probability = tams.compute_probability()
        except RuntimeError as e:
            print(e) # noqa: T201
            continue

        probabilities[i] = probability

        print(f"[{i}] : {probability}") # noqa: T201

        if plot_ensemble:
            plot_in_landscape(fmodel, tams.get_database(), i)

    print(f"Averaged transition P_K: {probabilities.mean()}, RE: {np.sqrt(probabilities.var()) / probabilities.mean()}")  # noqa : T201

