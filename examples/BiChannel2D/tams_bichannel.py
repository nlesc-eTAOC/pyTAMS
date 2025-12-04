import numpy as np
from bichannel2d import BiChannel2D
from bichannel2d import plot_in_landscape
from pytams.tams import TAMS

if __name__ == "__main__":
    # For convenience
    fmodel = BiChannel2D

    # Enable TAMS trajectory plots
    plot_ensemble = False

    # Number of consecutive TAMS runs
    K = 20

    probabilities = np.zeros(K)

    # Run the model several times
    for i in range(K):
        # Initialize the algorithm object
        tams = TAMS(fmodel_t=fmodel)

        # Run TAMS and report
        try:
            probability = tams.compute_probability()
        except RuntimeError as e:
            print(e)  # noqa: T201
            continue

        probabilities[i] = probability

        print(f"[{i}] : {probability}")  # noqa: T201

        if plot_ensemble:
            plot_in_landscape(fmodel, tams.get_database(), i)

    print(f"Averaged transition P_K: {probabilities.mean()}, RE: {np.sqrt(probabilities.var()) / probabilities.mean()}")  # noqa : T201
