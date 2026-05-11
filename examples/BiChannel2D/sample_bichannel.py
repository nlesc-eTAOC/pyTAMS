import numpy as np
from bichannel2d import BiChannel2D
from bichannel2d import plot_in_landscape
from pytams.sampler import build_sampler

if __name__ == "__main__":
    # For convenience
    fmodel = BiChannel2D

    # Enable TAMS trajectory plots
    plot_ensemble = False

    # Number of consecutive TAMS runs
    K = 1

    probabilities = np.zeros(K)

    # Run the model several times
    for i in range(K):
        # Initialize the sampler
        sampler = build_sampler(fmodel_t=fmodel)

        # Sample and report
        try:
            sampler.run()
            probability = sampler.database.get_event_probability()
        except RuntimeError as e:
            print(e)  # noqa: T201
            continue

        probabilities[i] = probability

        print(f"[{i}] : {probability}")  # noqa: T201

        if plot_ensemble:
            plot_in_landscape(fmodel, sampler.database(), i)

    print(f"Averaged transition P_K: {probabilities.mean()}, RE: {np.sqrt(probabilities.var()) / probabilities.mean()}")  # noqa : T201
