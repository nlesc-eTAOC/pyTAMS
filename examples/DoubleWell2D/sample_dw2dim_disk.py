import numpy as np
from DoubleWell2D_disk import Doublewell2DDisk
from pytams.sampler import build_sampler

if __name__ == "__main__":
    # For convenience
    fmodel = Doublewell2DDisk

    # Run the model K times
    K = 1

    probabilities = np.zeros(K)

    # Run the model several times
    for i in range(K):
        # Initialize the sampler
        sampler = build_sampler(fmodel_t=fmodel)

        # Run sampling and report
        try:
            sampler.run()
            probability = sampler.database.get_event_probability()
        except RuntimeError as e:
            print(e)  # noqa: T201
            continue

        probabilities[i] = probability

    print(f"Averaged transition P_K: {probabilities.mean()}, RE: {np.sqrt(probabilities.var()) / probabilities.mean()}")  # noqa : T201
