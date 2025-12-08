import numpy as np
from ornsteinuhlenbeck import OrnsteinUhlenbeck
from pytams.tams import TAMS

if __name__ == "__main__":
    # For convenience
    fmodel = OrnsteinUhlenbeck

    # Run the model K times
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
            print(e)  # noqa: T201
            continue

        probabilities[i] = probability
        print(f"[{i}] P_k = {probability}")  # noqa: T201

    print(f"Averaged transition P_K: {probabilities.mean()}, RE: {np.sqrt(probabilities.var()) / probabilities.mean()}")  # noqa : T201
