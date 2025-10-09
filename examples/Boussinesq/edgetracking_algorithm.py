from pathlib import Path
from pytams.trajectory import Trajectory
import numpy as np
import shutil

def dist(state1, state2):
    """
    Computes the distance metric between two states.
    (Absolute value of the difference between mean of streamfunction in box)
    """
    state1_pos = np.mean(state1[3, 5:15, 32:48], axis=(0, 1))
    state2_pos = np.mean(state2[3, 5:15, 32:48], axis=(0, 1))
    return abs(state1_pos - state2_pos)

def mapper(fmodel, input_params, state, suffix: str = ""):
    """
    Maps a state to an attractor (ON=-1 or OFF=1).
    """
    traj = Trajectory(0, fmodel, input_params, workdir = Path(f"./.edge_tmp/tmp_wkdir{suffix}"))

    # The model expect a state as a numpy file
    state_file = "init_state.npy"
    state_path = Path(traj._workdir / state_file)
    np.save(state_path, state)
    traj._fmodel.set_current_state(state_path)

    # Advance the model to the final time specified in toml file
    traj.advance()
    # Delete temporary workdir
    shutil.rmtree(f"./.edge_tmp/tmp_wkdir{suffix}")

    return int(traj.get_score_array()[-1])

def forward_finite_time(fmodel, input_params, state, tfinal, suffix: str = ""):
    traj = Trajectory(0, fmodel, input_params, workdir = Path(f"./.edge_tmp/tmp_wkdir{suffix}"))

    # The model expect a state as a numpy file
    state_file = "init_state.npy"
    state_path = Path(traj._workdir / state_file)
    np.save(state_path, state)
    traj._fmodel.set_current_state(state_path)

    # Advance the model to the final time specified in toml file
    traj.advance(tfinal)

    # Get the final state array
    final_state = np.load(traj.get_last_state())

    # Delete temporary workdir
    shutil.rmtree(f"./.edge_tmp/tmp_wkdir{suffix}")

    return final_state



def bisect_to_edge(fmodel, input_params, state1, state2, abstol=1e-3):
    """
    Bisects between two states until ending up with two states that lie on either side
    of the basin boundary but at a distance of less than 'abstol'.
    """
    mapper1 = mapper(fmodel, input_params, state1, suffix = "1")
    mapper2 = mapper(fmodel, input_params, state2, suffix = "2")

    if mapper1 == mapper2:
        raise("Both initial states belong to the same basin of attraction.")
    
    d = dist(state1, state2)
    while d > abstol:
        state = (state1 + state2)/2
        if mapper(fmodel, input_params, state, suffix = "0") == mapper(fmodel, input_params, state1, suffix = "1"):
            state1 = state
        elif mapper(fmodel, input_params, state, suffix = "0") == mapper(fmodel, input_params, state2, suffix = "2"):
            state2 = state
        else:
            raise("Bisected state could not be mapped to any of the given attractors.")
        d = dist(state1, state2)
        print(f"Distance : {d} , tol {abstol}")

    return state1, state2


def edgetracking(fmodel, input_params, dt_increment, state1, state2, eps1=1e-3, eps2=5e-3, maxiter=100):
    """
    Edge tracking algorithm.

    Tracks along the basin boundary starting from initial states 'state1' and 'state2'.
    Stops after 'maxiter' iterations.
    Returns a triple of lists of states (upper, lower, edgetrack).

    Keyword args
    ============
    - eps1: bisection distance threshold
    - eps2: divergence distance threshold
    - maxiter: number of iterations until the algorithm stops 
    """

    upper, lower, edgetrack = [], [], []

    iter = 0
    while iter <= maxiter:

        state1, state2 = bisect_to_edge(fmodel, input_params, state1, state2, abstol=eps1)

        d = dist(state1, state2)
        while d < eps2:
            state1 = forward_finite_time(fmodel, input_params, state1, dt_increment, suffix = "1")
            state2 = forward_finite_time(fmodel, input_params, state2, dt_increment, suffix = "2")

            d = dist(state1, state2)

        upper.append(state1)
        lower.append(state2)
        edgetrack.append((state1+state2)/2)

        iter += 1
        print(iter)
    
    return upper, lower, edgetrack
