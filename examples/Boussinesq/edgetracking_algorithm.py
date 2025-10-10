import logging
import shutil
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.trajectory import Trajectory

_logger = logging.getLogger(__name__)


def dist(state1: npt.NDArray[np.number], state2: npt.NDArray[np.number]) -> float:
    """Computes the distance metric between two states.

    (Absolute value of the difference between mean of streamfunction in box)
    """
    state1_pos = np.mean(state1[3, 5:15, 32:48], axis=(0, 1))
    state2_pos = np.mean(state2[3, 5:15, 32:48], axis=(0, 1))
    return abs(state1_pos - state2_pos)


def mapper(fmodel: Any, input_params: dict[Any, Any], state: npt.NDArray[np.number], suffix: str = "") -> int:
    """Maps a state to an attractor (ON=-1 or OFF=1)."""
    input_params["trajectory"]["sparse_freq"] = 1000
    traj = Trajectory(0, fmodel, input_params, workdir=Path(f"./.edge_tmp/tmp_wkdir{suffix}"))

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


def forward_finite_time(
    fmodel: Any, input_params: dict[Any, Any], state: npt.NDArray[np.number], tfinal: float, suffix: str = ""
) -> npt.NDArray[np.number]:
    """Advance the model by a finite time."""
    input_params["trajectory"]["sparse_freq"] = 1
    traj = Trajectory(0, fmodel, input_params, workdir=Path(f"./.edge_tmp/tmp_wkdir{suffix}"))

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


def bisect_to_edge(
    fmodel: Any,
    input_params: dict[Any, Any],
    state1: npt.NDArray[np.number],
    state2: npt.NDArray[np.number],
    abstol: float = 1e-3,
) -> tuple[npt.NDArray[np.number], npt.NDArray[np.number]]:
    """Bisects between two states until ending up with two states.

    These two states lie on either side
    of the basin boundary but at a distance of less than 'abstol'.
    """
    mapper1 = mapper(fmodel, input_params, state1, suffix="1")
    mapper2 = mapper(fmodel, input_params, state2, suffix="2")

    if mapper1 == mapper2:
        err_msg = "Both initial states belong to the same basin of attraction."
        _logger.exception(err_msg)
        raise RuntimeError(err_msg)

    mapper0 = None

    d = dist(state1, state2)
    while d > abstol:
        state = (state1 + state2) / 2
        mapper0 = mapper(fmodel, input_params, state, suffix="0")
        if mapper0 == mapper1:
            state1 = state
        elif mapper0 == mapper2:
            state2 = state
        else:
            err_msg = "Bisected state could not be mapped to any of the given attractors."
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)
        d = dist(state1, state2)
        inf_msg = f"Distance : {d} , tol {abstol}"
        _logger.info(inf_msg)

    return state1, state2


def edgetracking(
    fmodel: Any,
    input_params: dict[Any, Any],
    dt_increment: float,
    state1: npt.NDArray[np.number],
    state2: npt.NDArray[np.number],
    eps1: float = 1e-3,
    eps2: float = 5e-3,
    maxiter: int = 100,
) -> tuple[list[npt.NDArray[np.number]], list[npt.NDArray[np.number]], list[npt.NDArray[np.number]]]:
    """Edge tracking algorithm.

    Tracks along the basin boundary starting from initial states 'state1' and 'state2'.
    Stops after 'maxiter' iterations.
    Returns a triple of lists of states (upper, lower, edgetrack).

    Keyword Args:
    ============
    - eps1: bisection distance threshold
    - eps2: divergence distance threshold
    - maxiter: number of iterations until the algorithm stops
    """
    upper, lower, edgetrack = [], [], []

    ite_counter = 0
    while ite_counter <= maxiter:
        inf_msg = f" ## -> iteration : {ite_counter}"
        _logger.info(inf_msg)

        state1, state2 = bisect_to_edge(fmodel, input_params, state1, state2, abstol=eps1)

        d = dist(state1, state2)
        total_model_time = 0.0
        while d < eps2:
            state1 = forward_finite_time(fmodel, input_params, state1, dt_increment, suffix="1")
            state2 = forward_finite_time(fmodel, input_params, state2, dt_increment, suffix="2")
            total_model_time += dt_increment

            d = dist(state1, state2)

        inf_msg = f"Reached d = {d} after running the model for {total_model_time}"
        _logger.info(inf_msg)

        upper.append(state1)
        lower.append(state2)
        edgetrack.append((state1 + state2) / 2)

        ite_counter += 1

    return upper, lower, edgetrack
