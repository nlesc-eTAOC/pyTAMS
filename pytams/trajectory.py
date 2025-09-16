from __future__ import annotations
import logging
import shutil
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.xmlutils import dict_to_xml
from pytams.xmlutils import make_xml_snapshot
from pytams.xmlutils import new_element
from pytams.xmlutils import read_xml_snapshot
from pytams.xmlutils import xml_to_dict

if TYPE_CHECKING:
    from pytams.fmodel import ForwardModelBaseClass

_logger = logging.getLogger(__name__)


class WallTimeLimitError(Exception):
    """Exception for running into wall time limit."""


def form_trajectory_id(n: int, nb: int = 0) -> str:
    """Helper to assemble a trajectory ID string.

    Args:
        n : trajectory index
        nb : number of branching

    Returns:
        trajectory ID
    """
    return f"traj{n:06}_{nb:04}"


def get_index_from_id(identity: str) -> tuple[int, int]:
    """Helper to get trajectory index from ID string.

    Args:
        identity : trajectory ID

    Returns:
        trajectory index and number of branching
    """
    return int(identity[-10:-5]), int(identity[-4:])


@dataclass
class Snapshot:
    """A dataclass defining a snapshot.

    Gathering what defines a snapshot into an object.
    The time and score are of float type, but the
    actual type of the noise and state are completely
    determined by the forward model.
    A snapshot is allowed to have a state or not to
    accomodate memory savings.

    Attributes:
        time : snapshot time
        score : score function value
        noise : noise used to reach this snapshot
        state : model state
    """

    time: float
    score: float
    noise: Any
    state: Any | None = None

    def has_state(self) -> bool:
        """Check if snapshot has state.

        Returns:
            bool : True if state is not None
        """
        return self.state is not None


class Trajectory:
    """A class defining a stochastic trajectory.

    The trajectory class is a container for time-ordered snapshots.
    It contains an instance of the forward model, current and end times, and
    a list of the model snapshots. Note that the class uses a plain list of snapshots
    and not a more computationally efficient data structure such as a numpy array
    for convenience. It is assumed that the computational cost of running TAMS
    reside in the forward model and the overhead of the trajectory class is negligible.

    It also provide the forward model with the necessary context to advance in time,
    method to move forward in time, methods to save/load the trajectory to/from disk
    as well as accessor to the trajectory history (time, state, score, ...).

    The _computed_steps variable store the number of steps actually taken by the
    trajectory. It differs from the _step variable when a trajectory is branched
    from an ancestor.

    Attributes:
        _parameters_full : the full parameters dictionary
        _tid : the trajectory index
        _checkFile : the trajectory checkpoint file
        _workdir : the model working directory
        _score_max : the maximum score
        _snaps : a list of snapshots
        _step : the current step counter
        _computed_steps : the number of steps explicitly advanced by the trajectory
        _t_cur : the current time
        _t_end : the end time
        _dt : the stochastic time step size
    """

    def __init__(
        self,
        traj_id: int,
        fmodel_t: type[ForwardModelBaseClass] | None,
        parameters: dict[Any, Any],
        workdir: Path | None = None,
        frozen: bool = False,
    ) -> None:
        """Create a trajectory.

        Args:
            traj_id: a int for the trajectory index
            fmodel_t: the forward model type
            parameters: a dictionary of input parameters
            workdir: an optional working directory
            frozen: whether the trajectory is frozen (no fmodel)
        """
        # Stash away the full parameters dict
        self._parameters_full: dict[Any, Any] = parameters

        traj_params = parameters.get("trajectory", {})
        if "end_time" not in traj_params or "step_size" not in traj_params:
            err_msg = "Trajectory 'end_time' and 'step_size' must be specified in the input file !"
            _logger.error(err_msg)
            raise ValueError

        # The workdir is a runtime parameter, not saved in the chkfile.
        self._tid: int = traj_id
        self._workdir: Path = Path.cwd() if workdir is None else workdir
        self._score_max: float = 0.0
        self._has_ended: bool = False
        self._has_converged: bool = False
        self._computed_steps: int = 0

        # TAMS is expected to start at t = 0.0, but the forward model
        # itself can have a different internal starting point
        # or an entirely different time scale.
        self._step: int = 0
        self._t_cur: float = 0.0
        self._t_end: float = traj_params.get("end_time")
        self._dt: float = traj_params.get("step_size")

        # Trajectory convergence is defined by a target score, with
        # the score provided by the forward model, mapping the model state to
        # a s \in [0,1]. A default value of 0.95 is provided.
        self._convergedVal: float = traj_params.get("targetscore", 0.95)

        # List of snapshots
        self._snaps: list[Snapshot] = []

        # When using sparse state or for other reasons, the noise for the next few
        # steps might be already available. This backlog is used to store them.
        self._noise_backlog: list[Any] = []

        # Keep track of the branching history during TAMS
        # iterations
        self._branching_history: list[int] = []

        # For large models, the state may not be available at each snapshot due
        # to memory constraint (both in-memory and on-disk). Sparse state can
        # be specified. Finally, writing a chkfile to disk at each step might
        # incur a performance hit and is by default disabled.
        self._sparse_state_int: int = traj_params.get("sparse_freq", 1)
        self._sparse_state_beg: int = traj_params.get("sparse_start", 0) + 1
        self._write_chkfile_all: bool = traj_params.get("chkfile_dump_all", False)
        self._checkFile: Path = Path(f"{self.idstr()}.xml")

        # Each trajectory has its own instance of the forward model
        if frozen or fmodel_t is None:
            self._fmodel = None
        else:
            self._fmodel = fmodel_t(parameters, self.idstr(), self._workdir)

    def set_checkfile(self, path: Path) -> None:
        """Setter of the trajectory checkFile.

        Args:
            path: the new checkFile
        """
        self._checkFile = path

    def set_workdir(self, path: Path) -> None:
        """Setter of the trajectory working directory.

        And propagate the workdir to the forward model.

        Args:
            path: the new working directory
        """
        self._workdir = path
        if self._fmodel is not None:
            self._fmodel.set_workdir(path)

    def get_workdir(self) -> Path:
        """Get the trajectory working directory.

        Returns:
            the working directory
        """
        return self._workdir

    def id(self) -> int:
        """Return trajectory Id.

        Returns:
            the trajectory id
        """
        return self._tid

    def idstr(self) -> str:
        """Return trajectory Id as a padded string.

        Returns:
            the trajectory id as a string
        """
        return form_trajectory_id(self._tid, self.get_nbranching())

    def advance(self, t_end: float = 1.0e12, walltime: float = 1.0e12) -> None:
        """Advance the trajectory to a prescribed end time.

        This is the main time loop of the trajectory object.
        Unless specified otherwise, the trajectory will advance until
        the end time is reached or the model has converged.

        If the walltime limit is reached, a WallTimeLimitError exception is raised.
        Note that this exception is treated as a warning not an error by the
        TAMS workers.

        Args:
            t_end: the end time of the advance
            walltime: a walltime limit to advance the model to t_end

        Returns:
            None

        Raises:
            WallTimeLimitError: if the walltime limit is reached
            RuntimeError: if the model advance run into a problem
        """
        start_time = time.monotonic()
        remaining_time = walltime - time.monotonic() + start_time
        end_time = min(t_end, self._t_end)

        if not self._fmodel:
            err_msg = f"Trajectory {self.idstr()} is frozen, without forward model. Advance() deactivated."
            _logger.error(err_msg)
            raise RuntimeError(err_msg)

        while self._t_cur < end_time and not self._has_converged and remaining_time >= 0.05 * walltime:
            # Do a single step and keep track of remaining walltime
            _ = self._one_step()

            remaining_time = walltime - time.monotonic() + start_time

        if self._t_cur >= self._t_end or self._has_converged:
            self._has_ended = True

        if self._has_ended:
            self._fmodel.clear()

        if remaining_time < 0.05 * walltime:
            warn_msg = f"{self.idstr()} ran out of time in advance()"
            _logger.warning(warn_msg)
            raise WallTimeLimitError(warn_msg)

    def _one_step(self) -> float:
        """Perform a single step of the forward model.

        Perform a single time step of the forward model. This
        function will also set the noise to use for the next step
        in the forward model if a backlog is available.
        """
        if not self._fmodel:
            err_msg = f"Trajectory {self.idstr()} is frozen, without forward model. Advance() deactivated."
            _logger.exception(err_msg)
            raise RuntimeError(err_msg)

        # Add the initial snapshot to the list
        if self._step == 0:
            self._setup_noise()
            self._append_snapshot()

        # Trigger storing the end state of the current time step
        # if the next trajectory snapshot needs it
        need_end_state = (self._sparse_state_beg + self._step + 1) % self._sparse_state_int == 0

        try:
            dt = self._fmodel.advance(self._dt, need_end_state)
        except Exception:
            err_msg = f"ForwardModel advance error at step {self._step:08}"
            _logger.exception(err_msg)
            raise

        self._step += 1
        self._t_cur = self._t_cur + dt
        score = self._fmodel.score()

        # Prepare the noise for the next step
        self._setup_noise()

        # Append a snapshot at the beginning of the time step
        self._append_snapshot(score)

        if self._write_chkfile_all:
            self.store()

        self._score_max = max(self._score_max, score)

        # The default ABC method simply check for a score above
        # the target value, but concrete implementations can override
        # with mode complex convergence criteria
        self._has_converged = self._fmodel.check_convergence(self._step, self._t_cur, score, self._convergedVal)

        # Increment the computed step counter
        self._computed_steps += 1

        return score

    def _setup_noise(self) -> None:
        """Prepare the noise for the next step."""
        # Set the noise for the next model step
        # if a noise backlog is available, use it otherwise
        # make a new noise increment
        if self._fmodel:
            if self._noise_backlog:
                self._fmodel.set_noise(self._noise_backlog.pop())
            else:
                self._fmodel.set_noise(self._fmodel.make_noise())

    def _append_snapshot(self, score: float | None = None) -> None:
        """Append the current snapshot to the trajectory list."""
        # Append the current snapshot to the trajectory list
        if self._fmodel:
            need_state = (self._sparse_state_beg + self._step) % self._sparse_state_int == 0 or self._step == 0
            self._snaps.append(
                Snapshot(
                    time=self._t_cur,
                    score=score if score else self._fmodel.score(),
                    noise=self._fmodel.get_noise(),
                    state=self._fmodel.get_current_state() if need_state else None,
                ),
            )

    @classmethod
    def restore_from_checkfile(
        cls,
        checkfile: Path,
        fmodel_t: type[ForwardModelBaseClass],
        parameters: dict[Any, Any],
        workdir: Path | None = None,
        frozen: bool = False,
    ) -> Trajectory:
        """Return a trajectory restored from an XML chkfile."""
        if not checkfile.exists():
            err_msg = f"Trajectory {checkfile} does not exist."
            _logger.exception(err_msg)
            raise FileNotFoundError

        # Read in trajectory metadata
        tree = ET.parse(checkfile.absolute())
        root = tree.getroot()
        metadata = xml_to_dict(root.find("metadata"))
        t_id = metadata["id"]

        rest_traj = Trajectory(
            traj_id=t_id,
            fmodel_t=fmodel_t,
            parameters=parameters,
            workdir=workdir,
            frozen=frozen,
        )

        rest_traj._t_end = metadata["t_end"]
        rest_traj._t_cur = metadata["t_cur"]
        rest_traj._dt = metadata["dt"]
        rest_traj._score_max = metadata["score_max"]
        rest_traj._has_ended = metadata["ended"]
        rest_traj._has_converged = metadata["converged"]
        rest_traj._branching_history = metadata["branching_history"].tolist()
        rest_traj._computed_steps = metadata.get("c_step", 0)

        snapshots = root.find("snapshots")
        if snapshots is not None:
            for snap in snapshots:
                time, score, noise, state = read_xml_snapshot(snap)
                rest_traj._snaps.append(Snapshot(time, score, noise, state))

        # Current step with python indexing, so remove 1
        rest_traj._step = len(rest_traj._snaps) - 1

        # If the trajectory is frozen, that is all we need. Otherwise
        # handle sparse state, noise backlog and necessary fmodel initialization
        if rest_traj._fmodel:
            # Remove snapshots from the list until a state is available
            need_update = False
            for k in range(len(rest_traj._snaps) - 1, -1, -1):
                if not rest_traj._snaps[k].has_state():
                    # Append the noise history to the backlog
                    rest_traj._noise_backlog.append(rest_traj._snaps[k].noise)
                    rest_traj._snaps.pop()
                    need_update = True
                else:
                    # Because the noise in the snapshot is the noise
                    # used to reach the next state, append the last to the backlog too
                    rest_traj._noise_backlog.append(rest_traj._snaps[k].noise)
                    break

            rest_traj._t_cur = rest_traj._snaps[-1].time

            # Current step with python indexing, so remove 1
            rest_traj._step = len(rest_traj._snaps) - 1

            # Ensure everything is set to start the time stepping loop
            rest_traj._setup_noise()
            rest_traj._fmodel.set_current_state(rest_traj._snaps[-1].state)

            # Reset score_max, ended and converged
            if need_update:
                rest_traj.update_metadata()

            # Enable the model to perform tweaks
            # after a trajectory restore
            rest_traj._fmodel.post_trajectory_restore_hook(len(rest_traj._snaps) - 1, rest_traj._t_cur)

        return rest_traj

    @classmethod
    def branch_from_trajectory(
        cls,
        from_traj: Trajectory,
        rst_traj: Trajectory,
        score: float,
    ) -> Trajectory:
        """Create a new trajectory.

        Loading the beginning of a provided trajectory
        for all entries with score below a given score.
        This effectively branches the trajectory.

        Although the rst_traj is provided as an argument, it is
        only used to set metadata of the branched trajectory.

        Args:
            from_traj: an already existing trajectory to restart from
            rst_traj: the trajectory being restarted
            score: a threshold score
        """
        # Check for empty trajectory
        if len(from_traj._snaps) == 0:
            tid, nb = get_index_from_id(rst_traj.idstr())
            new_workdir = Path(rst_traj.get_workdir().parents[0] / form_trajectory_id(tid, nb + 1))
            fmodel_t = type(from_traj._fmodel) if from_traj._fmodel else None
            rest_traj = Trajectory(
                traj_id=rst_traj.id(),
                fmodel_t=fmodel_t,
                parameters=from_traj._parameters_full,
                workdir=new_workdir,
            )
            rest_traj.set_checkfile(Path(rst_traj.get_checkfile().parents[0] / f"{rest_traj.idstr()}.xml"))
            return rest_traj

        # To ensure that TAMS converges, branching occurs on
        # the first snapshot with a score *strictly* above the target
        # Traverse the trajectory until a snapshot with a score >
        # the target is encountered
        high_score_idx = 0
        last_snap_with_state = 0
        while from_traj._snaps[high_score_idx].score <= score:
            high_score_idx += 1
            if from_traj._snaps[high_score_idx].has_state():
                last_snap_with_state = high_score_idx

        # Init empty trajectory
        tid, nb = get_index_from_id(rst_traj.idstr())
        new_workdir = Path(rst_traj.get_workdir().parents[0] / form_trajectory_id(tid, nb + 1))
        fmodel_t = type(from_traj._fmodel) if from_traj._fmodel else None
        rest_traj = Trajectory(
            traj_id=rst_traj.id(),
            fmodel_t=fmodel_t,
            parameters=from_traj._parameters_full,
            workdir=new_workdir,
        )
        rest_traj._branching_history = rst_traj._branching_history
        rest_traj._branching_history.append(from_traj.id())
        rest_traj.set_checkfile(Path(rst_traj.get_checkfile().parents[0] / f"{rest_traj.idstr()}.xml"))

        # Append snapshots, up to high_score_idx + 1 to
        # ensure > behavior
        for k in range(high_score_idx + 1):
            if k < last_snap_with_state:
                rest_traj._snaps.append(from_traj._snaps[k])
            elif k == last_snap_with_state:
                rest_traj._snaps.append(from_traj._snaps[k])
                rest_traj._noise_backlog.append(from_traj._snaps[k].noise)
            else:
                rest_traj._noise_backlog.append(from_traj._snaps[k].noise)

        # Reverse the backlog to ensure correct order
        rest_traj._noise_backlog.reverse()

        # Update trajectory metadata
        rest_traj._t_cur = rest_traj._snaps[-1].time
        rest_traj._step = len(rest_traj._snaps) - 1
        if rest_traj._fmodel:
            rest_traj._setup_noise()
            rest_traj._fmodel.set_current_state(rest_traj._snaps[-1].state)
            rest_traj.update_metadata()

            # Enable the model to perform tweaks
            # after a trajectory restart
            rest_traj._fmodel.post_trajectory_branching_hook(len(rest_traj._snaps) - 1, rest_traj._t_cur)

        return rest_traj

    def store(self, traj_file: Path | None = None) -> None:
        """Store the trajectory to an XML chkfile."""
        root = ET.Element(self.idstr())
        root.append(dict_to_xml("params", self._parameters_full["trajectory"]))
        mdata = ET.SubElement(root, "metadata")
        mdata.append(new_element("id", self._tid))
        mdata.append(new_element("t_cur", self._t_cur))
        mdata.append(new_element("t_end", self._t_end))
        mdata.append(new_element("dt", self._dt))
        mdata.append(new_element("score_max", self._score_max))
        mdata.append(new_element("ended", self._has_ended))
        mdata.append(new_element("converged", self._has_converged))
        mdata.append(new_element("branching_history", np.array(self._branching_history, dtype=int)))
        mdata.append(new_element("c_step", self._computed_steps))
        snaps_xml = ET.SubElement(root, "snapshots")
        for k in range(len(self._snaps)):
            snaps_xml.append(
                make_xml_snapshot(
                    k,
                    self._snaps[k].time,
                    self._snaps[k].score,
                    self._snaps[k].noise,
                    self._snaps[k].state,
                ),
            )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)

        if traj_file is not None:
            tree.write(traj_file.as_posix())
        else:
            tree.write(self._checkFile.as_posix())

    def update_metadata(self) -> None:
        """Update trajectory score/ending metadata.

        Update the maximum of the score function over the trajectory
        as well as the bool values for has_converged and has_ended.
        """
        new_score_max = 0.0
        for snap in self._snaps:
            new_score_max = max(new_score_max, snap.score)
        self._score_max = new_score_max
        if self._fmodel:
            self._has_converged = self._fmodel.check_convergence(
                self._step, self._t_cur, self._score_max, self._convergedVal
            )
        else:
            self._has_converged = False
        if self._t_cur >= self._t_end or self._has_converged:
            self._has_ended = True

    def current_time(self) -> float:
        """Return the current trajectory time."""
        return self._t_cur

    def step_size(self) -> float:
        """Return the time step size."""
        return self._dt

    def score_max(self) -> float:
        """Return the maximum of the score function."""
        return self._score_max

    def is_converged(self) -> bool:
        """Return True for converged trajectory."""
        return self._has_converged

    def has_ended(self) -> bool:
        """Return True for terminated trajectory."""
        return self._has_ended

    def has_started(self) -> bool:
        """Return True if computation has started."""
        return self._t_cur > 0.0

    def get_checkfile(self) -> Path:
        """Return the trajectory check file name."""
        return self._checkFile

    def get_time_array(self) -> npt.NDArray[np.number]:
        """Return the trajectory time instants."""
        times = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            times[k] = self._snaps[k].time
        return times

    def get_score_array(self) -> npt.NDArray[np.number]:
        """Return the trajectory scores."""
        scores = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            scores[k] = self._snaps[k].score
        return scores

    def get_noise_array(self) -> npt.NDArray[Any]:
        """Return the trajectory noises."""
        noises = np.zeros(len(self._snaps), dtype=type(self._snaps[0].noise))
        for k in range(len(self._snaps)):
            noises[k] = self._snaps[k].noise
        return noises

    def get_length(self) -> int:
        """Return the trajectory length."""
        return len(self._snaps)

    def get_nbranching(self) -> int:
        """Return the number of branching events."""
        return len(self._branching_history)

    def get_computed_steps_count(self) -> int:
        """Return the number of compute steps taken."""
        return self._computed_steps

    def get_last_state(self) -> Any | None:
        """Return the last state in the trajectory."""
        for snap in reversed(self._snaps):
            if snap.has_state():
                return snap.state

        return None

    def delete(self) -> None:
        """Clear the trajectory on-disk data."""
        self._checkFile.unlink()
        if self._workdir.exists():
            shutil.rmtree(self._workdir)
