from __future__ import annotations
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Optional
import numpy as np
import numpy.typing as npt
from pytams.xmlutils import dict_to_xml
from pytams.xmlutils import make_xml_snapshot
from pytams.xmlutils import new_element
from pytams.xmlutils import read_xml_snapshot
from pytams.xmlutils import xml_to_dict

_logger = logging.getLogger(__name__)


class WallTimeLimit(Exception):
    """Exception for running into wall time limit."""
    pass

def formTrajID(n: int) -> str:
    """Helper to assemble a trajectory ID string."""
    return f"traj{n:06}"

def getIndexFromID(identity: str) -> int:
    """Helper to get trajectory index from ID string."""
    return int(identity[-6:])

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
    time : float
    score : float
    noise : Any
    state : Any | None = None

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

    Attributes:
        _parameters_full : the full parameters dictionary
        _tid : the trajectory index
        _checkFile : the trajectory checkpoint file
        _workdir : the model working directory
        _score_max : the maximum score
        _snaps : a list of snapshots
        _step : the current step counter
        _t_cur : the current time
        _t_end : the end time
        _dt : the stochastic time step size
    """

    def __init__(self,
                 trajId: int,
                 fmodel_t: Any,
                 parameters: dict,
                 workdir: os.PathLike | None = None) -> None:
        """Create a trajectory.

        Args:
            trajId: a int for the trajectory index
            fmodel_t: the forward model type
            parameters: a dictionary of input parameters
            workdir: an optional working directory
        """
        # Stash away the full parameters dict
        self._parameters_full : dict = parameters

        traj_params = parameters.get("trajectory", {})
        if ("end_time" not in traj_params or
            "step_size" not in traj_params):
            err_msg = "Trajectory 'end_time' and 'step_size' must be specified in the input file !"
            _logger.error(err_msg)
            raise ValueError

        # The workdir is a runtime parameter, not saved in the chkfile.
        self._tid : int = trajId
        self._checkFile : os.PathLike = Path(f"{self.idstr()}.xml")
        self._workdir : os.PathLike = Path(".") if workdir is None else workdir
        self._score_max : float = 0.0
        self._has_ended : bool = False
        self._has_converged : bool = False

        # TAMS is expected to start at t = 0.0, but the forward model
        # itself can have a different internal starting point
        # or an entirely different time scale.
        self._step : int = 0
        self._t_cur : float = 0.0
        self._t_end : float = traj_params.get("end_time")
        self._dt : float = traj_params.get("step_size")

        # Trajectory convergence is defined by a target score, with
        # the score provided by the forward model, mapping the model state to
        # a s \in [0,1]. A default value of 0.95 is provided.
        self._convergedVal : float = traj_params.get("targetscore", 0.95)

        # For large models, the state may not be available at each snapshot due
        # to memory constraint (both in-memory and on-disk). Sparse state can
        # be specified. Finally, writing a chkfile to disk at each step might
        # incur a performance hit and is by default disabled.
        self._sparse_state_int : int = traj_params.get("sparse_freq", 1)
        self._sparse_state_beg : int = traj_params.get("sparse_start", 0)
        self._write_chkfile_all : bool = traj_params.get("chkfile_dump_all", False)

        # List of snapshots
        self._snaps : list[Snapshot] = []

        # When using sparse state or for other reasons, the noise for the next few
        # steps might be already available. This backlog is used to store them.
        self._noise_backlog : list[Any] = []

        # Each trajectory has its own instance of the forward model
        self._fmodel = fmodel_t(parameters,
                                self.idstr(),
                                self._workdir)

    def set_checkfile(self, path: os.PathLike) -> None:
        """Setter of the trajectory checkFile.

        Args:
            path: the new checkFile
        """
        self._checkFile = path

    def set_work_dir(self, path: os.PathLike) -> None:
        """Setter of the trajectory working directory.

        And propagate the workdir to the forward model.

        Args:
            path: the new working directory
        """
        self._workdir = path
        self._fmodel.setWorkDir(path)

    def get_work_dir(self) -> os.PathLike:
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
        return formTrajID(self._tid)

    def advance(self,
                t_end: float = 1.0e12,
                walltime: float = 1.0e12) -> None:
        """Advance the trajectory to a prescribed end time.

        This is the main time loop of the trajectory object.
        Unless specified otherwise, the trajectory will advance until
        the end time is reached or the model has converged.

        If the walltime limit is reached, a WallTimeLimit exception is raised.
        Note that this exception is treated as a warning not an error by the
        TAMS workers.

        Args:
            t_end: the end time of the advance
            walltime: a walltime limit to advance the model to t_end

        Returns:
            None

        Raises:
            WallTimeLimit: if the walltime limit is reached
        """
        startTime = time.monotonic()
        remainingTime = walltime - time.monotonic() + startTime
        end_time = min(t_end, self._t_end)

        # Set the initial snapshot
        # Always add the initial state
        if self._step == 0:
           self._snaps.append(Snapshot(time=self._t_cur,
                                       score=self._fmodel.score(),
                                       noise=self._fmodel.getNoise(),
                                       state=self._fmodel.getCurState()
                                       )
                              )

        while (
            self._t_cur < end_time
            and not self._has_converged
            and remainingTime >= 0.05 * walltime
        ):
            # Do a single and keep track of remaining walltime
            score = self._one_step()

            remainingTime = walltime - time.monotonic() + startTime

        if self._t_cur >= self._t_end or self._has_converged:
            self._has_ended = True

        # If trajectory ended but no state
        # was stored on the last step -> pop snapshot and
        # force a state for the final step.
        if self._has_ended and not self._snaps[-1].has_state():
            self._snaps.pop(-1)
            self._snaps.append(Snapshot(self._t_cur,
                                        score,
                                        self._fmodel.getNoise(),
                                        self._fmodel.getCurState()
                                        )
                               )

        if self._has_ended:
            self._fmodel.clear()

        if remainingTime < 0.05 * walltime:
            warn_msg = f"{self.idstr()} ran out of time in advance()"
            _logger.warning(warn_msg)
            raise WallTimeLimit(warn_msg)

    def _one_step(self) -> float:
        """Perform a single step of the forward model.

        Perform a single time step of the forward model. This
        function will also set the noise to use for the next step
        in the forward model if a backlog is available.
        """
        if self._noise_backlog:
            self._fmodel.setNoise(self._noise_backlog[0])
            self._noise_backlog.pop(0)

        try:
            dt = self._fmodel.advance(self._dt)
        except Exception:
            err_msg = f"ForwardModel advance error at step {self._step:08}"
            _logger.error(err_msg)
            raise

        self._step += 1
        self._t_cur = self._t_cur + dt
        score = self._fmodel.score()

        if ((self._sparse_state_beg + self._step)%self._sparse_state_int == 0):
            self._snaps.append(Snapshot(time=self._t_cur,
                                        score=score,
                                        noise=self._fmodel.getNoise(),
                                        state=self._fmodel.getCurState()
                                        )
                               )
        else:
            self._snaps.append(Snapshot(time=self._t_cur,
                                        score=score,
                                        noise=self._fmodel.getNoise(),
                                        )
                               )

        if self._write_chkfile_all:
            self.store()

        if score > self._score_max:
            self._score_max = score

        if score >= self._convergedVal:
            self._has_converged = True

        return score


    @classmethod
    def restore_from_checkfile(
        cls,
        chkPoint: os.PathLike,
        fmodel_t: Any,
        parameters: dict,
        workdir: os.PathLike | None = None,
    ) -> Trajectory:
        """Return a trajectory restored from an XML chkfile."""
        assert Path(chkPoint).exists() is True

        # Read in trajectory metadata
        tree = ET.parse(chkPoint.absolute())
        root = tree.getroot()
        metadata = xml_to_dict(root.find("metadata"))
        t_id = metadata["id"]

        restTraj = Trajectory(trajId=t_id,
                              fmodel_t=fmodel_t,
                              parameters=parameters,
                              workdir = workdir,
                              )

        restTraj._t_end = metadata["t_end"]
        restTraj._t_cur = metadata["t_cur"]
        restTraj._dt = metadata["dt"]
        restTraj._score_max = metadata["score_max"]
        restTraj._has_ended = metadata["ended"]
        restTraj._has_converged = metadata["converged"]

        snapshots = root.find("snapshots")
        if snapshots is not None:
            for snap in snapshots:
                time, score, noise, state = read_xml_snapshot(snap)
                restTraj._snaps.append(Snapshot(time, score, noise, state))

        # Remove snapshots from the list until a state
        # is available
        need_update = False
        for k in range(len(restTraj._snaps)-1,-1,-1):
            if not restTraj._snaps[k].has_state():
                restTraj._noise_backlog.append(restTraj._snaps[k].noise)
                restTraj._snaps.pop(k)
                need_update = True
            else:
                break

        restTraj._fmodel.setCurState(restTraj._snaps[-1].state)
        restTraj._t_cur = restTraj._snaps[-1].time

        # Reset score_max, ended and converged
        if need_update:
            restTraj.update_metadata()

        # Enable the model to perform tweaks
        # after a trajectory restore
        restTraj._fmodel.post_trajectory_restore_hook(len(restTraj._snaps), restTraj._t_cur)

        return restTraj

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
            restTraj = Trajectory(
                trajId=rst_traj.id(),
                fmodel_t=type(from_traj._fmodel),
                parameters=from_traj._parameters_full,
                workdir=rst_traj.get_work_dir(),
            )
            restTraj.set_checkfile(rst_traj.get_checkfile())
            return restTraj

        # To ensure that TAMS converges, branching occurs on
        # the first snapshot with a score *strictly* above the target
        # Traverse the trajectory until a snapshot with a score >
        # the target is encountered
        high_score_idx = 0
        last_snap_with_state = 0
        while from_traj._snaps[high_score_idx].score <= score:
            high_score_idx += 1
            if (from_traj._snaps[high_score_idx].has_state()):
                last_snap_with_state = high_score_idx

        # Init empty trajectory
        restTraj = Trajectory(
            trajId=rst_traj.id(),
            fmodel_t=type(from_traj._fmodel),
            parameters=from_traj._parameters_full,
            workdir=rst_traj.get_work_dir(),
        )
        restTraj.set_checkfile(rst_traj.get_checkfile())

        # Append snapshots, up to high_score_idx + 1 to
        # ensure > behavior
        for k in range(high_score_idx + 1):
            if (k <= last_snap_with_state):
                restTraj._snaps.append(from_traj._snaps[k])
            else:
                restTraj._noise_backlog.append(from_traj._snaps[k].noise)

        # Update trajectory metadata
        restTraj._fmodel.setCurState(restTraj._snaps[-1].state)
        restTraj._t_cur = restTraj._snaps[-1].time
        restTraj._score_max = restTraj._snaps[-1].score

        # Enable the model to perform tweaks
        # after a trajectory restart
        restTraj._fmodel.post_trajectory_restart_hook(len(restTraj._snaps), restTraj._t_cur)

        return restTraj

    def store(self, traj_file: Optional[os.PathLike] = None) -> None:
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
        snaps_xml = ET.SubElement(root, "snapshots")
        for k in range(len(self._snaps)):
            snaps_xml.append(
                make_xml_snapshot(k,
                                  self._snaps[k].time,
                                  self._snaps[k].score,
                                  self._snaps[k].noise,
                                  self._snaps[k].state)
            )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        if traj_file is not None:
            tree.write(traj_file.absolute())
        else:
            tree.write(self._checkFile.absolute())

    def update_metadata(self) -> None:
        """Update trajectory score/ending metadata.

        Update the maximum of the score function over the trajectory
        as well as the bool values for has_converged and has_ended.
        """
        new_score_max = 0.0
        for snap in self._snaps:
            if (snap.score > new_score_max):
                new_score_max = snap.score
        self._score_max = new_score_max
        if new_score_max > self._convergedVal:
            self._has_converged = True
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

    def get_checkfile(self) -> os.PathLike:
        """Return the trajectory check file name."""
        return self._checkFile

    def get_time_array(self) -> npt.NDArray[np.float64]:
        """Return the trajectory time instants."""
        times = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            times[k] = self._snaps[k].time
        return times

    def get_score_array(self) -> npt.NDArray[np.float64]:
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

    def get_last_state(self) -> Any:
        """Return the last state in the trajectory."""
        for snap in reversed(self._snaps):
            if snap.has_state():
                return snap.state
