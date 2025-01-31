from __future__ import annotations
import logging
import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
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

class ForwardModelAdvance(Exception):
    """Exception for forward model advance fail."""

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

    def hasState(self) -> bool:
        """Check if snapshot has state.

        Returns:
            bool : True if state is not None
        """
        return self.state is not None


class Trajectory:
    """A class defining a stochastic trajectory.

    Includes an instance of the forward model, current and end times, and
    a list of the model snapshots for each new maximum of the
    score function along the way.
    """

    def __init__(self,
                 fmodel_t: Any,
                 parameters: dict,
                 trajId: int) -> None:
        """Create a trajectory.

        Args:
            fmodel_t: the forward model type
            parameters: a dictionary of input parameters
            trajId: a int for the trajectory index
        """
        # Stash away the run parameters
        self._parameters : dict = parameters

        self._step : int = 0
        self._t_cur : float = 0.0
        self._t_end : float = parameters.get("trajectory",{}).get("end_time", 1.0)
        self._dt : float = parameters.get("trajectory",{}).get("step_size", 0.1)
        self._stoichForcingAmpl : float = parameters.get("trajectory",{}).get("stoichforcing", 0.5)
        self._convergedVal : float = parameters.get("trajectory",{}).get("targetscore", 0.95)
        self._sparse_state_int : int = parameters.get("trajectory",{}).get("sparse_int", 1)
        self._sparse_state_beg : int = parameters.get("trajectory",{}).get("sparse_beg", 0)
        self._write_chkfile_all : bool = parameters.get("trajectory",{}).get("chkfile_dump_all", False)

        # List of snapshots
        self._snaps : list[Snapshot] = []

        self._noise_backlog : list[Any] = []

        self._score_max : float = 0.0

        self._tid : int = trajId
        self._checkFile : str = f"{self.idstr()}.xml"

        self._has_ended : bool = False
        self._has_converged : bool = False

        # Each trajectory have its own instance of the model
        self._fmodel = fmodel_t(self._parameters, self.idstr())

    def setCheckFile(self, file: str) -> None:
        """Setter of the trajectory checkFile."""
        self._checkFile = file

    def id(self) -> int:
        """Return trajectory Id."""
        return self._tid

    def idstr(self) -> str:
        """Return trajectory Id."""
        return formTrajID(self._tid)

    def advance(self, t_end: float = 1.0e12, walltime: float = 1.0e12) -> None:
        """Advance the trajectory to a prescribed end time.

        Args:
            t_end: the end time of the advance
            walltime: a walltime limit to advance the model to t_end
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
        if self._has_ended and not self._snaps[-1].hasState():
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

    def _one_step(self) -> Any:
        """Perform a single step of the forward model."""
        if self._noise_backlog:
            self._fmodel.setNoise(self._noise_backlog[0])
            self._noise_backlog.pop(0)

        try:
            dt = self._fmodel.advance(self._dt, self._stoichForcingAmpl)
        except ForwardModelAdvance:
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
    def restoreFromChk(
        cls,
        chkPoint: str,
        fmodel_t: Any,
        parameters: dict,
    ) -> Trajectory:
        """Return a trajectory restored from an XML chkfile."""
        assert os.path.exists(chkPoint) is True

        # Read in trajectory metadata
        tree = ET.parse(chkPoint)
        root = tree.getroot()
        metadata = xml_to_dict(root.find("metadata"))
        t_id = metadata["id"]

        restTraj = Trajectory(fmodel_t=fmodel_t, parameters=parameters, trajId=t_id)

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
            if not restTraj._snaps[k].hasState():
                restTraj._noise_backlog.append(restTraj._snaps[k].noise)
                restTraj._snaps.pop(k)
                need_update = True
            else:
                break

        restTraj._fmodel.setCurState(restTraj._snaps[-1].state)
        restTraj._t_cur = restTraj._snaps[-1].time

        # Reset score_max, ended and converged
        if need_update:
            restTraj.updateMetadata()

        return restTraj

    @classmethod
    def restartFromTraj(
        cls,
        traj: Trajectory,
        rstId: int,
        score: float,
    ) -> Trajectory:
        """Create a new trajectory.

        Loading the beginning of a provided trajectory
        for all entries with score below a given score

        Args:
            traj: an already existing trajectory to restart from
            rstId: the id of the trajectory being restarted
            score: a threshold score
        """
        # Check for empty trajectory
        if len(traj._snaps) == 0:
            restTraj = Trajectory(
                fmodel_t=type(traj._fmodel),
                parameters=traj._parameters,
                trajId=rstId,
            )
            return restTraj

        # To ensure that TAMS converges, branching occurs on
        # the first snapshot with a score *strictly* above the target
        # Traverse the trajectory until a snapshot with a score >
        # the target is encountered
        high_score_idx = 0
        last_snap_with_state = 0
        while traj._snaps[high_score_idx].score <= score:
            high_score_idx += 1
            if (traj._snaps[high_score_idx].hasState()):
                last_snap_with_state = high_score_idx

        # Init empty trajectory
        restTraj = Trajectory(
            fmodel_t=type(traj._fmodel), parameters=traj._parameters, trajId=rstId
        )

        # Append snapshots, up to high_score_idx + 1 to
        # ensure > behavior
        for k in range(high_score_idx + 1):
            if (k <= last_snap_with_state):
                restTraj._snaps.append(traj._snaps[k])
            else:
                restTraj._noise_backlog.append(traj._snaps[k].noise)

        # Update trajectory metadata
        restTraj._fmodel.setCurState(restTraj._snaps[-1].state)
        restTraj._t_cur = restTraj._snaps[-1].time
        restTraj._score_max = restTraj._snaps[-1].score

        # Enable the model to perform tweaks
        # after a trajectory restart
        restTraj._fmodel.post_trajectory_restart_hook(len(restTraj._snaps), restTraj._t_cur)

        return restTraj

    def store(self, traj_file: Optional[str] = None) -> None:
        """Store the trajectory to an XML chkfile."""
        root = ET.Element(self.idstr())
        root.append(dict_to_xml("params", self._parameters["trajectory"]))
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
            tree.write(traj_file)
        else:
            tree.write(self._checkFile)

    def updateMetadata(self) -> None:
        """Update trajectory score/ending metadata."""
        new_score_max = 0.0
        for snap in self._snaps:
            if (snap.score > new_score_max):
                new_score_max = snap.score
        self._score_max = new_score_max
        if new_score_max > self._convergedVal:
            self._has_converged = True
        if self._t_cur >= self._t_end or self._has_converged:
            self._has_ended = True

    def ctime(self) -> float:
        """Return the current trajectory time."""
        return self._t_cur

    def stepSize(self) -> float:
        """Return the time step size."""
        return self._dt

    def scoreMax(self) -> float:
        """Return the maximum of the score function."""
        return self._score_max

    def isConverged(self) -> bool:
        """Return True for converged trajectory."""
        return self._has_converged

    def hasEnded(self) -> bool:
        """Return True for terminated trajectory."""
        return self._has_ended

    def hasStarted(self) -> bool:
        """Return True if computation has started."""
        return self._t_cur > 0.0

    def checkFile(self) -> str:
        """Return the trajectory check file name."""
        return self._checkFile

    def getTimeArr(self) -> npt.NDArray[np.float64]:
        """Return the trajectory time instants."""
        times = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            times[k] = self._snaps[k].time
        return times

    def getScoreArr(self) -> npt.NDArray[np.float64]:
        """Return the trajectory scores."""
        scores = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            scores[k] = self._snaps[k].score
        return scores

    def getNoiseArr(self) -> npt.NDArray[Any]:
        """Return the trajectory noises."""
        noises = np.zeros(len(self._snaps), dtype=type(self._snaps[0].noise))
        for k in range(len(self._snaps)):
            noises[k] = self._snaps[k].noise
        return noises

    def getLength(self) -> int:
        """Return the trajectory length."""
        return len(self._snaps)

    def getLastState(self) -> Any:
        """Return the last state in the trajectory."""
        for snap in reversed(self._snaps):
            if snap.hasState():
                return snap.state
