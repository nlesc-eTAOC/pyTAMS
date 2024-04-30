from __future__ import annotations
import os
import time
import xml.etree.ElementTree as ET
from typing import Any
from typing import Optional
import numpy as np
import numpy.typing as npt
from pytams.xmlutils import dict_to_xml
from pytams.xmlutils import make_xml_snapshot
from pytams.xmlutils import new_element
from pytams.xmlutils import read_xml_snapshot
from pytams.xmlutils import xml_to_dict


class WallTimeLimit(Exception):
    """Exception for running into wall time limit."""

    pass


class Trajectory:
    """A class defining a stochastic trajectory.

    Includes an instance of the forward model, current and end times, and
    a list of the model snapshots for each new maximum of the
    score function along the way.
    """

    def __init__(self,
                 fmodel_t: Any,
                 parameters: dict,
                 trajId: str) -> None:
        """Create a trajectory.

        Args:
            fmodel_t: the forward model type
            parameters: a dictionary of input parameters
            trajId: a string for the trajectory id
        """
        self._fmodel = fmodel_t(parameters, trajId)
        self._parameters = parameters

        self._t_cur : float = 0.0
        self._t_end : float = parameters.get("trajectory",{}).get("end_time", 1.0)
        self._dt : float = parameters.get("trajectory",{}).get("step_size", 0.1)
        self._stoichForcingAmpl = parameters.get("trajectory",{}).get("stoichforcing", 0.5)
        self._convergedVal = parameters.get("trajectory",{}).get("targetscore", 0.95)

        # List of new maximums
        self._time : list[float] = []
        self._state : list[Any] = []
        self._score : list[float] = []
        self._noise : list [Any] = []

        self._score_max = 0.0

        self._tid = trajId
        self._checkFile = "{}.xml".format(trajId)

        self._has_ended = False
        self._has_converged = False

    def setCheckFile(self, file: str) -> None:
        """Setter of the trajectory checkFile."""
        self._checkFile = file

    def id(self) -> str:
        """Return trajectory Id."""
        return self._tid

    def advance(self, t_end: float = 1.0e12, walltime: float = 1.0e12) -> None:
        """Advance the trajectory to a prescribed end time.

        Args:
            t_end: the end time of the advance
            walltime: a walltime limit to advance the model to t_end
        """
        startTime = time.monotonic()
        remainingTime = walltime - time.monotonic() + startTime
        end_time = min(t_end, self._t_end)

        while (
            self._t_cur < end_time
            and not self._has_converged
            and remainingTime >= 0.05 * walltime
        ):
            dt = self._fmodel.advance(self._dt, self._stoichForcingAmpl)
            self._t_cur = self._t_cur + dt
            score = self._fmodel.score()
            self._time.append(self._t_cur)
            self._state.append(self._fmodel.getCurState())
            self._score.append(score)
            self._noise.append(self._fmodel.noise())
            if score > self._score_max:
                self._score_max = score

            if score >= self._convergedVal:
                self._has_converged = True

            remainingTime = walltime - time.monotonic() + startTime

        if self._t_cur >= self._t_end or self._has_converged:
            self._has_ended = True

        if self._has_ended:
            self._fmodel.clear()

        if remainingTime < 0.05 * walltime:
            raise WallTimeLimit("{} ran out of time in advance()".format(self._tid))


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
                restTraj._time.append(time)
                restTraj._score.append(score)
                restTraj._noise.append(noise)
                restTraj._state.append(state)

        return restTraj

    @classmethod
    def restartFromTraj(
        cls,
        traj: Trajectory,
        rstId: str,
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
        if not traj._score:
            restTraj = Trajectory(
                fmodel_t=type(traj._fmodel),
                parameters=traj._parameters,
                trajId=rstId,
            )
            return restTraj

        high_score_idx = 0
        while traj._score[high_score_idx] < score:
            high_score_idx += 1

        restTraj = Trajectory(
            fmodel_t=type(traj._fmodel), parameters=traj._parameters, trajId=rstId
        )
        for k in range(high_score_idx + 1):
            restTraj._score.append(traj._score[k])
            restTraj._time.append(traj._time[k])
            restTraj._noise.append(traj._noise[k])
            restTraj._state.append(traj._state[k])

        restTraj._fmodel.setCurState(restTraj._state[-1])
        restTraj._t_cur = restTraj._time[-1]
        restTraj._score_max = restTraj._score[-1]

        return restTraj

    def store(self, traj_file: Optional[str] = None) -> None:
        """Store the trajectory to an XML chkfile."""
        root = ET.Element(self._tid)
        root.append(dict_to_xml("params", self._parameters["trajectory"]))
        mdata = ET.SubElement(root, "metadata")
        mdata.append(new_element("id", self._tid))
        mdata.append(new_element("t_cur", self._t_cur))
        mdata.append(new_element("t_end", self._t_end))
        mdata.append(new_element("dt", self._dt))
        mdata.append(new_element("score_max", self._score_max))
        mdata.append(new_element("ended", self._has_ended))
        mdata.append(new_element("converged", self._has_converged))
        snaps = ET.SubElement(root, "snapshots")
        for k in range(len(self._score)):
            snaps.append(
                make_xml_snapshot(k,
                                  self._time[k],
                                  self._score[k],
                                  self._noise[k],
                                  self._state[k])
            )
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)
        if traj_file is not None:
            tree.write(traj_file)
        else:
            tree.write(self._checkFile)

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
        return np.array(self._time)

    def getScoreArr(self) -> npt.NDArray[np.float64]:
        """Return the trajectory scores."""
        return np.array(self._score)

    def getNoiseArr(self) -> npt.NDArray[Any]:
        """Return the trajectory noises."""
        return np.array(self._noise)

    def getLength(self) -> int:
        """Return the trajectory length."""
        return len(self._time)
