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

class ForwardModelAdvance(Exception):
    """Exception for forward model advance fail."""

    pass

class Snapshot:
    """A class defining a snapshot.

    Gathering what defines a snapshot into an object
    """
    def __init__(self,
                 time: float,
                 score: float,
                 noise: Any,
                 state: Optional[Any] = None) -> None:
        """Create a snapshot.
        """
        self._time = time
        self._score = score
        self._noise = noise
        if state is not None:
            self._state = state

    def Time(self) -> float:
        """Return the snapshot time."""
        return self._time
    def Score(self) -> float:
        """Return the snapshot score."""
        return self._score
    def Noise(self) -> Any:
        """Return the snapshot noise."""
        return self._noise
    def State(self) -> Any:
        """Return the snapshot state."""
        return self._state


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

        # List of snapshots
        self._snaps : list[Snapshot] = []

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
            try:
                dt = self._fmodel.advance(self._dt, self._stoichForcingAmpl)
            except:
                raise ForwardModelAdvance("Error advancing fmodel from {} with dt {}".format(self._t_cur,self._dt))
            self._t_cur = self._t_cur + dt
            score = self._fmodel.score()
            self._snaps.append(Snapshot(self._t_cur,
                                        score,
                                        self._fmodel.noise(),
                                        self._fmodel.getCurState()
                                        )
                               )
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
                restTraj._snaps.append(Snapshot(time, score, noise, state))

        restTraj._fmodel.setCurState(restTraj._snaps[-1].State())

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
        if len(traj._snaps) == 0:
            restTraj = Trajectory(
                fmodel_t=type(traj._fmodel),
                parameters=traj._parameters,
                trajId=rstId,
            )
            return restTraj

        high_score_idx = 0
        while traj._snaps[high_score_idx].Score() < score:
            high_score_idx += 1

        restTraj = Trajectory(
            fmodel_t=type(traj._fmodel), parameters=traj._parameters, trajId=rstId
        )
        for k in range(high_score_idx + 1):
            restTraj._snaps.append(traj._snaps[k])

        restTraj._fmodel.setCurState(restTraj._snaps[-1].State())
        restTraj._t_cur = restTraj._snaps[-1].Time()
        restTraj._score_max = restTraj._snaps[-1].Score()

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
        snaps_xml = ET.SubElement(root, "snapshots")
        for k in range(len(self._snaps)):
            snaps_xml.append(
                make_xml_snapshot(k,
                                  self._snaps[k].Time(),
                                  self._snaps[k].Score(),
                                  self._snaps[k].Noise(),
                                  self._snaps[k].State())
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
        times = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            times[k] = self._snaps[k].Time()
        return times

    def getScoreArr(self) -> npt.NDArray[np.float64]:
        """Return the trajectory scores."""
        scores = np.zeros(len(self._snaps))
        for k in range(len(self._snaps)):
            scores[k] = self._snaps[k].Score()
        return scores

    def getNoiseArr(self) -> npt.NDArray[Any]:
        """Return the trajectory noises."""
        noises = np.zeros(len(self._snaps), dtype=type(self._snaps[0].Noise))
        for k in range(len(self._snaps)):
            noises[k] = self._snaps[k].Noise()
        return noises

    def getLength(self) -> int:
        """Return the trajectory length."""
        return len(self._snaps)
