class Trajectory:
    """A class defining a stochastic trajectory.

    Includes an instance of the forward model, current and end times, and
    a list of the model snapshots for each new maximum of the
    score function along the way.
    """

    def __init__(self, fmodel, parameters, trajId: str = None) -> None:
        """Create a trajectory.

        Args:
            fmodel: the forward model
            parameters: a dictionary of input parameters
            trajId: a string for the trajectory id
        """
        self._fmodel = fmodel
        self._parameters = parameters

        self._t_cur = 0.0
        self._t_end = self._parameters.get("traj.end_time", 1.0)
        self._dt = self._parameters.get("traj.step_size", 0.1)

        # List of new maximums
        self._time = []
        self._state = []
        self._score = []

        self._score_max = 0.0

        self._tid = trajId

        self.has_ended = False
        self.has_converged = False

    def id(self) -> str:
        """Return trajectory Id."""
        return self._tid

    def advance(self, t_end: float = 1.0e12):
        """Advance the trajectory to a prescribed end time.

        Args:
            t_end: the end time of the advance

        Returns:
            self: return the updated trajectory
        """
        end_time = min(t_end, self._t_end)
        stoichForcingAmpl = self._parameters.get("traj.stoichForcing", 0.5)
        convergedVal = self._parameters.get("traj.targetScore", 0.95)

        while self._t_cur <= end_time and ~self.has_converged:
            self._t_cur = self._t_cur + self._dt
            self._fmodel.advance(self._dt, stoichForcingAmpl)
            score = self._fmodel.score()
            if score > self._score_max:
                self._time.append(self._t_cur)
                self._state.append(self._fmodel.getCurState())
                self._score.append(score)
                self._score_max = score

            if score >= convergedVal:
                self.has_converged = True

        if self._t_cur >= self._t_end or self.has_converged:
            self.has_ended = True

    @classmethod
    def restoreFromChk(
        cls,
        chkPoint,
    ):
        """TODO."""
        pass

    def printT(self):
        """Dump the trajectory to screen."""
        print("\n Trajectory: {} \n".format(self._tid))
        for k in range(len(self._time)):
            print("{} {} {}".format(self._time[k], self._score[k], self._state[k]))
        if self.has_converged:
            print(" Success")

    @classmethod
    def restartFromTraj(
        cls,
        traj,
        score: float,
    ):
        """Create a new trajectory.

        Loading the beginning of a provided trajectory
        for all entries with score below a given score

        Args:
            traj: an already existing trajectory
            score: a threshold score
        """
        # Check for empty trajectory
        if not traj._score:
            restTraj = Trajectory(fmodel=traj._fmodel, parameters=traj._parameters)
            return restTraj

        high_score_idx = 0
        while traj._score[high_score_idx] < score:
            high_score_idx += 1

        restTraj = Trajectory(fmodel=traj._fmodel, parameters=traj._parameters)
        for k in range(high_score_idx + 1):
            restTraj._score.append(traj._score[k])
            restTraj._time.append(traj._time[k])
            restTraj._state.append(traj._state[k])

        restTraj._fmodel.setCurState(restTraj._state[-1])
        restTraj._t_cur = restTraj._time[-1]
        restTraj._score_max = restTraj._score[-1]

        return restTraj

    def store(self, traj_id, traj_file):
        """Store the trajectory to disk."""
        pass

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
        return self.has_converged

