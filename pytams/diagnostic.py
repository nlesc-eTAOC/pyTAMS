"""Diagnostic class for pyREVS."""

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
from pytams.config import Config
from pytams.snapshot import Snapshot
from pytams.sqldb import DiagDB

_logger = logging.getLogger(__name__)


class DiagnosticPlugin:
    """A base class for diagnostic plugins.

    Plugins are attached to the trajectory objects.

    Attributes:
        _label: the diagnostic label
        _tid: the ID of the trajectory the plugin is attached to
        _weight: the weight of the trajectory
    """

    _label: str = ""
    _tid: int = -1
    _weight: float = 0.0

    def __init__(
        self,
        dlabel: str,
        params: dict[Any, Any],
        tid: int,
        weight: float,
        workdir: Path,
        fprocess: Callable[..., Any],
        ddb: DiagDB,
    ) -> None:
        self._label = dlabel
        self._tid = tid
        self._weight = weight
        self._params = params
        self._process = fprocess
        self._workdir = workdir
        self._ddb = ddb

    def get_crossed_levels(self, new_snapshot: Snapshot) -> list[float]:
        """Test to know if diagnostic is needed."""
        raise NotImplementedError

    def update(self, old_snapshot: Snapshot, new_snapshot: Snapshot) -> None:
        """Standard entry point called after every MCMC step."""
        crossed = self.get_crossed_levels(new_snapshot)

        for level in crossed:
            # Get the model to provide what to add to the database
            ldata = self._process(self._label, self._tid, level, old_snapshot, new_snapshot)

            # 2. Record the snapshot to the SQL database
            # We assume new_snapshot has the state we want to preserve
            self._ddb.add_diagnostic_entry(
                diaglabel=self._label,
                traj_id=self._tid,
                level=level,
                time=new_snapshot.time,
                weight=self._weight,
                ldata=pickle.dumps(ldata),  # This should be bytes/pickle
            )


class FirstTimeCrossingDiagnostic(DiagnosticPlugin):
    """Triggers when the score function crosses pre-defined levels.

    This is a central diagnostic plugin to TAMS, triggered when
    the score function crosses pre-defined levels, only for
    the first time.

    This allows to evaluate the probability of crossing any
    intermediate score levels from a TAMS run, as well as
    estimating mean first passage time.

    Attributes:
        _levels: the threshold levels
        _highest_recorded_score: the high water mark of the plugin
        _checked_db: True if the DB has been checked
    """

    def __init__(
        self,
        dlabel: str,
        params: dict[Any, Any],
        tid: int,
        tweight: float,
        workdir: Path,
        fprocess: Callable[..., Any],
        ddb: DiagDB,
    ) -> None:
        super().__init__(dlabel, params, tid, tweight, workdir, fprocess, ddb)
        s_min = self._params.get("score_min", 0.0)
        s_max = self._params.get("score_max", 1.0)
        n_levels = self._params.get("n_levels", 10)

        # Create the threshold levels
        # and keep track of level cleared (diag only once per level)
        self._levels: npt.NDArray[np.float64] = np.linspace(s_min, s_max, n_levels)

        # Watermark to ensure we only record 'first contact' going up
        # We will check the DB when the diagnostic is first called
        self._highest_recorded_score = -np.inf
        self._checked_db = False

    def get_crossed_levels(self, new_snapshot: Snapshot) -> list[float]:
        """Get the list of level crossed during last step.

        Args:
            new_snapshot: the new (end of the time step) snapshot

        Returns:
            the list of score levels crossed
        """
        if not self._checked_db:
            # Query the DB to resume the watermark
            self._highest_recorded_score = self._ddb.get_highest_recorded_level(self._tid, self._label)
            self._checked_db = True

        s_new = new_snapshot.score

        # We only trigger if the new score is higher than anything we've recorded
        if s_new <= self._highest_recorded_score:
            return []

        # Find levels that are:
        # 1. Between current highest recorded and the new score
        # 2. Specifically higher than the previous recorded watermark
        mask = (self._levels <= s_new) & (self._levels > self._highest_recorded_score)
        crossed: list[float] = self._levels[mask].tolist()

        if crossed:
            # Update the watermark to the highest level we just triggered
            self._highest_recorded_score = crossed[-1]

        return crossed


def diagnosticfactory(
    configs: dict[str, Config], tid: int, tweight: float, workdir: Path, fprocess: Callable[..., Any], ddb: DiagDB
) -> list[DiagnosticPlugin]:
    """Parse input parameters to generate a list of DiagnosticPlugin.

    Args:
        configs: a dict with a Config object for each diagnostic
        tid: the ID of the traj the diagnostic is attached to
        tweight: the weight of the traj
        workdir: the workdir associated with a trajectory
        fprocess: the forward model diagnostic function
        ddb: the diagnostic database to add the data to
    """
    diags_l: list[DiagnosticPlugin] = []

    for k, v_cfg in configs.items():
        v = v_cfg.as_dict()
        if v is None:
            err_msg = f"Diagnostic {k} is missing a parameter dict !"
            raise RuntimeError(err_msg)

        diag_type = v.get("type", "FirstCrossing")

        if diag_type == "FirstCrossing":
            diags_l.append(FirstTimeCrossingDiagnostic(k, v, tid, tweight, workdir, fprocess, ddb))
            dbg_msg = f"Created FirstCrossing Diagnostic {k} for traj {tid}"
            _logger.debug(dbg_msg)
        else:
            err_msg = f"Diagnostic {k} has unknown trigger type {diag_type} !"
            raise ValueError(err_msg)

    return diags_l


class DiagnosticAnalyst:
    """A class to handle analysing diagnostic statistics.

    Let's keep the analysis logic separated from the gathering logic.
    This class retrieves data from the diagnostic database and perform
    some computation (mostly conditional statistics on score iso-levels).
    """

    def __init__(self, db_path: str) -> None:
        self.db = DiagDB(db_path)

    def get_diagnostic_data(self, label: str) -> dict[float, list[tuple[Any, float, float, int]]]:
        """A user-facing access to the diag DB.

        An alias to the DB access for the analyst.

        Returns:
            A dictionary mapping each score iso-level (float) to a list of tuples.
            Each tuple contains (unpickled_data, trajectory_weight, time, tid).
        """
        return self.db.get_diagnostic_data(label)

    def compute_weighted_stats(self, label: str) -> dict[float, dict[str, Any]]:
        """Aggregate data and compute mean/variance per level."""
        # Use a generator to fetch data level by level to save memory
        data_map = self.db.get_diagnostic_data(label)

        stats_per_level = {}
        full_sum = -1.0
        for level, tdata in data_map.items():
            data = np.array([td[0] for td in tdata])
            weights = np.array([td[1] for td in tdata])
            times = np.array([td[2] for td in tdata])

            sum_w = np.sum(weights)
            if level <= 0.0:
                full_sum = sum_w

            if sum_w == 0:
                continue

            proba = weights / full_sum

            mean_time = np.average(times, weights=proba)
            weighted_mean = np.average(data, weights=proba, axis=0)
            delta = data - weighted_mean
            weighted_variance = np.average(delta**2, weights=proba, axis=0)

            stats_per_level[level] = {
                "mean": weighted_mean,
                "var": weighted_variance,
                "mean_time": mean_time,
                "count": len(tdata),
                "total_weight": np.sum(proba),
            }

        return stats_per_level

    def get_conditional_means(self, label: str) -> dict[float, dict[str, Any]]:
        """Compute means at level L_i conditioned on reaching or not reaching L_{i+1}.

        Returns:
            { level: {"reached_next": mean_val, "failed_next": mean_val} }
        """
        # 1. Get all raw data for this label
        raw_data = self.db.get_diagnostic_data(label)
        levels = sorted(raw_data.keys())

        results = {}

        for i in range(len(levels) - 1):
            current_level = levels[i]
            next_level = levels[i + 1]

            # Find IDs that exist at the next level
            successful_ids = {entry[3] for entry in raw_data[next_level]}

            reached_vals = []
            reached_weights = []
            failed_vals = []
            failed_weights = []

            for entry in raw_data[current_level]:
                val, w, tid = entry[0], entry[1], entry[3]

                if tid in successful_ids:
                    reached_vals.append(val)
                    reached_weights.append(w)
                else:
                    failed_vals.append(val)
                    failed_weights.append(w)

            results[current_level] = {
                "reached_next": np.average(reached_vals, weights=reached_weights, axis=0) if reached_vals else None,
                "failed_next": np.average(failed_vals, weights=failed_weights, axis=0) if failed_vals else None,
            }

        return results
