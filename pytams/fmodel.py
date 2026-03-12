"""A base class for the stochastic forward model."""

import copy
from abc import ABC
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import cast
from typing import final

_logger = getLogger(__name__)

# Define Generics for Noise and State
T_Noise = TypeVar("T_Noise")
T_State = TypeVar("T_State")


class ForwardModelBaseClass(ABC, Generic[T_Noise, T_State]):
    """A base class for the stochastic forward model.

    pyTAMS relies on a separation of the stochastic model,
    encapsulating the physics of interest, and the TAMS
    algorithm itself. The ForwardModelBaseClass defines
    the API the TAMS algorithm requires from the stochastic
    model.

    Concrete model classes must implement all the abstract
    functions defined in this base class.

    The base class handles some components needed by TAMS,
    so that the user does not have to ensure compatibility
    with TAMS requirements.

    Attributes:
        _noise: the noise to be used in the next model step
        _step: the current stochastic step counter
        _time: the current stochastic time
        _workdir: the working directory
    """

    _noise: T_Noise
    _step: int = 0
    _time: float = 0.0

    @final
    def __init__(self, a_id: int, params: dict[Any, Any], workdir: Path | None = None):
        """Base class __init__ method.

        The ABC init method calls the concrete class init method
        while performing some common initializations. Additionally
        this method create/append to a model dictionary to the
        parameter dictionary to ensure the 'deterministic' parameter
        is always available in the model dictionary.

        Upon initializing the model, a first call to make_noise
        is made to ensure the proper type is generated.

        Args:
            a_id: an int providing a unique id to the model instance
            params: a dict containing parameters
            workdir: an optional path to the working directory
        """
        # Initialize common tooling
        self._id = a_id
        self._step: int = 0
        self._time: float = 0.0
        self._workdir: Path = Path.cwd() if workdir is None else workdir

        # Add the deterministic parameter to the model dictionary
        lparams = copy.deepcopy(params)
        tams_conf = lparams.get("tams", {})
        model_conf = lparams.setdefault("model", {})
        model_conf["deterministic"] = tams_conf.get("deterministic", False)

        # Call the concrete class init method
        self._init_model(a_id, lparams)

        # Initialize property with type casting for mypy.
        self._noise = cast("T_Noise", None)

    @final
    def advance(self, dt: float, need_end_state: bool) -> float:
        """Base class advance function of the model.

        This is the advance function called by TAMS internals. It
        handles updating the model time and step counter, as well as
        reusing or generating noise only when needed.
        It also handles exceptions.

        Args:
            dt: the time step size over which to advance
            need_end_state: whether the step end state is needed

        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        try:
            actual_dt = self._advance(self._step, self._time, dt, self._noise, need_end_state)
            # Update internal counter. Note that actual_dt may differ
            # from requested dt in some occasions.
            self._step = self._step + 1
            self._time = self._time + actual_dt
        except Exception:
            err_msg = "Advance function ran into an error !"
            _logger.exception(err_msg)
            raise

        return actual_dt

    @final  # type: ignore[misc]
    @property
    def noise(self) -> T_Noise:
        """Return the model's latest noise increment."""
        if self._noise is None:
            self._noise = self.make_noise()
        return self._noise

    @noise.setter
    @final  # type: ignore[misc]
    def noise(self, a_noise: T_Noise) -> None:
        """Set the model's next noise increment."""
        self._noise = a_noise

    @final
    def clear(self) -> None:
        """Destroy internal data."""
        self._clear_model()

    @final
    def set_workdir(self, workdir: Path) -> None:
        """Setter of the model working directory.

        Args:
            workdir: the new working directory
        """
        self._workdir = workdir

    @abstractmethod
    def _init_model(self, m_id: int, params: dict[Any, Any]) -> None:
        """Concrete class specific initialization.

        Args:
            m_id: the model instance unique identifier
            params: an optional dict containing parameters
        """

    @abstractmethod
    def _advance(self, step: int, time: float, dt: float, noise: T_Noise, need_end_state: bool) -> float:
        """Concrete class advance function.

        This is the model-specific advance function.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            need_end_state: whether the step end state is needed
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """

    @abstractmethod
    def get_current_state(self) -> T_State:
        """Return the current state of the model.

        Note that the return type is left to the concrete model definition.
        """

    @abstractmethod
    def set_current_state(self, state: T_State) -> None:
        """Set the current state of the model.

        Args:
            state: the externally provide state
        """

    @abstractmethod
    def score(self) -> float:
        """Return the model's current state score.

        The score is a real.

        Returns:
            the score associated with the current model state
        """

    @abstractmethod
    def make_noise(self) -> T_Noise:
        """Return the model's latest noise increment.

        Note that the noise type is left to the concrete model definition.

        Returns:
            The model next noise increment
        """

    @final
    def post_trajectory_branching_hook(self, step: int, time: float) -> None:
        """Model post trajectory branching hook.

        Args:
            step: the current step counter
            time: the time of the simulation
        """
        self._step = step
        self._time = time
        self._trajectory_branching_hook()

    def _trajectory_branching_hook(self) -> None:
        """Model-specific post trajectory branching hook."""

    @final
    def post_trajectory_restore_hook(self, step: int, time: float) -> None:
        """Model post trajectory restore hook.

        Args:
            step: the current step counter
            time: the time of the simulation
        """
        self._step = step
        self._time = time
        self._trajectory_restore_hook()

    def _trajectory_restore_hook(self) -> None:
        """Model-specific post trajectory restore hook."""

    def check_convergence(self, step: int, time: float, current_score: float, target_score: float) -> bool:
        """Check if the model has converged.

        This default implementation checks if the current score is
        greater than or equal to the target score. The user can override
        this method to implement a different convergence criterion.

        Args:
            step: the current step counter
            time: the time of the simulation
            current_score: the current score
            target_score: the target score
        """
        _ = (step, time)
        return current_score >= target_score

    def _clear_model(self) -> Any:
        """Clear the concrete forward model internals."""

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "BaseClassForwardModel"
