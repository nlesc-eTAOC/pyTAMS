"""A base class for the stochastic forward model."""
from abc import ABCMeta
from abc import abstractmethod
from logging import getLogger
from pathlib import Path
from typing import Any
from typing import final

_logger = getLogger(__name__)

class ForwardModelBaseClass(metaclass=ABCMeta):
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
        _prescribed_noise: whether the noise is provided or need to be generated
        _noise: the noise to be used in the next model step
        _step: the current stochastic step counter
        _time: the current stochastic time
        _workdir: the working directory
    """

    @final
    def __init__(self,
                 params: dict[Any, Any],
                 ioprefix: str | None = None,
                 workdir: Path | None = None):
        """Base class __init__ method.

        The ABC init method calls the concrete class init method
        while performing some common initializations. Additionally,
        this method create/append to a model dictionary to the
        parameter dictionary to ensure the 'deterministic' parameter
        is always available in the model dictionary.

        Upon initializing the model, a first call to _make_noise
        is made to ensure the proper type is generated.

        Args:
            params: a dict containing parameters
            ioprefix: an optional string defining run folder
            workdir: an optional path to the working directory
        """
        # Initialize common tooling
        self._prescribed_noise : bool = False
        self._noise : Any = None
        self._step : int = 0
        self._time : float = 0.0
        self._workdir : Path = Path.cwd() if workdir is None else workdir

        # Add the deterministic parameter to the model dictionary
        # for consistency
        if params.get("model"):
            params["model"]["deterministic"] = params.get("tams", {}).get("deterministic", False)
        else:
            params["model"] = {"deterministic": params.get("tams", {}).get("deterministic", False)}

        # Call the concrete class init method
        self._init_model(params, ioprefix)

        # Generate the first noise increment
        # to at least get the proper type.
        self._noise = self._make_noise()

    @final
    def advance(self,
                dt: float) -> float:
        """Base class advance function of the model.

        This is the advance function called by TAMS internals. It
        handles updating the model time and step counter, as well as
        reusing or generating noise only when needed.
        It also handles exceptions.

        Args:
            dt: the time step size over which to advance

        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        # Get noise for the next model step
        # only if the noise has not been prescribed by
        # other mechanism already (rewinding trajectory for example).
        if not self._prescribed_noise:
            self._noise = self._make_noise()

        try:
            actual_dt = self._advance(self._step,
                                      self._time,
                                      dt,
                                      self._noise)
            # Update internal counter. Note that actual_dt may differ
            # from requested dt in some occasions.
            self._step = self._step + 1
            self._time = self._time + actual_dt
        except Exception:
            err_msg = "Advance function ran into an error !"
            _logger.exception(err_msg)
            raise

        # After a step, always reset flag to false
        self._prescribed_noise = False

        return actual_dt

    @final
    def get_noise(self) -> Any:
        """Return the model's latest noise increment."""
        return self._noise

    @final
    def set_noise(self, a_noise : Any) -> None:
        """Set the model's next noise increment."""
        self._prescribed_noise = True
        self._noise = a_noise

    @final
    def clear(self) -> None:
        """Destroy internal data."""
        self._clear_model()

    @final
    def set_workdir(self, workdir : Path) -> None:
        """Setter of the model working directory.

        Args:
            workdir: the new working directory
        """
        self._workdir = workdir

    @abstractmethod
    def _init_model(self,
                    params: dict[Any, Any] | None = None,
                    ioprefix: str | None = None) -> None:
        """Concrete class specific initialization.

        Args:
            params: an optional dict containing parameters
            ioprefix: an optional string defining run folder
        """

    @abstractmethod
    def _advance(self,
                 step: int,
                 time: float,
                 dt: float,
                 noise: Any) -> float:
        """Concrete class advance function.

        This is the model-specific advance function.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """

    @abstractmethod
    def get_current_state(self) -> Any:
        """Return the current state of the model."""

    @abstractmethod
    def set_current_state(self, state : Any) -> Any:
        """Set the current state of the model."""

    @abstractmethod
    def score(self) -> float:
        """Return the model's current state score."""

    @abstractmethod
    def _make_noise(self) -> Any:
        """Return the model's latest noise increment."""

    @final
    def post_trajectory_restart_hook(self,
                                     step : int,
                                     time : float) -> None:
        """Model post trajectory restart hook.

        Args:
            step: the current step counter
            time: the time of the simulation
        """
        self._step = step
        self._time = time
        self._trajectory_restart_hook()

    def _trajectory_restart_hook(self) -> None:
        """Model-specific post trajectory restart hook."""

    @final
    def post_trajectory_restore_hook(self,
                                     step : int,
                                     time : float) -> None:
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

    def check_convergence(self,
                          step: int,
                          time: float,
                          current_score: float,
                          target_score: float) -> bool:
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
