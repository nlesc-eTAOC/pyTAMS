from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional
from typing import final


class ForwardModelError(Exception):
    """Exception class for the forward model."""

    pass


class BaseClassCallError(ForwardModelError):
    """BaseClass ForwardModel method called !"""

    pass


class AdvanceError(ForwardModelError):
    """Concrete ForwardModel _advance error !"""

    pass


class ForwardModelBaseClass(metaclass=ABCMeta):
    """A base class for the stochastic forward model.

    Implement the core methods required of the forward
    model within the TAMS context. Exception are thrown
    if required functions are not overritten in actual model.
    """


    @final
    def __init__(self,
                 params: dict,
                 ioprefix: Optional[str] = None):
        """Base class __init__ method.

        The base forwardmodel class handles some components
        needed by TAMS regardless of the model.

        Args:
            params: an optional dict containing parameters
            ioprefix: an optional string defining run folder
        """
        # Initialize common tooling
        self._prescribed_noise : bool = False
        self._noise : Any = None
        self._step : int = 0
        self._time : float = 0.0

        # Add the deterministic parameter to the model dictionary
        # for clarity
        if params.get("model", None):
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
                dt: float,
                forcingAmpl: float) -> float:
        """Base class advance function of the model.

        The base class advance function update the internal
        step counter and manage the generation (or reuse)
        of the stochastic noise.
        It also handles exceptions.

        Args:
            dt: the time step size over which to advance
            forcingAmpl: stochastic multiplicator
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        # Get noise for the next model step
        if not self._prescribed_noise:
            self._noise = self._make_noise()

        try:
            actual_dt = self._advance(self._step,
                                      self._time,
                                      dt,
                                      self._noise,
                                      forcingAmpl)
            # Update internal counter. Note that actual_dt may differ
            # from requested dt in some occasions.
            self._step = self._step + 1
            self._time = self._time + actual_dt
        except AdvanceError:
            raise AdvanceError("Damn it !")

        # After a step, always reset flag to false
        self._prescribed_noise = False

        return actual_dt

    @final
    def getNoise(self) -> Any:
        """Return the model's latest noise increment."""
        return self._noise

    @final
    def setNoise(self, a_noise : Any) -> None:
        """Set the model's next noise increment."""
        self._prescribed_noise = True
        self._noise = a_noise

    @final
    def clear(self) -> None:
        """Destroy internal data."""
        self._clear_model()

    @abstractmethod
    def _init_model(self,
                    params: Optional[dict] = None,
                    ioprefix: Optional[str] = None) -> None:
        """Concrete class specific initialization.

        Args:
            params: an optional dict containing parameters
            ioprefix: an optional string defining run folder (TOCHECK)
        """
        pass

    @abstractmethod
    def _advance(self,
                 step: int,
                 time: float,
                 dt: float,
                 noise: Any,
                 forcingAmpl: float) -> float:
        """Concrete class advance function.

        Args:
            step: the current step counter
            time: the starting time of the advance call
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            forcingAmpl: stochastic multiplicator
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        raise BaseClassCallError("Calling base class _advance() method !")

    @abstractmethod
    def getCurState(self) -> Any:
        """Return the current state of the model."""
        raise BaseClassCallError("Calling base class getCurState method !")

    @abstractmethod
    def setCurState(self, state : Any) -> Any:
        """Set the current state of the model."""
        raise BaseClassCallError("Calling base class setCurState method !")

    @abstractmethod
    def score(self) -> Any:
        """Return the model's current state score."""
        raise BaseClassCallError("Calling base class score method !")

    @abstractmethod
    def _make_noise(self) -> Any:
        """Return the model's latest noise increment."""
        raise BaseClassCallError("Calling base class _make_noise method !")

    def _clear_model(self) -> Any:
        """Clear the concrete forward model internals."""
        pass

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "BaseClassForwardModel"
