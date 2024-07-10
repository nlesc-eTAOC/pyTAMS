from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import Optional


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


    def __init__(self,
                 params: Optional[dict] = None,
                 ioprefix: Optional[str] = None):
        """Base class __init__ method.

        The base forwardmodel class handles some components
        needed by TAMS regardless of the model.

        Args:
            params: an optional dict containing parameters
            ioprefix: an optional string defining run folder (TOCHECK)
        """
        self._init_model(params, ioprefix)
        self._prescribed_noise : bool = False
        self._noise : Any = None
        self._step : int = 0

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
            self._noise = self._get_noise()

        try:
            actual_dt = self._advance(self._step, dt, self._noise, forcingAmpl)
        except AdvanceError:
            raise AdvanceError("Damn it !")

        # After a step, always reset flag to false
        self._prescribed_noise = False

        return actual_dt

    @abstractmethod
    def _advance(self,
                 step: int,
                 dt: float,
                 noise: Any,
                 forcingAmpl: float) -> float:
        """Concrete class advance function.

        Args:
            step: the current step counter
            dt: the time step size over which to advance
            noise: the noise to be used in the model step
            forcingAmpl: stochastic multiplicator
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        raise BaseClassCallError("Calling base class _advance() method !")


    def getCurState(self) -> Any:
        """Return the current state of the model."""
        raise BaseClassCallError("Calling getCurState() method !")

    def setCurState(self, state : Any) -> Any:
        """Set the current state of the model."""
        raise BaseClassCallError("Calling setCurState method !")

    def score(self) -> Any:
        """Return the model's current state score."""
        raise BaseClassCallError("Calling score method !")

    def getNoise(self) -> Any:
        """Return the model's latest noise increment."""
        return self._noise

    @abstractmethod
    def _get_noise(self) -> Any:
        """Return the model's latest noise increment."""
        raise BaseClassCallError("Calling getNoise method !")

    def setNoise(self, a_noise : Any) -> None:
        """Set the model's next noise increment."""
        self._prescribed_noise = True
        self._noise = a_noise

    def clear(self) -> None:
        """Destroy internal data."""
        pass

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "BaseClassForwardModel"
