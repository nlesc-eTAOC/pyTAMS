from typing import Any
from typing import Optional


class ForwardModelError(Exception):
    """Exception class for the forward model."""

    pass


class TemplateCallError(ForwardModelError):
    """Template ForwardModel method called !"""

    pass


class ForwardModel:
    """A template class for the stochastic forward model.

    Implement the core methods required of the forward
    model within the TAMS context. Exception are thrown
    if those functions are not overritten in actual model.
    """

    def __init__(self,
                 params: Optional[dict] = None,
                 ioprefix: Optional[str] = None):
        """Might need something here."""
        pass

    def advance(self, dt: float, forcingAmpl: float) -> float:
        """Advance function of the model.

        Args:
            dt: the time step size over which to advance
            forcingAmpl: stochastic multiplicator
        Return:
            Some model will not do exactly dt (e.g. sub-stepping) return the actual dt
        """
        raise TemplateCallError("Calling advance() method !")

    def getCurState(self) -> Any:
        """Return the current state of the model."""
        raise TemplateCallError("Calling getCurState() method !")

    def setCurState(self, state : Any) -> Any:
        """Set the current state of the model."""
        raise TemplateCallError("Calling setCurState method !")

    def score(self) -> Any:
        """Return the model's current state score."""
        raise TemplateCallError("Calling score method !")

    def noise(self) -> Any:
        """Return the model's latest noise increment."""
        raise TemplateCallError("Calling noise method !")

    def clear(self) -> None:
        """Destroy internal data."""
        pass

    @classmethod
    def name(cls) -> str:
        """Return a the model name."""
        return "TemplateForwardModel"
