class ForwardModelError(Exception):
    """Exception class for the forward model.
    """
    pass


class TemplateCallError(ForwardModelError):
    """Template ForwardModel method called !"""
    pass


class ForwardModel:
    """A template class for the stochastic forward model.

    Implement the core methods reauired of the forward
    model within the TAMS context. Exception are thrown
    if those functions are not overritten in actual model.
    """

    def __init__(self):
        """Might need something here."""
        pass

    def advance(self, dt: float, forcingampl: float):
        """Advance function of the model.

        Args:
            dt: the time step size over which to advance
            forcingAmpl: stochastic multiplicator
        """
        raise TemplateCallError("Calling advance() method !")

    def getCurState(self):
        """Return the current state of the model."""
        raise TemplateCallError("Calling getCurState() method !")

    def setCurState(self, state):
        """Set the current state of the model."""
        raise TemplateCallError("Calling setCurState method !")

    def score(self):
        """Return the model's current state score."""
        raise TemplateCallError("Calling ForwardModel method !")
