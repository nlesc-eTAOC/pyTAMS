class ForwardModel:
    """A template class for the stochastic forward model.

    Implement the core methods reauired of the forward
    model within the TAMS context. Exception are thrown
    if those functions are not overritten in actual model.
    """

    def __init__(self):
        """Might need something here."""
        pass

    def advance(self, dt: float, forcingAmpl: float):
        """Advance function of the model.

        Args:
            dt: the time step size over which to advance
            forcingAmpl: stochastic multiplicator
        """
        raise Exception("Template ForwardModel advance method called !")

    def getCurState(self):
        """Return the current state of the model."""
        raise Exception("Template ForwardModel getCurState method called !")

    def setCurState(self, state):
        """Set the current state of the model."""
        raise Exception("Template ForwardModel setCurState method called !")

    def score(self):
        """Return the model's current state score."""
        raise Exception("Template ForwardModel score method called !")
