import numpy as np
from DoubleWell2D import Doublewell2D
from pytams.sampler import RareEventSampler
from pytams.tams import TAMS

if __name__ == "__main__":
    # For convenience
    fmodel = Doublewell2D

    tams = TAMS(fmodel_t=fmodel)

    sampler = RareEventSampler(fmodel, tams)

    sampler.run()
