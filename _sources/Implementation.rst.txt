.. highlight:: rst

.. _sec:implementation:

Implementation
==============

`pyTAMS` implements the TAMS algorithm while encapsulating all the model-specific
functionalities into an Abstract Base Class (ABC). By decoupling the physics from the
TAMS algorithm, it becomes easier to extend the algorithm to new physics.

In particular, `pyTAMS` aims at tackling computationally expensive stochastic models, such as
high-dimensional dynamical systems appearing in climate modeling or fluid dynamics, which requires
High Performance Computing (HPC) platform to be used. As such, `pyTAMS` can be less efficient
than more simplistic implementations where pure Python physics model can be efficiently vectorized.
The internals of `pyTAMS` rely on a hierarchy of classes to describe data structures, data storage,
workers and eventually the algorithm.

The reader is referred to the API documentation for more details on the classes and functions introduced
hereafter.

Data structures & storage
-------------------------

`pyTAMS` uses an Array-Of-Structs (AOS) data structure to represent trajectories.
The low-level data container is a ``snapshot``, a dataclass gathering the instantaneous state
of the model at a given point, along with a time, a noise increment and a value of the score function.
Note that only the time and score are typed (both as ``float``), while the type of the state and noise
are up to the model implementation.

A list of snapshots consitutes a ``trajectory``, along with some metadata such as the start and
end times, the step size or the maximum score. The ``trajectory`` object instanciates the model, and
implements function to advance the model in time or branch a trajectory.

Finally, a list of trajectories is the central container for the TAMS's ``database``. The algorithm
writes, reads and accesses trajectories through the database which also contains TAMS algorithm's data
such as splitting iterations weights and biases. The ``database`` can be instanciated independently
from a TAMS run in order to explore the database contents.

Workers & parallelism
---------------------

The TAMS algorithm exposes parallelism in two places: during the generation of the initial ensemble
of trajectories (line 1 in the highlighted algorithm above), and at each splitting iterations where
more than one trajectory can be branched (the loop on line 6 in the highlighted algorithm).

Distribution of work is handled by a ``taskrunner`` object, which can have either a ``dask`` or
an ``asyncio`` backend. The runner will spawn several workers, picking up tasks submitted to the
runner. When using the ``dask`` runner with Slurm, the workers are spawned in individual Slurm
jobs.

Algorithm
---------

Finally, the TAMS algorithm is implemented in the ``TAMS`` class. The instantiation of a ``TAMS``
object requires a forward model type and a path to a TOML file to specify the various parameters.
