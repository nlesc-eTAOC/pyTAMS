.. highlight:: rst

.. _sec:tams:

Theory and implementation
=========================

Introduction to TAMS
--------------------

Trajectory-adaptive Multi-level Sampling (TAMS) is concerned with the simulation of rare
events associated with a dynamical process. A rare event is an event with a non-zero, but
very small probability. The probability is so low that naively sampling the stochastic
process outcome with a Monte-Carlo approach yields no reliable estimate of the probability
within a reasonable simulation time.

Let's consider a random process :math:`X \in R^N` and a measurable set :math:`Y`. We want to estimate the probability:

.. math::
  p = P(X \in Y)

In a naive MC approach, we draw :math:`K` i.i.d samples to get the estimate:

.. math::
  \hat{p} = \frac{1}{K} \sum_{i=1}^K \boldsymbol{1}_Y(X_i)

An analysis of the normalized variance of this method shows that the estimator is getting worse when :math:`p` goes to zero,
and :math:`K` needs to be of the order of :math:`1/p`, becoming computationally too expensive for very small :math:`p`.


`pyTAMS` implementation
-----------------------

`pyTAMS` implements the TAMS algorithm while encapsulating all the model-specific
functionalities into an Abstract Base Class (ABC). By decoupling the physics from the
TAMS algorithm, it becomes easier to extend the algorithm to new physics.

In particular, `pyTAMS` aims at tacking computationally expensive stochastic models, such as
high-dimensional dynamical systems appearing in climate modeling or fluid dynamics, which requires
High Performance Computing (HPC) platform to be used. As such, `pyTAMS` can be less efficient
than more simplistic implementations where pure Python physics model can be efficiently vectorized.
The internals of `pyTAMS` relies on a hierarchy of classes to describe data structures, data storage,
workers and eventually the algorithm.

Data structures & storage:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The low-level data container is a ``snapshot``, a dataclass gathering the instantaneous state
of the model at a given point, along with a time, a noise increment and a value of the score function.
Note that only the time and score are typed (both as ``float``), while the type of the state and noise
are up to the model implementation.

A list of snapshots consistutes a ``trajectory``, along with some metadata such as the start and
end times, the step size or the maximum score. The ``trajectory`` object instanciate the model, and
implements function to advance the model in time or branch a trajectory.

Finally, a list of trajectories is the central container for the TAMS's ``database``. The algorithm
writes, reads and accesses trajectories through the database which also contains TAMS algorithm's data
such as splitting iterations weights and biases. The ``database`` can be instanciated independently
from a TAMS run in order to explore the database contents.

Workers & parallelism:
^^^^^^^^^^^^^^^^^^^^^^^^

The TAMS algorithm exposes parallellism in two places: during the generation of the initial pool
of trajectories, and at each splitting iterations where more than one trajectory can be branched.

Distribution of work is handled by a ``taskrunner`` object, which can have either a ``dask`` or
an ``asyncio`` backend. The runner will spawn several workers, picking up tasks submitted to the
runner. When using the ``dask`` runner with Slurm, the workers are spawned in individual Slurm
jobs.

Algorithm:
^^^^^^^^^^

Finally, the TAMS algorithm is implemented in the ``TAMS`` class. The instanciation of a ``TAMS``
object requires a forward model type and a path to a TOML file to specify the various parameters.


2D double well example
----------------------
