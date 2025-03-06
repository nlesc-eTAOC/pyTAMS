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
High Performance Computing (HPC) platform to be used.



2D double well example
----------------------
