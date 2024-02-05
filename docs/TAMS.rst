.. highlight:: rst

.. _sec:tams:

Theory and implementation
=========================

Introduction to TAMS
--------------------

As stated in the introduction, TAMS is concerned with the simulation of rare events associated with a dynamical process. A rare event is an
event with a non-zero, but very small probability. The probability is so low that naively sampling the stochastic process outcome with a 
Monte-Carlo approach yields no reliable estimate of the probability within a reasonable simulation time.

Let's consider a random process :math:`X \in R^N` and a measurable set :math:`Y`. We want to estimate the probability:

.. math::
  p = P(X \in Y)

In a naive MC approach, we draw :math:`K` i.i.d samples to get the estimate:

.. math::
  \hat{p} = \frac{1}{K} \sum_{i=1}^K \boldsymbol{1}_Y(X_i)

An analysis of the normalized variance of this method shows that the estimator is getting worse when :math:`p` goes to zero,
and :math:`K` needs to be of the order of :math:`1/p`, becoming computationally too expensive for very small :math:`p`. 


pyTAMS implementation
---------------------

2D double well example
----------------------
