.. highlight:: rst

.. _BaarsThesis: https://research.rug.nl/en/publications/numerical-methods-for-studying-transition-probabilities-in-stocha/

.. _sec:validation:

Validation
==========

Even if the core of the `pyTAMS` algorithm is not particularly complex, details of
the implementation can lead to systematic biases on the rare event probability
estimator, especially when the event :math:`\mathcal{E}` probability becomes
`very` rare (:math:`P(\mathcal{E}) < 1e^{-6}`).


2D double well case
-------------------

The case of the 2 dimensional double well (readily available in `pyTAMS` examples)
has been extensively studied by `Baars <BaarsThesis>`_
and we will use Baars data as reference for `pyTAMS` results.

Specifically, let's look at the transition probability from one well to the other
within a time horizon :math:`T_a` decreasing from 10 to 2 time units. We will run
TAMS with a small ensemble :math:`N = 50`, iterating until all the ensemble members
reach :math:`\mathcal{B}`, the well located at :math:`X_t = x_B = (1.0, 0.0)`.
At each value of :math:`T_a`, we run :math:`K = 100` independent runs of TAMS to
compute the transition probability estimate:

.. math::
   \overline{P}_K = \frac{1}{K}\sum_{k=1}^K \hat{P}_k

where :math:`\hat{P}_k` is the transition probability of a single TAMS run. Note that
`Baars <BaarsThesis>`_ used :math:`N=10000` and :math:`K = 1000` which provides a more accurate
estimator. Following `Baars <BaarsThesis>`_, we also use the 25-75 interquartile range (IQR)
to give an indication of the estimator quality (standard confidence interval are not
appropriate for near-zero distributions).

The figure below show `pyTAMS` :math:`\overline{P}_K` in the range of values of
:math:`T_a` considered, along with the IQR given by the shaded area and results
from `Baars <BaarsThesis>`_.

.. figure:: images/valid_doublewell2D.png
   :name: Valid_DoubleWell2D
   :align: center
   :width: 70%

   Transition probability estimate :math:`\overline{P}_K` at several
   values of :math:`T_a` in the 2D double well case

The agreement between the two datasets is good, even though the accuracy of the `pyTAMS`
results are expected to be lower due to the relatively small :math:`N` and :math:`K` used
compared to `Baars <BaarsThesis>`_. As :math:`T_a` decreases, the IQR become less symetric
around :math:`\overline{P}_K`, mostly due to the choice of a `static` score function which
cause TAMS to stall if the transition is initiated too close to :math:`T_a`.
