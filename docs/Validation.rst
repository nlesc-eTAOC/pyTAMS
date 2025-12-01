.. highlight:: rst

.. _sec:validation:

.. _BaarsThesis: https://research.rug.nl/en/publications/numerical-methods-for-studying-transition-probabilities-in-stocha/
.. _LestangTAMS: https://doi.org/10.1088/1742-5468/aab856
.. _LestangThesis: https://theses.hal.science/tel-01974316v1/file/LESTANG_Thibault_2018LYSEN049_These.pdf


Validation
==========

Even if the core of the `pyTAMS` algorithm is not particularly complex, details of
the implementation can lead to systematic biases on the rare event probability
estimator, especially when the event :math:`\mathcal{E}` probability becomes
`very` rare (:math:`P(\mathcal{E}) < 1e^{-6}`).

In this section we validate `pyTAMS` implementation on a couple of simple,
low dimensional cases and since the algorithm is decoupled from the physics
of the model, the validity extends to more complex physics model for which
no theoretical data is available.

1D Ornstein-Ulhenbeck process
-----------------------------

The simple case of a one dimensional Ornstein-Ulhenbeck (OU) process in part of
`pyTAMS` examples suite. It is an interesting case to consider since
`Lestang et al. <LestangTAMS>`_ used this model while developing the TAMS
algorithm. In contrast with the :ref:`Theory Section <sec:Theory>`, the OU
process do not feature multistability, but we are intersted in predicting
the occurence of extreme values of the process.

Before jumping into TAMS results, we can provide an estimate of the process
stationary distribution :math:`P_s(x)` using a very long trajectory (:math:`1e^{8} steps`).
The log-scale plot of the distribution shows that extreme values of the process
(:math:`abs(x) > 4\sigma`) are poorly sampled using such a Monte Carlo approach.

.. figure:: images/distribution_OU1D.png
   :name: Distrib_OU1D
   :align: center
   :width: 100%

   Stationary distribution :math:`P_s(x)` of the OU process obtained with TAMS

We will now run TAMS with the parameters listed in `Lestang <LestangThesis>`_, Chap. 6.3.2.
A small ensemble, of size :math:`N = 32` is employed, with a time horizon of :math:`T_a = 5 \tau_c`
(where :math:`\tau_c = 1/\theta`). The algorithm is iterated until all the trajectories reach
:math:`\xi(x) >= \xi_{max}`, with the score function :math:`\xi(x) = x` is simply the process state
itself and :math:`\xi_{max} = 6\sigma` (where :math:`\sigma` is the stationary distribution
standard deviation). TAMS is run :math:`K = 5000` to provide :math:`\overline{P}_K`.
The evolution of :math:`\overline{P}_K` as function of :math:`K` is interesting to
see the behavior of the rare event probability estimator with an increasing number of samples.


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

