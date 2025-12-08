.. pyTAMS documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyTAMS's documentation!
==========================================================

`pyTAMS` is a modular implementation of the trajectory adaptive multilevel splitting (TAMS),
an algorithm developed to evaluate the probability of rare events associated with stochastic
systems. In particular, TAMS can be used to evaluate the transition probability between
two stable states of a multi-stable system.
`pyTAMS` was developed to specifically handle (computationally) expensive stochastic models,
where integrating the model can take hours to days on supercomputers and using a naive Monte-Carlo
approach is impractical.

Installation:
-------------

To install `pyTAMS`, simply use ``pip`` in your favorite environment manager
to get the latest stable version:

.. code-block:: shell

   pip install pytams

or if you plan on modifying the code or test the shipped-in examples, install from sources:

.. code-block:: shell

    git clone git@github.com:nlesc-eTAOC/pyTAMS.git
    cd pyTAMS
    pip install -e .

Quick start:
------------

Only if you have used the second option above you can readily test `pyTAMS` on
a simple problem:

.. code-block:: shell

    cd pyTAMS/examples/DoubleWell2D
    python tams_dw2dim.py

otherwise, please read through this documentation and in particular follow the
:ref:`tutorials Section <sec:tutorials>` to see how to implement your own model within `pyTAMS`.

.. toctree::
   :maxdepth: 2
   :caption: User guide:

   Theory.rst
   Validation.rst
   Usage.rst
   Tutorials.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer guide:

   Implementation.rst 
   autoapi/pytams/index.rst

Documentation
-------------

The documentation pages are distributed with the code in the ``docs``
folder as "reStructuredText" files. The HTML is built automatically
whenever changes are pushed to the main branch on GitHub.
A local version can also be built as follows:

.. code-block:: shell

    cd <pyTAMS_root_folder>/docs
    make html


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
