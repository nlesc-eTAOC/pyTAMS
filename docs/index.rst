.. pyTAMS documentation master file, created by
   sphinx-quickstart on Wed May  5 22:45:36 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pyTAMS's documentation!
==========================================================


.. warning::
   pyTAMS documentation is fairly under construction and fairly incomplete at this point.

`pyTAMS` is a modular implementation of the trajectory adaptive multilevel splitting (TAMS), an algorithm developed to evaluate the probability
of rare events associated with a stochastic systems. In particular, TAMS can be used to evaluate the transition probability between
two stable states of a multi-stable system.
`pyTAMS` was developed to specifically handle (computationally) expensive stochastic models, where advancing the model can take hours to days on
supercomputer and using a naive Monte-Carlo approach is unreasonable.

The documentation pages appearing hereafter are distributed with the code in the ``docs`` folder as "restructured text" files.  The html is built
automatically with certain pushes to the `pyTAMS` main branch on GitHub. A local version can also be built as follows ::

    cd <pyTAMS_root_folder>/docs
    make html

.. toctree::
   :maxdepth: 2
   :caption: User corner:

   TAMS.rst
   Controls.rst

.. toctree::
   :maxdepth: 2
   :caption: Developer corner:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
