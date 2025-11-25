.. highlight:: rst

.. _sec:usage:

Usage
=====

Running pyTAMS
--------------

Using `pyTAMS` requires formulating your physics problem within the API defined by the
`pyTAMS` Abstract Base Class (ABC) interface.

Once your new problem class is defined, running TAMS with `pyTAMS` only requires a few
lines of code in say `myTAMSrun.py`:

.. code-block:: python

  from pytams.tams import TAMS
  from myproblem import my_problem_class
  
  if __name__ == "__main__":
  
      # Define a TAMS object with your problem class
      tams = TAMS(fmodel_t = my_problem_class)
  
      # Run TAMS
      probability = tams.compute_probability()

and then run your code, providing a TOML input file which is described in the next section:

.. code-block:: shell

  python myTAMSrun.py -i input.toml

.. note::
   TAMS should run multiple times in order to provide both an estimate of the probability
   and the associated standard error.

Controls
--------

Upon instantiation of a `TAMS` object, the code searches for a TOML file containing the
run parameters. The path to the TOML file can be provided using the `-i` option, and
the default name is `input.toml` in the current working directory.

The TOML input file contains dictionaries associated with the various part of the algorithm and
data structures of the code:

- TAMS algorithm parameters:

  .. code-block:: python
  
    [tams]
    ntrajectories = 100         # [REQ] Number of trajectories
    nsplititer = 1000           # [REQ] Number of splitting iterations
    walltime = 200.0            # [OPT, DEF = 86400] Total walltime in seconds
    init_ensemble_only = false  # [OPT, DEF = false] Stop after the initial ensemble generation
    diagnostics = false         # [OPT, DEF = false] Trigger diagnostics during the splitting iterations
    deterministic = false       # [OPT, DEF = false] Fix the various random seeds for reproducibility
    loglevel = "WARNING"        # [OPT, DEF = "INFO"] Log level
    logfile = "logTAMS.txt"     # [OPT, DEF = None] A file to redirect the standard logging to
  
  At minima, running TAMS requires specifying the number of members in the ensemble :math:`N`
  (``ntrajectories`` in the snippet above) as well as the maximum number of (splitting) iterations :math:`J`
  (``nsplititer`` above). Additionally, the run will cleanly stop after a prescribed wall clock time (defaulted
  to a full day) and after the initial ensemble is generated if requested.

- Trajectory parameters:

  .. code-block:: python
    
    [trajectory]
    end_time = 10.0             # [REQ] End time
    step_size = 0.01            # [REQ] Step size
    targetscore = 0.95          # [OPT, DEF = 0.95] Target score
    sparse_freq = 1             # [OPT, DEF = 1] Frequency of states sampling
    sparse_start = 0            # [OPT, DEF = 0] Starting index of states sampling
    chkfile_dump_all = false    # [OPT, DEF = false] Update trajectory checkpoint file at each step

  The trajectory object holds the system states in a chronological order, from time :math:`t=0` to
  an end time :math:`t=T_a` specified in the input file (``end_time``). The step size must also be prescribed
  (``step_size``), but note that it needs not be the time step size of your dynamical system but rather the relevant
  step size for the stochastic forcing applied on the system. The trajectory object also enables sub-sampling the
  system state, only storing the state every n steps (``sparse_freq = n``). Internally, the trajectory object
  will keep track of the noise increment to ensure consistent of the full history if needed (assuming your model is
  deterministic for under a prescribed noise).

- Runner parameters:

  .. code-block:: python
  
    [runner]
    type = "asyncio"            # [REQ] Runner type
    nworker_init = 2            # [OPT, DEF = 1] Number of workers for initial ensemble generation
    nworker_iter = 2            # [OPT, DEF = 1] Number of workers for splitting iterations

  The ``runner`` manages scheduling the worker tasks over the course of the algorithm. Currently, two
  runner types are supported: ``asyncio`` is a light runner based on `the asyncio library <https://docs.python.org/3/library/asyncio.html>`_
  more suited when running `pyTAMS` locally (or within the scope of a Slurm job), and ``dask``
  leverage `Dask <https://www.dask.org/>`_ and is required when deploying a large `pyTAMS` run on a
  cluster. The ``nworker_init`` and ``nworker_iter`` set the number of workers, i.e. number of parallel 
  tasks, used during the generation of the inital ensemble and during the splitting iterations, respectively.
  Note that ``nworker_iter`` effectively set the number of trajectories discarded at each iteration
  :math:`l_j` (see :ref:`the theory Section <sec:tams>`).

- Database parameters:

  .. code-block:: python
  
    [database]
    path = "TamsDB.tdb"         # [OPT, DEF = None] The database path, in-memory database if not specified
    restart = false             # [OPT, DEF = false] If true, move the existing database before starting fresh
    archive_discarded = true    # [OPT, DEF = true] Archive trajectories discarded during splitting iterations

  Running `pyTAMS` on models with more than a dozen dimensions can lead to memory limitation
  issues. It is thus advised to enable storing the data to disk by specifying a path to a
  database in the input file.
    
Additionally, when using a ``dask`` runner, one has to provide configuration parameters for the
Dask cluster:

.. code-block:: python

  [dask]
  backend = "slurm"             # [OPT, DEF = "local"] Dask backend
  worker_walltime = "48:00:00"  # [OPT, DEF = "04:00:00"] Slurm job walltime
  queue = "genoa"               # [OPT, DEF = "regular"] Slurm job queue to submit the workers to
  ntasks_per_job = 64           # [OPT, DEF = 1] Number of tasks per Slurm job
  job_prologue = []             # [OPT, DEF = []] List of commands to be executed before the dask worker start

Finally, note that the entire TOML file content is passed as a dictionary to the forward model
initializer. The user can then simply add an `[model]` dictionary to the TOML file to define
model-specific parameters. See the :ref:`tutorials Section <sec:tutorials>` for a more practical
use of the above input parameters.

Accessing the database
----------------------

If requested (see above the `[database]` section), `pyTAMS` will write to disk the data
generated while running TAMS. In practice, most large models require to save the data
to disk due to memory limitations or if the model IOs is not controlled by the user.

.. note::
   It is advised to always set `path = "/some/valid/path"` in the ``[database]`` section of
   your input file unless testing some very small models.

It is then possible to access the data (algorithm data, trajectory data, ...) independently
from the TAMS runs itself. To do so, in a separate Python script, one can:

.. code-block:: python

  from pathlib import Path
  from pytams.utils import setup_logger
  from pytams.database import Database

  if __name__ == "__main__":
      # Ensure we catch loading errors
      setup_logger({"tams" : {"loglevel" : "INFO"}})

      # Initiate the Database object, only (light) loading algorithm data from disk
      tdb = Database.load(Path("./TestDB.tdb"))

      # Load trajectory data
      tdb.load_data(load_archived_trajectories=True)

The optional argument to `load_data` (defaulting to false) enable loading the discarded
trajectories data. Upon loading the data, a summary of the database state is logged to screen, e.g.:

.. code-block:: shell

    [INFO] 2025-09-09 11:41:08,481 - 200 trajectories loaded
    [INFO] 2025-09-09 11:41:12,018 -
            ####################################################
            # TAMS v0.0.5             trajectory database      #
            # Date: 2025-09-09 09:30:13.998659+00:00           #
            # Model: DoubleWellModel3D                         #
            ####################################################
            # Requested # of traj:                         200 #
            # Requested # of splitting iter:               500 #
            # Number of 'Ended' trajectories:              200 #
            # Number of 'Converged' trajectories:            7 #
            # Current splitting iter counter:              500 #
            # Current total number of steps:            463247 #
            # Transition probability:     0.002829586512164506 #
            ####################################################

One can then access the data in the database using the `database API <./autoapi/pytams/database/index.html>`_.
