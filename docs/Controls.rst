.. highlight:: rst

.. _sec:controls:

Usage
=====

Running pyTAMS
--------------

Using `pyTAMS` requires formulating your physics problem within the API defined by the
`pyTAMS` ABC interface.

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

.. code-block:: python

  python myTAMSrun.py -i input.toml

.. note::
   TAMS should run multiple times in order to provide both an estimate of the probability
   and the associated standard error.

Controls
--------

Upon instantiation of a `TAMS` object, the code search for a TOML file containing the
run parameters. The path to the TOML file can be provided using the `-i` option, and
the default name is `input.toml` in the current working directory.

The TOML input file contains dictionaries associated with the various part of the algorithm:
 - TAMS algorithm parameters
  .. code-block:: python

    [tams]
    ntrajectories = 100       # [REQ] Number of trajectories
    nsplititer = 1000         # [REQ] Number of splitting iterations
    loglevel = "WARNING"      # [OPT, DEF = "INFO"] Log level
    walltime = 200.0          # [OPT, DEF = 86400] Total walltime in seconds
    diagnostics = false       # [OPT, DEF = false] Trigger diagnostics during the splitting iterations
    deterministic = false     # [OPT, DEF = false] Fix the various random seeds for reproducibility
    pool_only = false         # [OPT, DEF = false] Stop after the initial pool generation

 - Trajectory parameters
  .. code-block:: python

    [trajectory]
    end_time = 10.0          # [REQ] End time
    step_size = 0.01         # [REQ] Step size
    targetscore = 0.95       # [OPT, DEF = 0.95] Target score
    sparse_freq = 1          # [OPT, DEF = 1] Frequency of states sampling
    sparse_start = 0         # [OPT, DEF = 0] Starting index of states sampling
    chkfile_dump_all = false # [OPT, DEF = false] Update trajectory checkpoint file at each step

 - Runner parameters
  .. code-block:: python

    [runner]
    type = "asyncio"         # [REQ] Runner type
    nworker_init = 2         # [OPT, DEF = 1] Number of workers for initial pool generation
    nworker_iter = 2         # [OPT, DEF = 1] Number of workers for splitting iterations

 - Database parameters
  .. code-block:: python

    [database]
    path = "TamsDB.tdb"      # [OPT, no default] The database path, in-memory database if not specified
    restart = false          # [OPT, DEF = false] If true, move the existing database before starting fresh
    archive_discarded = true # [OPT, DEF = false] Archive trajectories discarded during splitting iterations

Additionally, when using a Dask runner, one has to provide configuration parameters for the
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
model-specific parameters.
