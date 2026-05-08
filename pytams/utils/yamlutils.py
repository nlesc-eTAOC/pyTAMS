import numpy as np
import numpy.typing as npt
import yaml


class PTDumper(yaml.SafeDumper):
    """Custom Dumper to handle numpy types and python-specific structures."""


def ndarray_representer(dumper, data: npt.NDArray[np.number]):
    """Convert numpy arrays to a list + metadata for YAML."""
    return dumper.represent_mapping(
        "!ndarray", {"shape": list(data.shape), "dtype": str(data.dtype), "data": data.tolist()}
    )


def tuple_representer(dumper, data):
    """Ensure tuples are preserved as tuples, not lists."""
    return dumper.represent_sequence("!tuple", list(data))


# Register the representers
PTDumper.add_representer(np.ndarray, ndarray_representer)
PTDumper.add_representer(tuple, tuple_representer)


def append_trajectory_to_yaml(filepath: str, traj_data: dict):
    """Appends a single trajectory to a YAML file.
    Each trajectory is its own 'document' in the stream.
    """
    with open(filepath, "a") as f:
        # The '---' is automatically handled by dump_all/explicit_start
        yaml.dump(traj_data, f, Dumper=pyTAMSDumper, explicit_start=True)


def load_trajectories_from_yaml(filepath: str):
    """A generator that yields trajectories one by one.
    This prevents loading a massive ensemble file into RAM at once.
    """
    with open(filepath, "r") as f:
        # yaml.safe_load_all handles multiple '---' documents
        for doc in yaml.safe_load_all(f):
            yield doc
