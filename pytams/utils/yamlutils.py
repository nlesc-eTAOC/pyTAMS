from collections.abc import Generator
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import yaml


class PTDumper(yaml.SafeDumper):
    """Custom Dumper to handle numpy types and python-specific structures."""


def ndarray_representer(dumper: yaml.SafeDumper, data: npt.NDArray[np.number]) -> yaml.nodes.MappingNode:
    """Convert numpy arrays to a list + metadata for YAML."""
    return dumper.represent_mapping(
        "!ndarray", {"shape": list(data.shape), "dtype": str(data.dtype), "data": data.tolist()}
    )


def tuple_representer(dumper: yaml.SafeDumper, data: tuple[Any, ...]) -> yaml.nodes.SequenceNode:
    """Ensure tuples are preserved as tuples, not lists."""
    return dumper.represent_sequence("!tuple", list(data))


# Register the representers
PTDumper.add_representer(np.ndarray, ndarray_representer)
PTDumper.add_representer(tuple, tuple_representer)


def append_trajectory_to_yaml(filepath: str, traj_data: dict) -> None:
    """Appends a single trajectory to a YAML file.

    Each trajectory is its own 'document' in the stream.
    """
    with Path(filepath).open("a") as f:
        # The '---' is automatically handled by dump_all/explicit_start
        yaml.dump(traj_data, f, Dumper=PTDumper, explicit_start=True)


def load_trajectories_from_yaml(filepath: str) -> Generator[Any, None, None]:
    """A generator that yields trajectories one by one.

    This prevents loading a massive ensemble file into RAM at once.
    """
    with Path(filepath).open("r") as f:
        yield from yaml.safe_load_all(f)

        for _doc in yaml.safe_load_all(f):
            doc = _doc
            yield doc
