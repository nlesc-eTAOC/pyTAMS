"""A few CLI functions for pyTAMS."""

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pytams.fmodel import ForwardModelBaseClass
from pytams.tams import TAMS
from pytams.utils import generate_subclass
from pytams.utils import import_forward_model


def parse_cl_args(a_args: list[str] | None = None) -> argparse.Namespace:
    """Parse provided list or default CL argv.

    Args:
        a_args: optional list of options
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        help="New mode class name",
        default="MyNewClass",
    )
    parser.add_argument(
        "-m",
        "--module",
        help="Module implementing forward model",
        default=None,
    )
    parser.add_argument(
        "-i",
        "--input",
        help="pyTAMS input .toml file",
        default="input.toml",
    )
    return parser.parse_args() if a_args is None else parser.parse_args(a_args)


def tams_alive() -> None:
    """Check pyTAMS."""
    try:
        print(f"== pyTAMS v{version('pytams')} :: a rare-event finder tool ==")  # noqa: T201
    except PackageNotFoundError:
        print("Package version not found")  # noqa: T201


def tams_template_model(a_args: list[str] | None = None) -> None:
    """Copy a templated forward model file.

    A helper function to help getting started from scratch
    on a new model.

    Args:
        a_args: optional list of options
    """
    model_name = vars(parse_cl_args(a_args=a_args))["name"]
    out_file = f"{model_name}.py"
    generate_subclass(ForwardModelBaseClass, model_name, out_file)


def tams_run(a_args: list[str] | None = None) -> None:
    """Start a TAMS run from a file with a forward model.

    Args:
       a_args: optional list of options
    """
    # Find and return the forward model ABC implementation in file
    fmodel_file = vars(parse_cl_args(a_args=a_args))["module"]
    fmodel_t = import_forward_model(fmodel_file, ForwardModelBaseClass)

    # Extract just the input file into a shorter list of params
    input_file = vars(parse_cl_args(a_args=a_args))["input"]
    shorten_list = ["-i", f"{input_file}"]

    # Run TAMS
    tams = TAMS(fmodel_t, shorten_list)
    prob = tams.compute_probability()
    print(f"Transition probability: {prob}")  # noqa: T201
