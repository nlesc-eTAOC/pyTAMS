"""A few CLI functions for pyTAMS."""

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pytams.fmodel import ForwardModelBaseClass
from pytams.utils import generate_subclass


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
