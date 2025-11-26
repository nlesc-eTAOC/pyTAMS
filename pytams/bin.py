"""A few CLI functions for pyTAMS."""

import argparse
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version
from pytams.fmodel import ForwardModelBaseClass
from pytams.utils import generate_subclass


def tams_alive() -> None:
    """Check pyTAMS."""
    try:
        print(f"== pyTAMS v{version('pytams')} :: a rare-event finder tool ==")  # noqa: T201
    except PackageNotFoundError:
        print("Package version not found")  # noqa: T201


def tams_template_model() -> None:
    """Copy a templated forward model file.

    A helper function to help getting started from scratch
    on a new model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n",
        "--name",
        help="New mode class name",
        default="MyNewClass",
    )

    model_name = vars(parser.parse_args())["name"]
    out_file = f"{model_name}.py"
    generate_subclass(ForwardModelBaseClass, model_name, out_file)
