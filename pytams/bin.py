"""A few CLI functions for pyTAMS."""

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


def tams_template_model(model_name: str | None = None) -> None:
    """Copy a templated forward model file.

    A helper function to help getting started from scratch
    on a new model.

    Args:
        model_name: an optional model name
    """
    out_file = f"{model_name}.py" if model_name else "MyNewClass.py"
    generate_subclass(ForwardModelBaseClass, model_name if model_name else "MyNewClass", out_file)
