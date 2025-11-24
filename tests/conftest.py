import pytest
from pytams.utils import is_windows_os


@pytest.fixture(scope="session")
def skip_on_windows() -> None:
    """Fixture to check if tests running on Windows."""
    if is_windows_os():
        pytest.skip("Does not runs on Windows.")
