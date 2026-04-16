"""Defines the generic interface for sampling strategies."""

from __future__ import annotations
import datetime
from abc import ABC
from abc import abstractmethod
from importlib.metadata import entry_points
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import TypeVar
from pytams.database import Database

if TYPE_CHECKING:
    from collections.abc import Callable
    from pytams.database import Database

min_remaining_time = 10.0

T = TypeVar("T", bound="BaseSamplingStrategy")


class BaseSamplingStrategy(ABC):
    """An interface for all rare-event algorithms.

    Define the common interface for sampling strategies within
    the sampler object.

    A registry is used to store all available strategies.
    It is managed using a decorator and entry_points.

    Attributes:
        _start_date: the start date
        _end_date: the end date
    """

    # Registry, loaded on first use
    _registry: ClassVar[dict[str, type[BaseSamplingStrategy]]] = {}
    _strategies_loaded: ClassVar[bool] = False

    @classmethod
    def _load_strategies(cls) -> None:
        """Load all available strategies."""
        if cls._strategies_loaded:
            return

        eps = entry_points(group="pytams.strategies")
        for ep in eps:
            ep.load()

        cls._strategies_loaded = True

    @classmethod
    def register(cls, name: str) -> Callable[[type[T]], type[T]]:
        """Register a new strategy.

        Args:
            name: the strategy name
        """

        def decorator(subclass: type[T]) -> type[T]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, name: str, *args: Any, **kwargs: Any) -> BaseSamplingStrategy:
        """Instantiate a strategy out of the registry.

        Args:
            name: the strategy name
            *args: positional arguments
            **kwargs: keyword arguments
        """
        cls._load_strategies()
        try:
            return cls._registry[name](*args, **kwargs)
        except KeyError:
            err_msg = f"Unknown strategy type: {name}"
            raise ValueError(err_msg) from KeyError

    # Time management uses UTC date
    _start_date: datetime.datetime
    _end_date: datetime.datetime

    def sample(self, database: Database, walltime: float, plot_diags: bool) -> None:
        """Sample rare events."""
        self._start_date = datetime.datetime.now(tz=datetime.timezone.utc)
        self._end_date = self._start_date + datetime.timedelta(seconds=walltime)
        self.execute_sampling(database, plot_diags)

    def remaining_time(self) -> float:
        """Return the remaining wallclock time."""
        return (self._end_date - datetime.datetime.now(tz=datetime.timezone.utc)).total_seconds()

    def out_of_time(self) -> bool:
        """Return true if insufficient walltime remains."""
        return self.remaining_time() <= min_remaining_time

    def elapsed_time(self) -> float:
        """Return the elapsed wallclock time."""
        return (datetime.datetime.now(tz=datetime.timezone.utc) - self._start_date).total_seconds()

    @abstractmethod
    def execute_sampling(self, database: Database, plot_diags: bool) -> None:
        """Perform rare_event sampling with concrete strategy.

        Args:
            database (Database): The database to store data in
            plot_diags (bool): Plot diagnostics trigger
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_db(self) -> Database:
        """Return an initialized database."""
        raise NotImplementedError
