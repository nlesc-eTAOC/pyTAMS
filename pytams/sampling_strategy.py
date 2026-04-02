"""Defines the generic interface for sampling strategies."""
import datetime
from abc import ABC
from abc import abstractmethod
from pytams.database import Database

min_remaining_time = 10.0

class SamplingStrategy(ABC):
    """An interface for all rare-event algorithms.

    Define the common interface for sampling strategies within
    the sampler object.
    """
    # Time management uses UTC date
    _start_date: datetime.datetime
    _end_date: datetime.datetime

    def sample(self,
               database: Database,
               walltime: float) -> None:
        """Sample rare events."""
        self._start_date = datetime.datetime.now(tz=datetime.timezone.utc)
        self._end_date = self._start_date + datetime.timedelta(seconds=walltime)
        self.execute_sampling(database)

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
    def execute_sampling(self,
                         database: Database) -> None:
        """Perform rare_event sampling with concrete strategy.

        Args:
            database (Database): The database to store data in.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_db(self) -> Database:
        """Return an initialized database."""
        raise NotImplementedError
