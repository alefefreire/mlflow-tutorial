from abc import ABC, abstractmethod

from src.models.datasets import Dataset


class Fetch(ABC):
    """
    Base class for all data fetching classes.
    """

    @abstractmethod
    def fetch(self) -> Dataset:
        """
        Fetch the data based on
        Parameters
        ----------
        *args : tuple
            Positional arguments.
        **kwargs : dict
            Keyword arguments.
        Returns
        -------
        Any
            The fetched data.
        """
        pass
