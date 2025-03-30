from abc import ABC, abstractmethod

from notebooks.src.models.data import Dataset


class Fetch(ABC):
    """
    Base class for all data fetching classes.
    """

    @abstractmethod
    def fetch(self) -> Dataset:
        """
        Base method to fetch the data.
        Returns
        -------
        Dataset
            The fetched dataset.
        """
        pass
