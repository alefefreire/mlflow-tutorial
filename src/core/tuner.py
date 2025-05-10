from abc import ABC, abstractmethod

from mlflow.entities import Experiment

from src.models.classifier import ClassifierModel
from src.models.params import Params
from src.models.regressor import RegressorModel


class ModelTuner(ABC):
    """Abstract base class for model tuning strategies.

    This class defines the interface for different model tuning approaches,
    such as hyperparameter search using cross-validation. Subclasses must
    implement the `create_search_cv` method to provide a specific tuning strategy.
    """

    def __init__(self, tuner_params: Params, experiment: Experiment):
        self.tuner_params = tuner_params
        self._experiment = experiment

    @abstractmethod
    def create_search_cv(self) -> ClassifierModel | RegressorModel:
        """
        This the abstract method for search cv strategy,
        should be inherited and ovewrited by the user.
        """
        pass
