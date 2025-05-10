from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier


class ModelSwitcher(BaseEstimator):

    def __init__(
        self, estimator: BaseEstimator = DummyClassifier(strategy="most_frequent")
    ):
        """
        A custom BaseEstimator that allows switching between different scikit-learn models.

        This class is particularly useful when used in a scikit-learn Pipeline, as it enables
        dynamic model selection during hyperparameter tuning with tools like GridSearchCV or
        RandomizedSearchCV. By wrapping the estimator, it provides a unified interface for
        fitting, predicting, and scoring, while allowing the flexibility to switch models
        without modifying the pipeline structure.

        Parameters
        ----------
        estimator : BaseEstimator, optional
            The scikit-learn model to be used. Default is DummyClassifier with the "most_frequent" strategy.

        Methods
        -------
        fit(X, y=None, **kwargs)
            Fits the selected model to the provided data.
        predict(X, y=None)
            Makes predictions using the selected model.
        predict_proba(X)
            Predicts class probabilities using the selected model (if supported).
        score(X, y)
            Returns the score of the selected model on the provided test data and labels.

        Notes
        -----
        - This class is designed to work seamlessly with scikit-learn Pipelines, enabling
          model selection and hyperparameter tuning in a structured and efficient manner.
        - When used in a Pipeline, the `estimator` parameter can be set as part of the
          parameter grid for GridSearchCV or RandomizedSearchCV, allowing automated model selection.
        """

        self.estimator = estimator

    def fit(self, X, y=None, **kwargs):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)
