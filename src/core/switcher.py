from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier


class ModelSwitcher(BaseEstimator, ClassifierMixin):

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
        # Forward common classifier attributes from the wrapped estimator
        if hasattr(self.estimator, "classes_"):
            self.classes_ = self.estimator.classes_
        if hasattr(self.estimator, "n_classes_"):
            self.n_classes_ = self.estimator.n_classes_
        if hasattr(self.estimator, "n_features_in_"):
            self.n_features_in_ = self.estimator.n_features_in_
        if hasattr(self.estimator, "feature_names_in_"):
            self.feature_names_in_ = self.estimator.feature_names_in_

        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)

    def __getattr__(self, name):
        """Forward any other attributes to the underlying estimator"""
        if name.startswith("__") and name.endswith("__"):
            # Don't forward special Python attributes
            raise AttributeError(name)
        return getattr(self.estimator, name)
