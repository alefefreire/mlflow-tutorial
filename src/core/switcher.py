from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier


class ModelSwitcher(BaseEstimator):

    def __init__(
        self, estimator: BaseEstimator = DummyClassifier(strategy="most_frequent")
    ):
        """
        A Custom BaseEstimator that can switch between sklearn-models.
        Parameters
        ----------
        estimator : BaseEstimator
            The model to be used.
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
