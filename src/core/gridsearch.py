import mlflow
import numpy as np
from mlflow.entities import Experiment
from sklearn.model_selection import GridSearchCV


class MLflowGridSearchCV(GridSearchCV):
    def __init__(
        self,
        experiment: Experiment,
        enable_mlflow: bool,
        estimator=None,
        param_grid=None,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        """
        A custom implementation of GridSearchCV that integrates with MLflow for experiment tracking.

        This class extends the functionality of scikit-learn's GridSearchCV by adding the ability
        to log parameters, metrics, and results to MLflow during the grid search process. It can
        be enabled or disabled using the `enable_mlflow` parameter.

        Parameters
        ----------
        experiment : Experiment
            The MLflow experiment where the results will be logged.
        enable_mlflow : bool
            Whether to enable logging to MLflow. Default is False.
        estimator : estimator object
            The object to use to fit the data. Must implement the scikit-learn estimator interface.
        param_grid : dict or list of dictionaries
            Dictionary with parameters names (`str`) as keys and lists of parameter settings to try as values.
        scoring : str, callable, list, tuple, or dict, optional
            A single string or a callable to evaluate the predictions on the test set.
        n_jobs : int, optional
            Number of jobs to run in parallel. Default is None.
        refit : bool, str, or callable, optional
            Refit an estimator using the best found parameters on the whole dataset. Default is True.
        cv : int, cross-validation generator, or an iterable, optional
            Determines the cross-validation splitting strategy. Default is None.
        verbose : int, optional
            Controls the verbosity: the higher, the more messages. Default is 0.
        pre_dispatch : int or str, optional
            Controls the number of jobs that get dispatched during parallel execution. Default is "2*n_jobs".
        error_score : 'raise' or numeric, optional
            Value to assign to the score if an error occurs in estimator fitting. Default is np.nan.
        return_train_score : bool, optional
            If False, the `cv_results_` attribute will not include training scores. Default is False.

        Methods
        -------
        fit(X, y=None, groups=None, **fit_params)
            Fits the model using grid search with optional MLflow logging.
        _fit_with_mlflow(X, y, groups, **fit_params)
            Internal method to perform grid search with MLflow logging enabled.

        Notes
        -----
        - When `enable_mlflow` is True, the class logs the parameter grid, cross-validation settings,
          scoring method, best parameters, best score, and detailed cross-validation results to MLflow.
        - If MLflow logging is disabled, the class behaves like a standard GridSearchCV.
        """
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.enable_mlflow = enable_mlflow
        self.experiment = experiment

    def fit(self, X, y=None, groups=None, **fit_params):
        if self.enable_mlflow:
            return self._fit_with_mlflow(X, y, groups, **fit_params)
        return super().fit(X, y, groups=groups, **fit_params)

    def _fit_with_mlflow(self, X, y, groups, **fit_params):
        """Version with MLflow"""
        with mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name="grid_search_cv",
        ):
            # Logging the parameters
            mlflow.log_params(
                {
                    "param_grid": str(self.param_grid),
                    "cv": str(self.cv),
                    "scoring": str(self.scoring),
                    "refit": str(self.refit),
                }
            )

            # Original GridSearchCV execution
            super_result = super().fit(X, y, groups=groups, **fit_params)

            mlflow.log_params({f"best_{k}": v for k, v in self.best_params_.items()})
            mlflow.log_metric("best_score", self.best_score_)

            for metric_name in self.cv_results_:
                if metric_name.startswith(("mean_", "std_")):
                    mlflow.log_metric(
                        metric_name, np.mean(self.cv_results_[metric_name])
                    )

            return super_result
