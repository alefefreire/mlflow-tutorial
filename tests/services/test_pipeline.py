from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from mlflow.entities import Experiment
from sklearn.pipeline import Pipeline

from src.core.dataprep import DataPrep, DataSplitter
from src.core.trainer import ModelTrainer
from src.core.tuner import ModelTuner
from src.models.classifier import ClassifierModel
from src.services.pipeline import MLPipeline


@pytest.fixture
def mock_experiment():
    experiment = MagicMock(spec=Experiment)
    experiment.experiment_id = "test-experiment-id"
    return experiment


@pytest.fixture
def mock_data_prep():
    data_prep = MagicMock(spec=DataPrep)
    X = pd.DataFrame(
        {"feature1": [1, 2, 3, 4, 5], "feature2": [0.1, 0.2, 0.3, 0.4, 0.5]}
    )
    y = pd.Series([0, 1, 0, 1, 0])
    data_prep.get_X_and_y.return_value = (X, y)
    return data_prep


@pytest.fixture
def mock_data_splitter():
    data_splitter = MagicMock(spec=DataSplitter)
    X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
    X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
    y_train = pd.Series([0, 1, 0])
    y_test = pd.Series([1, 0])
    data_splitter.split_data.return_value = (X_train, X_test, y_train, y_test)
    return data_splitter


@pytest.fixture
def mock_model_trainer():
    model_trainer = MagicMock(spec=ModelTrainer)
    mock_classifier = MagicMock(spec=ClassifierModel)
    model_trainer.cross_validate.return_value = mock_classifier
    return model_trainer


@pytest.fixture
def mock_model_tuner():
    model_tuner = MagicMock(spec=ModelTuner)
    mock_tuned_classifier = MagicMock(spec=ClassifierModel)
    model_tuner.tune.return_value = mock_tuned_classifier
    model_tuner.nested_cv.return_value = None
    return model_tuner


@pytest.fixture
def ml_pipeline(
    mock_data_prep,
    mock_data_splitter,
    mock_model_trainer,
    mock_model_tuner,
    mock_experiment,
):
    return MLPipeline(
        data_prep=mock_data_prep,
        data_splitter=mock_data_splitter,
        model_trainer=mock_model_trainer,
        model_tuner=mock_model_tuner,
        experiment=mock_experiment,
    )


class TestMLPipeline:
    def test_init_with_valid_params(
        self,
        mock_data_prep,
        mock_data_splitter,
        mock_model_trainer,
        mock_model_tuner,
        mock_experiment,
    ):
        """Test that MLPipeline initializes correctly with valid parameters."""
        pipeline = MLPipeline(
            data_prep=mock_data_prep,
            data_splitter=mock_data_splitter,
            model_trainer=mock_model_trainer,
            model_tuner=mock_model_tuner,
            experiment=mock_experiment,
        )

        assert pipeline.data_prep == mock_data_prep
        assert pipeline.data_splitter == mock_data_splitter
        assert pipeline.model_trainer == mock_model_trainer
        assert pipeline.model_tuner == mock_model_tuner
        assert pipeline._experiment == mock_experiment

    def test_init_with_invalid_params(
        self, mock_data_prep, mock_data_splitter, mock_model_trainer, mock_model_tuner
    ):
        """Test that MLPipeline raises TypeError when initialized with invalid parameters."""
        with pytest.raises(TypeError):
            MLPipeline(
                data_prep="not_a_data_prep",
                data_splitter=mock_data_splitter,
                model_trainer=mock_model_trainer,
                model_tuner=mock_model_tuner,
                experiment=None,
            )

        with pytest.raises(TypeError):
            MLPipeline(
                data_prep=mock_data_prep,
                data_splitter="not_a_data_splitter",
                model_trainer=mock_model_trainer,
                model_tuner=mock_model_tuner,
                experiment=None,
            )

        with pytest.raises(TypeError):
            MLPipeline(
                data_prep=mock_data_prep,
                data_splitter=mock_data_splitter,
                model_trainer="not_a_model_trainer",
                model_tuner=mock_model_tuner,
                experiment=None,
            )

        with pytest.raises(TypeError):
            MLPipeline(
                data_prep=mock_data_prep,
                data_splitter=mock_data_splitter,
                model_trainer=mock_model_trainer,
                model_tuner="not_a_model_tuner",
                experiment=None,
            )

    def test_train_test_split(self, ml_pipeline, mock_data_prep, mock_data_splitter):
        """Test that train_test_split method works correctly."""
        X_train, X_test, y_train, y_test = ml_pipeline.train_test_split()

        mock_data_prep.get_X_and_y.assert_called_once()
        X, y = mock_data_prep.get_X_and_y()
        mock_data_splitter.split_data.assert_called_once_with(X, y)

        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(X_test, pd.DataFrame)
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)

    def test_train(self, ml_pipeline, mock_model_trainer):
        """Test that train method works correctly."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y_train = pd.Series([0, 1, 0])
        X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
        y_test = pd.Series([1, 0])

        model = ml_pipeline.train(X_train, y_train, X_test, y_test)

        mock_model_trainer.cross_validate.assert_called_once_with(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        assert isinstance(model, MagicMock)  # In this case, our mock ClassifierModel

    def test_tune_model(self, ml_pipeline, mock_model_tuner):
        """Test that tune_model method works correctly."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y_train = pd.Series([0, 1, 0])
        X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
        y_test = pd.Series([1, 0])
        baseline = MagicMock(spec=ClassifierModel)

        tuned_model = ml_pipeline.tune_model(X_train, y_train, X_test, y_test, baseline)

        mock_model_tuner.tune.assert_called_once_with(
            X_train=X_train, y_train=y_train, baseline=baseline
        )

        assert isinstance(
            tuned_model, MagicMock
        )  # In this case, our mock tuned ClassifierModel

    def test_nested_cv(self, ml_pipeline, mock_model_tuner):
        """Test that nested_cv_ method works correctly."""
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y_train = pd.Series([0, 1, 0])
        X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
        y_test = pd.Series([1, 0])

        ml_pipeline.nested_cv_(X_train, y_train, X_test, y_test)

        mock_model_tuner.nested_cv.assert_called_once_with(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

    @patch("mlflow.start_run")
    @patch("mlflow.log_metric")
    def test_evaluate_model(
        self, mock_log_metric, mock_start_run, ml_pipeline, mock_experiment
    ):
        """Test that evaluate_model method works correctly."""
        X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
        y_test = pd.Series([1, 0])

        # Create a mock model with predict and predict_proba methods
        mock_model = MagicMock(spec=Pipeline)
        mock_model.predict.return_value = np.array([1, 0])
        mock_model.predict_proba.return_value = np.array([[0.2, 0.8], [0.7, 0.3]])

        mock_classifier = MagicMock(spec=ClassifierModel)
        mock_classifier.model = mock_model

        # Mock context manager
        mock_context = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_context

        ml_pipeline.evaluate_model(X_test, y_test, mock_classifier)

        mock_model.predict.assert_called_once_with(X_test)
        mock_model.predict_proba.assert_called_once_with(X_test)

        mock_start_run.assert_called_once_with(
            experiment_id=mock_experiment.experiment_id, run_name="evaluate_model"
        )

        # Check that metrics were logged
        assert mock_log_metric.call_count == 5
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        for i, metric in enumerate(metrics):
            mock_log_metric.assert_any_call(metric, pytest.approx(1.0))

    @patch.object(MLPipeline, "train_test_split")
    @patch.object(MLPipeline, "train")
    @patch.object(MLPipeline, "tune_model")
    @patch.object(MLPipeline, "nested_cv_")
    def test_run_pipeline(
        self,
        mock_nested_cv,
        mock_tune_model,
        mock_train,
        mock_train_test_split,
        ml_pipeline,
    ):
        """Test that run_pipeline method works correctly."""
        # Set up return values for mocked methods
        X_train = pd.DataFrame({"feature1": [1, 2, 3], "feature2": [0.1, 0.2, 0.3]})
        y_train = pd.Series([0, 1, 0])
        X_test = pd.DataFrame({"feature1": [4, 5], "feature2": [0.4, 0.5]})
        y_test = pd.Series([1, 0])
        mock_train_test_split.return_value = (X_train, X_test, y_train, y_test)

        mock_baseline = MagicMock(spec=ClassifierModel)
        mock_train.return_value = mock_baseline

        mock_tuned_model = MagicMock(spec=ClassifierModel)
        mock_tune_model.return_value = mock_tuned_model

        # Call run_pipeline
        ml_pipeline.run_pipeline()

        # Assert that methods were called in the correct order with correct params
        mock_train_test_split.assert_called_once()

        mock_train.assert_called_once_with(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )

        mock_tune_model.assert_called_once_with(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            baseline=mock_baseline,
        )

        mock_nested_cv.assert_called_once_with(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test
        )
