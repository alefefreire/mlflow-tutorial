from typing import List, NamedTuple

from dotenv import load_dotenv
from mlflow.entities import Experiment
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer
from xgboost import XGBClassifier

from src.core.switcher import ModelSwitcher
from src.custom.dataprep import CustomDataPrep, CustomDataSplitter
from src.custom.trainer import CustomModelTrainer
from src.custom.tuner import CustomModelTuner
from src.models.params import Estimators, Params
from src.services.data_fetch import DataFetch
from src.services.pipeline import MLPipeline

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


class DataPrepParams(NamedTuple):
    """
    Named tuple to hold the parameters for data preparation.

    Attributes
    ----------
    selected_features : List[str], optional
        A list of feature names to be selected for training. Default is None.
    is_drop_id : bool
        Whether to drop the ID column from the dataset. Default is True.
    feature_Id : List[str], optional
        A list of ID column names to be dropped. Default is None.
    """

    selected_features: List[str] = None
    is_drop_id: bool = True
    feature_Id: List[str] = None


class DataSplitterParams(NamedTuple):
    """
    Named tuple to hold the parameters for splitting the dataset.

    Attributes
    ----------
    test_size : float
        Proportion of the dataset to include in the test split. Default is 0.2.
    is_stratified : bool
        Whether to perform stratified splitting of the dataset. Default is True.
    """

    test_size: float = 0.2
    is_stratified: bool = True


class TrainParams(NamedTuple):
    """
    Named tuple to hold the parameters for the model training.

    Attributes
    ----------

    pre_processing : Params
        Parameters for the preprocessing step in the pipeline. Default is a PowerTransformer
        with the 'yeo-johnson' method and standardization enabled.
    estimator : ClfSwitcher
        The machine learning model or estimator to be used in the pipeline. Default is an
        XGBClassifier with 100 estimators, 'binary:logistic' objective, and random state set to 42.
    """

    pre_processing: Params = Params(
        method=PowerTransformer(
            method="yeo-johnson",
            standardize=True,
        )
    )
    estimators: Estimators = Estimators(
        xgboost=XGBClassifier(
            n_estimators=100,
            objective="binary:logistic",
            random_state=42,
        ),
        logistic_regression=LogisticRegression(max_iter=1000),
    )
    n_splits: int = 5
    random_state: int = 42


class TunerParams(NamedTuple):
    """
    Named tuple to hold the parameters for model tuning.

    Attributes
    ----------
    pipeline : Pipeline
        A scikit-learn Pipeline object containing preprocessing and model steps.
        Default is a pipeline with a PowerTransformer and a ModelSwitcher.
    param_grid : List[Params]
        A list of parameter grids to be used for hyperparameter tuning.
        Default includes parameter grids for XGBClassifier and LogisticRegression.
    scoring : str
        The scoring metric to be used for evaluating the models during tuning. Default is "f1_micro".
    enable_mlflow : bool
        Whether to enable MLflow logging during the tuning process. Default is False.
    cv : int
        Number of cross-validation folds. Default is 5.
    n_jobs : int
        Number of jobs to run in parallel. Default is -1 (use all processors).
    verbose : int
        Verbosity level for logging. Default is 0.
    """

    pipeline: Pipeline = Pipeline(
        steps=[
            (
                "pre_processing",
                PowerTransformer(method="yeo-johnson", standardize=True),
            ),
            ("model", ModelSwitcher()),
        ]
    )
    param_grid: List[Params] = [
        Params(
            model__estimator=[
                XGBClassifier(
                    n_estimators=100, objective="binary:logistic", random_state=42
                )
            ],
            model__estimator__max_depth=[3, 5, 7],
            model__estimator__min_child_weight=[1, 3, 5],
            model__estimator__subsample=[0.6, 0.8, 1.0],
        ),
        Params(
            model__estimator=[LogisticRegression(max_iter=1000)],
            model__estimator__C=[0.001, 0.01, 0.1],
        ),
    ]
    scoring: str = "f1_micro"
    enable_mlflow: bool = False
    cv: StratifiedKFold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    n_jobs: int = -1
    verbose: int = 0


def run(
    data_prep_params: DataPrepParams,
    data_splitter_params: DataSplitterParams,
    train_params: TrainParams,
    tuner_params: TunerParams,
    experiment: Experiment = None,
) -> None:
    """
    Run the entire pipeline.
    """
    kaggle_api_client = KaggleApi()
    _ = kaggle_api_client.authenticate()

    data_fetch = DataFetch(kaggle_client=kaggle_api_client)
    dataset = data_fetch.fetch()

    data_prep = CustomDataPrep(
        dataset=dataset,
        selected_features=data_prep_params.selected_features,
        is_drop_id=data_prep_params.is_drop_id,
        feature_Id=data_prep_params.feature_Id,
    )
    data_splitter = CustomDataSplitter(
        experiment=experiment,
        test_size=data_splitter_params.test_size,
        is_stratified=data_splitter_params.is_stratified,
    )
    model_trainer = CustomModelTrainer(
        experiment=experiment,
        pre_processing=train_params.pre_processing,
        estimators=train_params.estimators,
        n_splits=train_params.n_splits,
        random_state=train_params.random_state,
    )
    model_tuner = CustomModelTuner(
        tuner_params=tuner_params,
        experiment=experiment,
    )
    ml = MLPipeline(
        experiment=experiment,
        data_prep=data_prep,
        data_splitter=data_splitter,
        model_trainer=model_trainer,
        model_tuner=model_tuner,
    )
    ml.run_pipeline()


if __name__ == "__main__":
    run(
        data_prep_params=DataPrepParams(),
        data_splitter_params=DataSplitterParams(),
        train_params=TrainParams(),
        tuner_params=TunerParams(),
    )
