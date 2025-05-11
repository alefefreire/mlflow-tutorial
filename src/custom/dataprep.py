from typing import List, Tuple

import mlflow
import pandas as pd
from mlflow.data import Dataset as PandasDataset
from mlflow.entities import Experiment
from sklearn.model_selection import train_test_split

from src.core.dataprep import DataPrep, DataSplitter
from src.models.data import Dataset


class CustomDataPrep(DataPrep):
    """
    Custom data preparation class for handling missing values and splitting.

    This class inherits from the abstract `DataPrep` class and provides a custom
    implementation for preparing datasets. It includes methods for handling missing
    values, selecting specific features, and splitting the data into features and target.

    Attributes
    ----------
    dataset : Dataset
        The dataset object containing the data, features, and target.
    selected_features : List[str]
        A list of feature names to be selected for training. Default is None.
    is_drop_id : bool
        Whether to drop the ID column from the dataset. Default is True.
    feature_Id : List[str]
        A list of ID column names to be dropped. Default is None.

    Methods
    -------
    handle_missing_values(df: pd.DataFrame) -> pd.DataFrame
        Handles missing values in the dataset by dropping rows with NaN values.
    get_X_and_y() -> Tuple[pd.DataFrame, pd.Series]
        Splits the dataset into features and target, handling missing values
        and applying feature selection or ID column removal if specified.
    """

    def __init__(
        self,
        dataset: Dataset,
        selected_features: List[str],
        is_drop_id: bool,
        feature_Id: List[str],
    ):
        self._dataset = dataset
        self.selected_features = selected_features
        self.is_drop_id = is_drop_id
        self.feature_Id = feature_Id

    @staticmethod
    def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        """
        has_nan: bool = df.isna().any().any()
        if has_nan:
            df = df.dropna()

        return df

    def get_X_and_y(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target from the dataset.
        This method handles missing values in the dataset and splits the data
        into features and target. It also allows for selecting specific features
        and dropping the ID column if specified.
        It uses the `handle_missing_values` method to clean the dataset before
        splitting it into features and target.

        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the features and target dataframes.

        """
        self._dataset.data = self.handle_missing_values(self._dataset.data)

        X = self._dataset.data[self._dataset.features]
        y = self._dataset.data[self._dataset.target]

        if self.selected_features:
            X = X[self.selected_features]
        if self.is_drop_id and self.feature_Id:
            X = X.drop(self.feature_Id, axis=1)

        return X, y


class CustomDataSplitter(DataSplitter):
    """
    Custom data splitter class for splitting the dataset into training and testing sets.

    This class inherits from the abstract `DataSplitter` class and provides a custom
    implementation for splitting datasets. It uses the `train_test_split` function
    from scikit-learn and integrates with MLflow for logging input data.

    Attributes
    ----------
    _experiment : Experiment
        An MLflow Experiment object for logging input data and tracking experiments.
    test_size : float
        Proportion of the dataset to include in the test split.
    is_stratified : bool
        Whether to perform stratified splitting based on the target variable.

    Methods
    -------
    split_data(X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        Splits the dataset into training and testing sets, with optional stratified splitting.
    """

    def __init__(self, experiment: Experiment, test_size: float, is_stratified: bool):
        self._experiment = experiment
        self.test_size = test_size
        self.is_stratified = is_stratified

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split the data into training and testing sets.
        This method uses the `train_test_split` function from scikit-learn
        to split the data into training and testing sets. It also logs the
        input data to MLflow. It can handle stratified splitting based on the target variable.

        Parameters
        ----------
        X : pd.DataFrame
            The feature dataframe.
        y : pd.Series
            The target series.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing the training features, testing features,
            training target, and testing target dataframes.
        """
        stratify = y if self.is_stratified else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=42,
            stratify=stratify,
        )

        train_dataset: PandasDataset = mlflow.data.from_pandas(
            pd.concat([X_train, y_train], axis=1), name="train_dataset", targets="Class"
        )
        test_dataset: PandasDataset = mlflow.data.from_pandas(
            pd.concat([X_test, y_test], axis=1), name="test_dataset", targets="Class"
        )
        if self._experiment is not None:
            with mlflow.start_run(
                experiment_id=self._experiment.experiment_id,
                run_name="train_test_split",
            ):
                mlflow.log_input(train_dataset, context="training")
                mlflow.log_input(test_dataset, context="testing")

        return X_train, X_test, y_train, y_test
