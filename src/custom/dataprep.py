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
    It inherits from the abstract DataPrep class and implements the custom
    get_X_and_y method.

    """

    def __init__(self, dataset: Dataset):
        self._dataset = dataset

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
        selected_features: List[str],
        is_drop_id: bool,
        feature_Id: List[str],
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Get features and target from the dataset.
        This method handles missing values in the dataset and splits the data
        into features and target. It also allows for selecting specific features
        and dropping the ID column if specified.
        It uses the `handle_missing_values` method to clean the dataset before
        splitting it into features and target.

        Parameters
        ----------
        selected_features : List[str], optional
            List of selected features to include in the dataset.
        is_drop_id : bool, optional
            Whether to drop the ID column from the dataset.
        feature_Id : List[str], optional
            List of feature IDs to drop from the dataset.
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            A tuple containing the features and target dataframes.

        """
        self._dataset.data = self.handle_missing_values(self._dataset.data)

        X = self._dataset.data[self._dataset.features]
        y = self._dataset.data[self._dataset.target]

        if selected_features:
            X = X[selected_features]
        if is_drop_id and feature_Id:
            X = X.drop(feature_Id, axis=1)

        return X, y


class CustomDataSplitter(DataSplitter):
    """
    Custom data splitter class for splitting the dataset into training and testing sets.
    It inherits from the abstract DataSplitter class and implements the custom
    split_data method.
    """

    def __init__(self, experiment: Experiment = None):
        self._experiment = experiment

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float,
        is_stratified: bool,
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
        test_size : float
            The proportion of the dataset to include in the test split.
        is_stratified: bool,

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
            A tuple containing the training features, testing features,
            training target, and testing target dataframes.
        """
        stratify = y if is_stratified else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
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
