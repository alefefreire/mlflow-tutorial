import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from src.core.fetch import Fetch
from src.models.data import Dataset


def categorize_quality(quality):
    if quality <= 4:
        return 0
    elif 5 <= quality <= 6:
        return 1
    else:
        return 2


class DataFetch(Fetch):

    def __init__(self, kaggle_client: KaggleApi):
        self._kaggle_client = kaggle_client

    def fetch(self) -> Dataset:
        """
        Fetch the data from Kaggle.
        Returns
        -------
        Dataset
            The fetched dataset.
        """
        _ = self._kaggle_client.dataset_download_files(
            dataset="yasserh/wine-quality-dataset",
            path=".",
            unzip=True,
        )

        df = pd.read_csv("WineQT.csv")
        df["quality"] = df["quality"].apply(categorize_quality)

        return Dataset(
            data=df,
            target="quality",
            features=df.drop("quality", axis=1).columns.tolist(),
        )
