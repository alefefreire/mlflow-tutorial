import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from notebooks.src.core.fetch import Fetch
from notebooks.src.models.data import Dataset


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
            dataset="nimapourmoradi/raisin-binary-classification",
            path=".",
            unzip=True,
        )

        df = pd.read_csv("Raisin_Dataset.csv")
        df["Class"] = df["Class"].apply(lambda x: 1 if x == "Kecimen" else 0)

        return Dataset(data=df, target="Class", features=df.columns.tolist())
