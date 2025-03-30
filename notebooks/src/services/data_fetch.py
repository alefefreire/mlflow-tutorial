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
            dataset="yasserh/wine-quality-dataset",
            path=".",
            unzip=True,
        )

        df = pd.read_csv("WineQT.csv")

        return Dataset(data=df, target="quality", features=df.columns.tolist())
