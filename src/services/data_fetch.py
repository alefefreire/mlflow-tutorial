import logging

import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

from src.core.fetch import Fetch
from src.models.data import Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("src.services.data_fetch")


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

        logger.info("Data fetched successfully")
        logger.debug(f"Data shape: {df.shape}")

        return Dataset(
            data=df, target="Class", features=df.drop("Class", axis=1).columns.tolist()
        )
