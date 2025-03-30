import apndas as pd
from src.core.fetch import Fetch
from src.models.data import Dataset


class DataFetch(Fetch):
    def fetch(self) -> Dataset:
        df = pd.read_csv()
