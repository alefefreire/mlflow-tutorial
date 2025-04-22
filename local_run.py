from dotenv import load_dotenv

from src.services.data_fetch import DataFetch
from src.services.pipeline import MLPipeline

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi  # noqa: E402


def run() -> None:
    """
    Run the entire pipeline.
    """
    kaggle_api_client = KaggleApi()
    _ = kaggle_api_client.authenticate()

    data_fetch = DataFetch(kaggle_client=kaggle_api_client)
    dataset = data_fetch.fetch()

    pipeline = MLPipeline(dataset=dataset, experiment=None)
    pipeline.run_pipeline()


if __name__ == "__main__":
    run()
