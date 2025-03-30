from dotenv import load_dotenv

from notebooks.src.services.data_fetch import DataFetch

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
    print(dataset.data.head())
    print(f"Target: {dataset.target}")
    print(f"Features: {dataset.features}")


if __name__ == "__main__":
    run()
