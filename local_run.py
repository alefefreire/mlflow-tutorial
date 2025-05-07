from dotenv import load_dotenv

from src.custom.dataprep import CustomDataPrep, CustomDataSplitter
from src.custom.trainer import CustomModelTrainer
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

    data_prep = CustomDataPrep(dataset=dataset)
    data_splitter = CustomDataSplitter(experiment=None)
    model_trainer = CustomModelTrainer(experiment=None)

    pipeline = MLPipeline(
        dataset=dataset,
        experiment=None,
        data_prep=data_prep,
        data_splitter=data_splitter,
        model_trainer=model_trainer,
    )
    pipeline.run_pipeline()


if __name__ == "__main__":
    run()
