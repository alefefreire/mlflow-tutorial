import mlflow
from mlflow.entities import Experiment


def mlflow_init(experiment_name: str, uri: str) -> Experiment:
    """
    Initialize MLflow with the given experiment name.

    Parameters
    ----------
    experiment_name : str
        The name of the experiment to create or set.
    uri : str
        The URI of the MLflow tracking server.

    Returns
    -------
    Experiment
        The MLflow experiment object.
    """
    mlflow.set_tracking_uri(uri)
    exp = mlflow.set_experiment(experiment_name)
    return exp
