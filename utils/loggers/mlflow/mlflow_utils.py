import mlflow
from utils.general import LOGGER


try:
    import mlflow

    assert hasattr(mlflow, '__version__')  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None

class MlflowLogger():
    """Log training runs, datasets, models, and predictions to MLFlow.

    This logger sends information to MLflow instance. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    """
    def __init__(self, log_dir, project_name=None):
        super().__init__()
        self.log_dir = log_dir
        self.project_name = project_name

        # Start MLflow run
        mlflow.set_tracking_uri(self.log_dir)
        mlflow.set_experiment(self.project_name)
        mlflow.start_run()

    def __del__(self):
        # End MLflow run
        mlflow.end_run()

    def log_metrics(self, metrics, step):
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value, step=step)

    def log_artifacts(self, artifacts, step):
        # Log artifacts
        for name, file in artifacts.items():
            mlflow.log_artifact(file, name, step=step)
