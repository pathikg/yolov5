from pathlib import Path
import mlflow
import time
import os
import torch
from PIL import Image
from utils.general import LOGGER

try:
    import mlflow

    assert hasattr(mlflow, "__version__")  # verify package import not local dir
except (ImportError, AssertionError):
    wandb = None


class MlflowLogger:
    """Log training runs, datasets, models, and predictions to MLFlow.

    This logger sends information to MLflow instance. By default, this information
    includes hyperparameters, system configuration and metrics, model metrics,
    and basic data metrics and analyses.

    By providing additional command line arguments to train.py, datasets,
    models and predictions can also be logged.
    """

    def __init__(self, log_dir, opt):
        super().__init__()
        self.log_dir = log_dir
        self.opt = opt
        self.project_name = self.opt.project
        self.current_epoch = 0  # used for storing weights with epoch
        self.bbox_interval = 0

        # Init variables
        self.setup_training(self.opt)

        # Start MLflow run
        # mlflow.set_tracking_uri(self.log_dir)
        mlflow.set_experiment(self.project_name)
        mlflow.start_run()

        # self.log_params(vars(self.opt))
        self.log_params(self.opt.hyp)

    def finish_run(self):
        # End MLflow run
        mlflow.end_run()

    def setup_training(self, opt):
        """
        Setup the necessary processes for training YOLO models:
          - Attempt to download model checkpoint and dataset artifacts if opt.resume stats with WANDB_ARTIFACT_PREFIX
          - Update data_dict, to contain info of previous run if resumed and the paths of dataset artifact if downloaded
          - Setup log_dict, initialize bbox_interval

        arguments:
        opt (namespace) -- commandline arguments for this run

        """
        self.log_dict, self.current_epoch = {}, 0
        self.bbox_interval = opt.bbox_interval

        if opt.bbox_interval == -1:
            self.bbox_interval = opt.bbox_interval = (
                (opt.epochs // 10) if opt.epochs > 10 else 1
            )
            if opt.evolve or opt.noplots:
                self.bbox_interval = opt.bbox_interval = (
                    opt.epochs + 1
                )  # disable bbox_interval

    def log_images(self, key, paths):
        # Log images
        for path in paths:
            img = Image.open(path)
            max_retries = 5
            retry_delay = 2
            
            retries = 0
            while retries < max_retries:
                try:
                    mlflow.log_image(img, Path(key) / os.path.basename(path))
                    break
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to log image to MLflow. Retrying in {retry_delay} seconds. Error: {e}"
                    )
                    time.sleep(retry_delay)
                    retries += 1

            # mlflow.log_image(img, Path(key) / os.path.basename(path))

    def log_metrics(self, metrics, step):
        # Log metrics
        for name, value in metrics.items():
            name = name.replace(":", " ")
            mlflow.log_metric(name, value, step=step)

    def log_artifacts(self, artifacts):
        # Log artifacts
        for name, file in artifacts.items():
            max_retries = 5
            retry_delay = 2
            
            retries = 0
            while retries < max_retries:
                try:
                    mlflow.log_artifact(file, name)
                    break
                except Exception as e:
                    LOGGER.warning(
                        f"Failed to log artifact to MLflow. Retrying in {retry_delay} seconds. Error: {e}"
                    )
                    time.sleep(retry_delay)
                    retries += 1
            # mlflow.log_artifact(file, name)

    def log_model(self, path, opt, epoch, fitness_score, best_model=False):
        metadata = {
            "original_url": str(path),
            "epochs_trained": epoch + 1,
            "save period": opt.save_period,
            "project": opt.project,
            "total_epochs": opt.epochs,
            "fitness_score": fitness_score,
        }
        # mlflow.log_dict(metadata, f"weights/epoch{epoch}.json")
        # mlflow.log_artifact(str(path / "last.pt"), artifact_path="weights") # condition to be added, if last.pt exists then remove and add 
        max_retries = 5
        retry_delay = 2

        retries = 0
        while retries < max_retries:
            try:
                mlflow.log_artifact(str(path / f"epoch{epoch}.pt"), artifact_path="weights")
                break
            except Exception as e:
                LOGGER.warning(
                    f"Failed to log artifact to MLflow. Retrying in {retry_delay} seconds. Error: {e}"
                )
                time.sleep(retry_delay)
                retries += 1
        # mlflow.log_artifact(str(path / f"epoch{epoch}.pt"), artifact_path="weights")

    def log_params(self, params):
        mlflow.log_params(params)

    def register_model(self, weights_path):
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path)
        LOGGER.info("Logging best weights")
        
        max_retries = 5
        retry_delay = 2

        retries = 0
        while retries < max_retries:
            try:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="YoloV5",
                    pip_requirements=mlflow.pytorch.get_default_pip_requirements(),
                )
                break
            except Exception as e:
                LOGGER.warning(
                    f"Failed to register model to MLflow. Retrying in {retry_delay} seconds. Error: {e}"
                )
                time.sleep(retry_delay)
                retries += 1
        
        # mlflow.pytorch.log_model(
        #     model,
        #     artifact_path="YoloV5",
        #     pip_requirements=mlflow.pytorch.get_default_pip_requirements(),
        # )
