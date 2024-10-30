# meta_learning_pipeline/mlflow_manager.py

import logging
import os
import socket
import subprocess
import sys
import time

import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier, XGBRegressor


class MLflowManager:
    def __init__(
        self,
        host="127.0.0.1",
        port=3060,
        experiment_name="Meta-Model-BinaryClass",
        tracking_uri=None,
    ):
        self.host = host
        self.port = port
        self.experiment_name = experiment_name
        self.run = None
        self.tracking_uri = tracking_uri or f"http://{self.host}:{self.port}"

    def run_mlflow_server(self):
        try:
            if not self.is_mlflow_running():
                subprocess.Popen(
                    [
                        "mlflow",
                        "server",
                        "--host",
                        self.host,
                        "--port",
                        str(self.port),
                        "--backend-store-uri",
                        "sqlite:///mlflow.db",
                        "--default-artifact-root",
                        "file:///mlruns",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                logging.info(
                    f"MLflow server started on http://{self.host}:{self.port}"
                )
                # Wait for MLflow server to start
                time.sleep(5)
            else:
                logging.info("MLflow server is already running.")
        except Exception as e:
            logging.error(f"Failed to start MLflow server: {e}")
            sys.exit(1)

    def is_mlflow_running(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            result = sock.connect_ex((self.host, self.port))
            return result == 0

    def init_mlflow(self):
        self.run_mlflow_server()
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        logging.info(f"MLflow tracking URI set to {self.tracking_uri}")
        logging.info(f"MLflow experiment set to '{self.experiment_name}'")

    def start_run(self, run_name):
        self.run = mlflow.start_run(run_name=run_name)
        logging.info(f"MLflow run '{run_name}' started.")

    def end_run(self):
        if self.run:
            mlflow.end_run()
            logging.info("MLflow run ended.")

    def log_param(self, key, value):
        mlflow.log_param(key, value)

    def log_metric(self, key, value):
        mlflow.log_metric(key, value)

    def log_artifact(self, file_path):
        if os.path.exists(file_path):
            mlflow.log_artifact(file_path)
            logging.info(f"Logged artifact: {file_path}")
        else:
            logging.warning(f"Artifact not found: {file_path}")

    def log_model(self, model, artifact_path):
        try:
            if isinstance(
                model,
                (
                    XGBClassifier,
                    XGBRegressor,
                ),
            ):
                mlflow.xgboost.log_model(model, artifact_path)

            else:
                mlflow.sklearn.log_model(model, artifact_path)
            logging.info(f"Logged model: {artifact_path}")
        except Exception as e:
            logging.error(f"Failed to log model '{artifact_path}': {e}")
