import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime

# Set Matplotlib to use the 'Agg' backend before importing pyplot
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
from pymfe.mfe import MFE
from scipy.stats import entropy, kurtosis, skew
from sklearn.base import clone
from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor,
)
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

os.makedirs("results", exist_ok=True)
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/meta_learning_pipeline.log"),
    ],
)

warnings.filterwarnings("ignore")


class MLflowManager:
    def __init__(
        self,
        host="127.0.0.1",
        port=3060,
        experiment_name="Meta-Model-Regression",
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
        import socket

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
                    XGBRegressor,
                ),
            ):
                mlflow.xgboost.log_model(model, artifact_path)
            else:
                mlflow.sklearn.log_model(model, artifact_path)
            logging.info(f"Logged model: {artifact_path}")
        except Exception as e:
            logging.error(f"Failed to log model '{artifact_path}': {e}")


class DatasetLoader:
    def __init__(self):
        self.datasets = []

    def load_california_housing(self):
        print("Loading California Housing dataset...")
        try:
            california = fetch_california_housing(as_frame=True)
            X_ca = california.data
            y_ca = california.target
            self.datasets.append(("California Housing", X_ca, y_ca))
            print("California Housing dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading California Housing dataset: {e}")

    def load_concrete_compressive_strength(self):
        print("Loading Concrete Compressive Strength dataset...")
        try:
            concrete_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
            concrete_column_names = [
                "Cement",
                "BlastFurnaceSlag",
                "FlyAsh",
                "Water",
                "Superplasticizer",
                "CoarseAggregate",
                "FineAggregate",
                "Age",
                "CompressiveStrength",
            ]
            concrete = pd.read_excel(
                concrete_url, header=0, names=concrete_column_names
            )
            X_concrete = concrete.drop("CompressiveStrength", axis=1)
            y_concrete = concrete["CompressiveStrength"]
            self.datasets.append(("Concrete Compressive Strength", X_concrete, y_concrete))
            print("Concrete Compressive Strength dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading Concrete Compressive Strength dataset: {e}")

    def load_energy_efficiency(self):
        print("Loading Energy Efficiency dataset...")
        try:
            energy_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
            energy_column_names = [
                "Relative_Compactness",
                "Surface_Area",
                "Wall_Area",
                "Roof_Area",
                "Overall_Height",
                "Orientation",
                "Glazing_Area",
                "Glazing_Area_Distribution",
                "Heating_Load",
                "Cooling_Load",
            ]
            energy = pd.read_excel(energy_url, header=0, names=energy_column_names)
            X_energy = energy.drop(["Heating_Load", "Cooling_Load"], axis=1)
            y_energy = energy["Heating_Load"]  # Choose "Heating_Load" or "Cooling_Load" as needed
            self.datasets.append(("Energy Efficiency Heating", X_energy, y_energy))
            print("Energy Efficiency dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading Energy Efficiency dataset: {e}")


    def load_auto_mpg(self):
        print("Loading Auto MPG dataset...")
        try:
            auto_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
            auto_columns = [
                "mpg",
                "cylinders",
                "displacement",
                "horsepower",
                "weight",
                "acceleration",
                "model_year",
                "origin",
                "car_name",
            ]
            auto = pd.read_csv(
                auto_url, delim_whitespace=True, names=auto_columns, na_values="?"
            )
            auto.dropna(inplace=True)
            auto = auto.drop("car_name", axis=1)
            # One-hot encode 'origin'
            auto = pd.get_dummies(auto, columns=["origin"], drop_first=True)
            X_auto = auto.drop("mpg", axis=1)
            y_auto = auto["mpg"]
            self.datasets.append(("Auto MPG", X_auto, y_auto))
            print("Auto MPG dataset loaded and preprocessed successfully.")
        except Exception as e:
            print(f"Error loading Auto MPG dataset: {e}")

    def load_all_datasets(self):
        self.load_california_housing()
        # self.load_concrete_compressive_strength()
        self.load_energy_efficiency()
        self.load_auto_mpg()
        if not self.datasets:
            print("No valid regression datasets loaded.")
            sys.exit(1)
        print("\nAll datasets loaded successfully.\n")


class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")

    def preprocess(self, X):
        # Handle categorical variables
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled

    def split_data(self, X, y, random_state=42):
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.4, random_state=random_state
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=random_state
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_and_evaluate(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        model,
        model_name,
        dataset_name,
        mlflow_manager,
    ):
        logging.info(f"  Training {model_name}...")
        # Log model parameters
        mlflow_manager.log_param(
            f"{dataset_name}_{model_name}_n_samples", X_train.shape[0]
        )
        mlflow_manager.log_param(
            f"{dataset_name}_{model_name}_n_features", X_train.shape[1]
        )

        # Clone the model to avoid modifying the original
        model_clone = clone(model)

        try:
            model_clone.fit(X_train, y_train)
        except Exception as e:
            logging.error(
                f"Failed to train {model_name} on {dataset_name}: {e}"
            )
            return None

        # Evaluate on validation set
        y_pred_val = model_clone.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(y_val, y_pred_val)

        # Evaluate on test set
        y_pred_test = model_clone.predict(X_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(y_test, y_pred_test)

        logging.info(f"    - Validation MAE: {mae_val:.4f}")
        logging.info(f"    - Validation MSE: {mse_val:.4f}")
        logging.info(f"    - Validation RMSE: {rmse_val:.4f}")
        logging.info(f"    - Validation R2: {r2_val:.4f}")
        logging.info(f"    - Test MAE: {mae_test:.4f}")
        logging.info(f"    - Test MSE: {mse_test:.4f}")
        logging.info(f"    - Test RMSE: {rmse_test:.4f}")
        logging.info(f"    - Test R2: {r2_test:.4f}")

        # Log metrics
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_mae_val", mae_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_mse_val", mse_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_rmse_val", rmse_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_r2_val", r2_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_mae_test", mae_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_mse_test", mse_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_rmse_test", rmse_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_r2_test", r2_test
        )

        # Save model
        model_artifact_path = f"{dataset_name}_{model_name}_model"
        mlflow_manager.log_model(model_clone, model_artifact_path)

        performance = {
            "mae_val": mae_val,
            "mse_val": mse_val,
            "rmse_val": rmse_val,
            "r2_val": r2_val,
            "mae_test": mae_test,
            "mse_test": mse_test,
            "rmse_test": rmse_test,
            "r2_test": r2_test,
        }

        return performance


class MetaFeatureExtractor:
    def __init__(self):
        pass

    def extract_meta_features(self, X, y):
        # Convert y to a NumPy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y = y.values

        meta_features = {}
        # Basic Meta-Features
        meta_features["n_samples"] = X.shape[0]
        meta_features["n_features"] = X.shape[1]
        meta_features["feature_mean"] = np.mean(X, axis=0).mean()
        meta_features["feature_std"] = np.std(X, axis=0).mean()
        meta_features["coeff_variation"] = (
            np.std(X, axis=0) / (np.mean(X, axis=0) + 1e-10)
        ).mean()

        # PCA
        n_components = min(5, X.shape[1])
        pca = PCA(n_components=n_components)
        pca.fit(X)
        meta_features["pca_explained_variance"] = np.sum(
            pca.explained_variance_ratio_
        )

        # Mutual Information
        mi = mutual_info_regression(
            X, y, discrete_features=False, random_state=42
        )
        meta_features["avg_mutual_info"] = np.mean(mi)

        # Skewness and Kurtosis
        skewness = skew(X, axis=0)
        kurtosis_values = kurtosis(X, axis=0)
        meta_features["avg_skewness"] = np.mean(skewness)
        meta_features["avg_kurtosis"] = np.mean(kurtosis_values)

        # Mean Absolute Correlation
        if X.shape[1] > 1:
            corr_matrix = np.corrcoef(X, rowvar=False)
            mask = np.ones(corr_matrix.shape, dtype=bool)
            np.fill_diagonal(mask, 0)
            abs_corr = np.abs(corr_matrix[mask])
            meta_features["mean_abs_correlation"] = np.mean(abs_corr)
        else:
            meta_features["mean_abs_correlation"] = 0

        # Zero Variance Features
        zero_variance_features = np.sum(np.var(X, axis=0) == 0)
        meta_features["n_zero_variance_features"] = zero_variance_features

        # Variance Statistics
        variances = np.var(X, axis=0)
        meta_features["mean_variance"] = np.mean(variances)
        meta_features["median_variance"] = np.median(variances)

        # Feature Entropy
        feature_entropies = [
            entropy(np.histogram(X[:, i], bins=10)[0] + 1e-10)
            for i in range(X.shape[1])
        ]
        meta_features["mean_feature_entropy"] = np.mean(feature_entropies)

        # Additional Meta-Features using pymfe
        try:
            mfe = MFE()
            mfe.fit(X, y)
            ft = mfe.extract()
            extracted_features = dict(zip(ft[0], ft[1]))
            # Add extracted features to meta_features
            for key, value in extracted_features.items():
                if isinstance(value, (int, float, np.integer, np.float64)):
                    meta_features[key] = value
                elif isinstance(value, (list, np.ndarray)):
                    meta_features[key] = np.mean(value)
        except Exception as e:
            logging.warning(f"Failed to extract additional meta-features: {e}")

        # Handle any potential NaN values by replacing them with zero
        for key, value in meta_features.items():
            if isinstance(value, float) and np.isnan(value):
                meta_features[key] = 0.0

        return meta_features


class MetaModelManager:
    def __init__(self):
        pass

    def train_and_evaluate_meta_model(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        model_type="xgb",
        metric_name="",
        mlflow_manager=None,
    ):
        logging.info(
            f"  Training Meta-Model for {metric_name} using {model_type.upper()}..."
        )
        # Define hyperparameter grids for different models
        param_distributions = {}
        models = {}

        if model_type == "xgb":
            models["xgb"] = XGBRegressor(
                random_state=42, objective="reg:squarederror"
            )
            param_distributions["xgb"] = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [3, 5, 7, 9, 11, 13, 15, 17],
                "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
            }
        elif model_type == "rf":
            models["rf"] = RandomForestRegressor(random_state=42)
            param_distributions["rf"] = {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [None, 10, 20, 30, 40, 50],
                "max_features": ["auto", "sqrt", "log2"],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "bootstrap": [True, False],
            }
        elif model_type == "svr":
            models["svr"] = SVR()
            param_distributions["svr"] = {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                "kernel": ["linear", "rbf", "poly", "sigmoid"],
                "degree": [2, 3, 4],  # Relevant for 'poly' kernel
                "epsilon": [0.1, 0.2, 0.5, 1],
            }
        elif model_type == "knn":
            models["knn"] = KNeighborsRegressor()
            param_distributions["knn"] = {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"],
                "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                "leaf_size": [10, 20, 30, 40, 50],
                "p": [1, 2],
            }
        elif model_type == "ada":
            models["ada"] = AdaBoostRegressor(random_state=42)
            param_distributions["ada"] = {
                "n_estimators": [50, 100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "loss": ["linear", "square", "exponential"],
            }
        else:
            logging.error(
                f"Unsupported model type for meta-model: {model_type}"
            )
            return None, None, None

        best_model = None
        best_score = float("inf")
        best_params = {}
        best_model_name = ""

        for name, model in models.items():
            param_dist = param_distributions[name]
            randomized_search = RandomizedSearchCV(
                model,
                param_distributions=param_dist,
                scoring="neg_mean_absolute_error",
                cv=5,
                n_jobs=-1,
                verbose=0,
                n_iter=50,  # Adjusted number of iterations for efficiency
                random_state=41,  # For reproducibility
            )
            randomized_search.fit(X_train, y_train)
            if randomized_search.best_score_ < best_score:
                best_score = randomized_search.best_score_
                best_model = randomized_search.best_estimator_
                best_params = randomized_search.best_params_
                best_model_name = name

        logging.info(
            f"    - Best {best_model_name.upper()} params: {best_params}"
        )

        # Evaluate on training set using cross-validation
        cv_scores = cross_val_score(
            best_model,
            X_train,
            y_train,
            cv=5,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        error_train = -cv_scores.mean()

        # Evaluate on test set
        pred_test = best_model.predict(X_test)
        error_test = mean_absolute_error(y_test, pred_test)

        logging.info(
            f"    - Mean Absolute Error on Training Set (CV): {error_train:.4f}"
        )
        logging.info(
            f"    - Mean Absolute Error on Test Set: {error_test:.4f}"
        )

        if mlflow_manager:
            mlflow_manager.log_metric(
                f"mean_abs_error_{metric_name}_train_{model_type}", error_train
            )
            mlflow_manager.log_metric(
                f"mean_abs_error_{metric_name}_test_{model_type}", error_test
            )
            mlflow_manager.log_param(
                f"best_params_{metric_name}_{model_type}", best_params
            )

        return best_model, error_train, error_test

    def train_final_meta_model(self, X_meta_scaled, y_meta, model_type="xgb"):
        if model_type == "xgb":
            xgb_model_final = XGBRegressor(
                random_state=42, objective="reg:squarederror"
            )
            xgb_model_final.fit(X_meta_scaled, y_meta)
            return xgb_model_final
        else:
            logging.error(f"Unknown model type: {model_type}")
            return None


class MetaLearningPipeline:
    def __init__(self):
        # Initialize MLflow Manager
        self.mlflow_manager = MLflowManager()
        self.mlflow_manager.init_mlflow()

        # Initialize Dataset Loader
        self.dataset_loader = DatasetLoader()
        self.dataset_loader.load_all_datasets()

        # Define Models

        self.models = {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            "Decision Tree Regressor": DecisionTreeRegressor(
                criterion="absolute_error",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ),
            "SVR": SVR(),
            "K-Nearest Neighbors Regressor": KNeighborsRegressor(
                n_neighbors=5, algorithm="auto", leaf_size=30, p=2, n_jobs=-1
            ),
            "XGBoost Regressor": XGBRegressor(
                objective="reg:squarederror",
                learning_rate=0.1,
                n_estimators=100,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ),
            "AdaBoost Regressor": AdaBoostRegressor(
                n_estimators=50,
                learning_rate=1.0,
                loss="linear",
                random_state=42,
            ),
            "Bagging Regressor": BaggingRegressor(
                estimator=DecisionTreeRegressor(),
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            "Stacking Regressor": StackingRegressor(
                estimators=[
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=10, random_state=42, n_jobs=-1
                        ),
                    ),
                    ("knn", KNeighborsRegressor(n_neighbors=3, n_jobs=-1)),
                ],
                final_estimator=LinearRegression(),
                cv=5,
                n_jobs=-1,
            ),
            "Voting Regressor": VotingRegressor(
                estimators=[  # Define base models for voting
                    (
                        "lr",
                        LinearRegression(),
                    ),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=100, random_state=42, n_jobs=-1
                        ),
                    ),
                    ("knn", KNeighborsRegressor(n_neighbors=5, n_jobs=-1)),
                ],
                n_jobs=-1,  # Utilize all available cores
            ),
        }

        # Initialize Model Trainer
        self.model_trainer = ModelTrainer(self.models)

        # Initialize Meta Feature Extractor
        self.meta_feature_extractor = MetaFeatureExtractor()

        # Initialize Meta Model Manager
        self.meta_model_manager = MetaModelManager()

        # Lists to store meta-data
        self.meta_features_list_val = []
        self.performance_list_val = []
        self.meta_features_list_test = []
        self.performance_list_test = []

    def start(self):
        # Start MLflow run
        run_name = f"META_RUN_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_manager.start_run(run_name=run_name)

        # Iterate over each dataset
        for idx, (dataset_name, X, y) in enumerate(
            self.dataset_loader.datasets, 1
        ):
            logging.info(
                f"\nProcessing {dataset_name} ({idx}/{len(self.dataset_loader.datasets)})..."
            )

            # Preprocess data
            X_scaled = self.model_trainer.preprocess(X)
            X_train, X_val, X_test, y_train, y_val, y_test = (
                self.model_trainer.split_data(X_scaled, y)
            )

            # Extract meta-features from the validation set for meta-model training
            meta_features_val = (
                self.meta_feature_extractor.extract_meta_features(X_val, y_val)
            )
            meta_features_val["dataset_name"] = dataset_name

            # Extract meta-features from the test set for meta-model testing
            meta_features_test = (
                self.meta_feature_extractor.extract_meta_features(
                    X_test, y_test
                )
            )
            meta_features_test["dataset_name"] = dataset_name

            # Iterate over each model
            for model_name, model in self.models.items():
                performance = self.model_trainer.train_and_evaluate(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    X_test,
                    y_test,
                    model,
                    model_name,
                    dataset_name,
                    self.mlflow_manager,
                )

                if performance is None:
                    logging.warning(
                        f"Skipping performance logging for {model_name} on {dataset_name}."
                    )
                    continue

                # Append performance metrics for validation set
                performance_entry_val = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "mae_val": performance["mae_val"],
                    "mse_val": performance["mse_val"],
                    "rmse_val": performance["rmse_val"],
                    "r2_val": performance["r2_val"],
                }
                self.performance_list_val.append(performance_entry_val)

                # Append performance metrics for test set
                performance_entry_test = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "mae_test": performance["mae_test"],
                    "mse_test": performance["mse_test"],
                    "rmse_test": performance["rmse_test"],
                    "r2_test": performance["r2_test"],
                }
                self.performance_list_test.append(performance_entry_test)

                # Append meta-features for validation set
                meta_features_entry_val = meta_features_val.copy()
                meta_features_entry_val["model_name"] = model_name
                self.meta_features_list_val.append(meta_features_entry_val)

                # Append meta-features for test set
                meta_features_entry_test = meta_features_test.copy()
                meta_features_entry_test["model_name"] = model_name
                self.meta_features_list_test.append(meta_features_entry_test)

        # Create meta-dataset for validation set
        meta_features_df_val = pd.DataFrame(self.meta_features_list_val)
        performance_df_val = pd.DataFrame(self.performance_list_val)
        meta_dataset_val = pd.merge(
            meta_features_df_val,
            performance_df_val,
            on=["dataset_name", "model_name"],
        )
        os.makedirs("results", exist_ok=True)
        meta_dataset_val.to_csv("results/meta_dataset_val.csv", index=False)
        logging.info(
            "\nMeta-dataset for validation set created and saved to 'results/meta_dataset_val.csv'."
        )
        self.mlflow_manager.log_artifact("results/meta_dataset_val.csv")

        # Create meta-dataset for test set
        meta_features_df_test = pd.DataFrame(self.meta_features_list_test)
        performance_df_test = pd.DataFrame(self.performance_list_test)
        meta_dataset_test = pd.merge(
            meta_features_df_test,
            performance_df_test,
            on=["dataset_name", "model_name"],
        )
        meta_dataset_test.to_csv("results/meta_dataset_test.csv", index=False)
        logging.info(
            "\nMeta-dataset for test set created and saved to 'results/meta_dataset_test.csv'."
        )
        self.mlflow_manager.log_artifact("results/meta_dataset_test.csv")

        # Prepare data for meta-models
        # Separate dataset meta-features and model_name
        X_meta_val = meta_dataset_val.drop(
            ["dataset_name", "mae_val", "mse_val", "rmse_val", "r2_val"], axis=1
        )
        X_meta_test = meta_dataset_test.drop(
            ["dataset_name", "mae_test", "mse_test", "rmse_test", "r2_test"], axis=1
        )

        # One-Hot Encode model names
        model_names_val = pd.get_dummies(
            X_meta_val["model_name"], prefix="model"
        )
        X_meta_val = X_meta_val.drop("model_name", axis=1)

        model_names_test = pd.get_dummies(
            X_meta_test["model_name"], prefix="model"
        )
        X_meta_test = X_meta_test.drop("model_name", axis=1)

        # Scale the dataset meta-features only
        scaler_meta = StandardScaler()
        X_meta_features_val = scaler_meta.fit_transform(X_meta_val)
        X_meta_features_test = scaler_meta.transform(X_meta_test)

        # Ensure that model_name one-hot encoded features are aligned
        model_names_val, model_names_test = model_names_val.align(
            model_names_test, join="left", axis=1, fill_value=0
        )

        # Concatenate scaled meta-features with unscaled model_name features
        X_meta_val_processed = np.hstack(
            [X_meta_features_val, model_names_val.values]
        )
        X_meta_test_processed = np.hstack(
            [X_meta_features_test, model_names_test.values]
        )

        # Handle NaN values in meta-features
        def handle_nan(array):
            # Assuming array is already numpy and scaled
            array = np.nan_to_num(array, nan=0.0)
            return array

        X_meta_val_processed = handle_nan(X_meta_val_processed)
        X_meta_test_processed = handle_nan(X_meta_test_processed)

        # Convert to numpy arrays if not already
        X_meta_val_processed = np.array(X_meta_val_processed)
        X_meta_test_processed = np.array(X_meta_test_processed)

        # Targets for meta-model training and testing (using validation split)
        y_meta_mae_val = meta_dataset_val["mae_val"].values
        y_meta_mse_val = meta_dataset_val["mse_val"].values
        y_meta_rmse_val = meta_dataset_val["rmse_val"].values
        y_meta_r2_val = meta_dataset_val["r2_val"].values

        # Targets for meta-model testing (using test split)
        y_meta_mae_test = meta_dataset_test["mae_test"].values
        y_meta_mse_test = meta_dataset_test["mse_test"].values
        y_meta_rmse_test = meta_dataset_test["rmse_test"].values
        y_meta_r2_test = meta_dataset_test["r2_test"].values

        # Normalize performance metrics
        logging.info("\nNormalizing performance metrics...")
        scaler_mae = StandardScaler()
        scaler_mse = StandardScaler()
        scaler_rmse = StandardScaler()
        scaler_r2 = StandardScaler()

        # Fit scalers on validation set and transform both validation and test sets
        y_meta_mae_val_scaled = scaler_mae.fit_transform(y_meta_mae_val.reshape(-1, 1)).ravel()
        y_meta_mse_val_scaled = scaler_mse.fit_transform(y_meta_mse_val.reshape(-1, 1)).ravel()
        y_meta_rmse_val_scaled = scaler_rmse.fit_transform(y_meta_rmse_val.reshape(-1, 1)).ravel()
        y_meta_r2_val_scaled = scaler_r2.fit_transform(y_meta_r2_val.reshape(-1, 1)).ravel()

        y_meta_mae_test_scaled = scaler_mae.transform(y_meta_mae_test.reshape(-1, 1)).ravel()
        y_meta_mse_test_scaled = scaler_mse.transform(y_meta_mse_test.reshape(-1, 1)).ravel()
        y_meta_rmse_test_scaled = scaler_rmse.transform(y_meta_rmse_test.reshape(-1, 1)).ravel()
        y_meta_r2_test_scaled = scaler_r2.transform(y_meta_r2_test.reshape(-1, 1)).ravel()

        # Save scalers
        logging.info("Saving and logging scalers...")
        joblib.dump(scaler_mae, "results/scaler_mae.pkl")
        joblib.dump(scaler_mse, "results/scaler_mse.pkl")
        joblib.dump(scaler_rmse, "results/scaler_rmse.pkl")
        joblib.dump(scaler_r2, "results/scaler_r2.pkl")

        self.mlflow_manager.log_artifact("results/scaler_mae.pkl")
        self.mlflow_manager.log_artifact("results/scaler_mse.pkl")
        self.mlflow_manager.log_artifact("results/scaler_rmse.pkl")
        self.mlflow_manager.log_artifact("results/scaler_r2.pkl")
        logging.info("Scalers saved and logged successfully.")

        # Train and evaluate meta-models on normalized validation metrics
        logging.info(
            "\nTraining and evaluating meta-models for normalized validation metrics."
        )
        meta_model_manager = self.meta_model_manager

        # MAE Meta-Model
        xgb_model_mae_final, error_train_mae, error_test_mae = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_mae_val_scaled,
                X_meta_test_processed,
                y_meta_mae_test_scaled,
                model_type="xgb",
                metric_name="MAE",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # MSE Meta-Model
        xgb_model_mse_final, error_train_mse, error_test_mse = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_mse_val_scaled,
                X_meta_test_processed,
                y_meta_mse_test_scaled,
                model_type="xgb",
                metric_name="MSE",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # RMSE Meta-Model
        xgb_model_rmse_final, error_train_rmse, error_test_rmse = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_rmse_val_scaled,
                X_meta_test_processed,
                y_meta_rmse_test_scaled,
                model_type="xgb",
                metric_name="RMSE",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # R2 Meta-Model
        xgb_model_r2_final, error_train_r2, error_test_r2 = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_r2_val_scaled,
                X_meta_test_processed,
                y_meta_r2_test_scaled,
                model_type="xgb",
                metric_name="R2",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # Save the final meta-models
        logging.info("Saving and logging final meta-models...")
        if xgb_model_mae_final:
            self.mlflow_manager.log_model(
                xgb_model_mae_final, "final_meta_model_mae_xgb"
            )
        if xgb_model_mse_final:
            self.mlflow_manager.log_model(
                xgb_model_mse_final, "final_meta_model_mse_xgb"
            )
        if xgb_model_rmse_final:
            self.mlflow_manager.log_model(
                xgb_model_rmse_final, "final_meta_model_rmse_xgb"
            )
        if xgb_model_r2_final:
            self.mlflow_manager.log_model(
                xgb_model_r2_final, "final_meta_model_r2_xgb"
            )
        logging.info("Final meta-models logged to MLflow successfully.")

        # Predicting on validation and test sets
        logging.info(
            "\nPredicting validation and test metrics using meta-models."
        )
        if xgb_model_mae_final:
            predicted_mae_val_scaled = xgb_model_mae_final.predict(
                X_meta_val_processed
            )
            predicted_mae_val = scaler_mae.inverse_transform(predicted_mae_val_scaled.reshape(-1, 1)).ravel()
            predicted_mae_test_scaled = xgb_model_mae_final.predict(
                X_meta_test_processed
            )
            predicted_mae_test = scaler_mae.inverse_transform(predicted_mae_test_scaled.reshape(-1, 1)).ravel()
        else:
            predicted_mae_val = np.zeros_like(y_meta_mae_val)
            predicted_mae_test = np.zeros_like(y_meta_mae_test)

        if xgb_model_mse_final:
            predicted_mse_val_scaled = xgb_model_mse_final.predict(X_meta_val_processed)
            predicted_mse_val = scaler_mse.inverse_transform(predicted_mse_val_scaled.reshape(-1, 1)).ravel()
            predicted_mse_test_scaled = xgb_model_mse_final.predict(
                X_meta_test_processed
            )
            predicted_mse_test = scaler_mse.inverse_transform(predicted_mse_test_scaled.reshape(-1, 1)).ravel()
        else:
            predicted_mse_val = np.zeros_like(y_meta_mse_val)
            predicted_mse_test = np.zeros_like(y_meta_mse_test)

        if xgb_model_rmse_final:
            predicted_rmse_val_scaled = xgb_model_rmse_final.predict(
                X_meta_val_processed
            )
            predicted_rmse_val = scaler_rmse.inverse_transform(predicted_rmse_val_scaled.reshape(-1, 1)).ravel()
            predicted_rmse_test_scaled = xgb_model_rmse_final.predict(
                X_meta_test_processed
            )
            predicted_rmse_test = scaler_rmse.inverse_transform(predicted_rmse_test_scaled.reshape(-1, 1)).ravel()
        else:
            predicted_rmse_val = np.zeros_like(y_meta_rmse_val)
            predicted_rmse_test = np.zeros_like(y_meta_rmse_test)

        if xgb_model_r2_final:
            predicted_r2_val_scaled = xgb_model_r2_final.predict(
                X_meta_val_processed
            )
            predicted_r2_val = scaler_r2.inverse_transform(predicted_r2_val_scaled.reshape(-1, 1)).ravel()
            predicted_r2_test_scaled = xgb_model_r2_final.predict(
                X_meta_test_processed
            )
            predicted_r2_test = scaler_r2.inverse_transform(predicted_r2_test_scaled.reshape(-1, 1)).ravel()
        else:
            predicted_r2_val = np.zeros_like(y_meta_r2_val)
            predicted_r2_test = np.zeros_like(y_meta_r2_test)

        # Create DataFrames for predictions
        predictions_val_df = meta_dataset_val.copy()
        predictions_val_df["predicted_mae"] = predicted_mae_val
        predictions_val_df["predicted_mse"] = predicted_mse_val
        predictions_val_df["predicted_rmse"] = predicted_rmse_val
        predictions_val_df["predicted_r2"] = predicted_r2_val

        predictions_test_df = meta_dataset_test.copy()
        predictions_test_df["predicted_mae"] = predicted_mae_test
        predictions_test_df["predicted_mse"] = predicted_mse_test
        predictions_test_df["predicted_rmse"] = predicted_rmse_test
        predictions_test_df["predicted_r2"] = predicted_r2_test

        # Print out comparison of predicted metrics and real metrics
        logging.info(
            "\nComparison of Predicted and Actual Metrics on Validation Set:"
        )
        logging.info(
            predictions_val_df[
                [
                    "dataset_name",
                    "model_name",
                    "mae_val",
                    "predicted_mae",
                    "mse_val",
                    "predicted_mse",
                    "rmse_val",
                    "predicted_rmse",
                    "r2_val",
                    "predicted_r2",
                ]
            ]
        )

        logging.info(
            "\nComparison of Predicted and Actual Metrics on Test Set:"
        )
        logging.info(
            predictions_test_df[
                [
                    "dataset_name",
                    "model_name",
                    "mae_test",
                    "predicted_mae",
                    "mse_test",
                    "predicted_mse",
                    "rmse_test",
                    "predicted_rmse",
                    "r2_test",
                    "predicted_r2",
                ]
            ]
        )

        # Save predictions
        os.makedirs("results", exist_ok=True)
        predictions_val_df.to_csv(
            "results/meta_model_predictions_val.csv", index=False
        )
        self.mlflow_manager.log_artifact(
            "results/meta_model_predictions_val.csv"
        )
        logging.info(
            "\nValidation predictions saved to 'results/meta_model_predictions_val.csv' and logged to MLflow."
        )

        predictions_test_df.to_csv(
            "results/meta_model_predictions_test.csv", index=False
        )
        self.mlflow_manager.log_artifact(
            "results/meta_model_predictions_test.csv"
        )
        logging.info(
            "\nTest predictions saved to 'results/meta_model_predictions_test.csv' and logged to MLflow."
        )

        # Plot and log comparisons
        self.plot_and_log_comparisons(predictions_val_df, predictions_test_df)

        # Evaluate meta-model performance
        logging.info("\nEvaluating the meta-models' performance.")
        # For validation set
        predictions_val_df["mae_abs_error"] = abs(
            predictions_val_df["predicted_mae"] - predictions_val_df["mae_val"]
        )
        predictions_val_df["mse_abs_error"] = abs(
            predictions_val_df["predicted_mse"] - predictions_val_df["mse_val"]
        )
        predictions_val_df["rmse_abs_error"] = abs(
            predictions_val_df["predicted_rmse"] - predictions_val_df["rmse_val"]
        )
        predictions_val_df["r2_abs_error"] = abs(
            predictions_val_df["predicted_r2"] - predictions_val_df["r2_val"]
        )

        # For test set
        predictions_test_df["mae_abs_error"] = abs(
            predictions_test_df["predicted_mae"] - predictions_test_df["mae_test"]
        )
        predictions_test_df["mse_abs_error"] = abs(
            predictions_test_df["predicted_mse"] - predictions_test_df["mse_test"]
        )
        predictions_test_df["rmse_abs_error"] = abs(
            predictions_test_df["predicted_rmse"] - predictions_test_df["rmse_test"]
        )
        predictions_test_df["r2_abs_error"] = abs(
            predictions_test_df["predicted_r2"] - predictions_test_df["r2_test"]
        )

        # Calculate mean absolute errors
        mean_mae_abs_error_val_final = predictions_val_df[
            "mae_abs_error"
        ].mean()
        mean_mse_abs_error_val_final = predictions_val_df[
            "mse_abs_error"
        ].mean()
        mean_rmse_abs_error_val_final = predictions_val_df[
            "rmse_abs_error"
        ].mean()
        mean_r2_abs_error_val_final = predictions_val_df[
            "r2_abs_error"
        ].mean()

        mean_mae_abs_error_test_final = predictions_test_df[
            "mae_abs_error"
        ].mean()
        mean_mse_abs_error_test_final = predictions_test_df[
            "mse_abs_error"
        ].mean()
        mean_rmse_abs_error_test_final = predictions_test_df[
            "rmse_abs_error"
        ].mean()
        mean_r2_abs_error_test_final = predictions_test_df[
            "r2_abs_error"
        ].mean()

        logging.info(
            f"\nMean Absolute Error of MAE Meta-Model on Validation Set: {mean_mae_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of MSE Meta-Model on Validation Set: {mean_mse_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of RMSE Meta-Model on Validation Set: {mean_rmse_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of R2 Meta-Model on Validation Set: {mean_r2_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of MAE Meta-Model on Test Set: {mean_mae_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of MSE Meta-Model on Test Set: {mean_mse_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of RMSE Meta-Model on Test Set: {mean_rmse_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of R2 Meta-Model on Test Set: {mean_r2_abs_error_test_final:.4f}"
        )

        # Log evaluation metrics
        self.mlflow_manager.log_metric(
            "mean_mae_abs_error_val_final", mean_mae_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_mse_abs_error_val_final", mean_mse_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_rmse_abs_error_val_final", mean_rmse_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_r2_abs_error_val_final", mean_r2_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_mae_abs_error_test_final", mean_mae_abs_error_test_final
        )
        self.mlflow_manager.log_metric(
            "mean_mse_abs_error_test_final", mean_mse_abs_error_test_final
        )
        self.mlflow_manager.log_metric(
            "mean_rmse_abs_error_test_final", mean_rmse_abs_error_test_final
        )
        self.mlflow_manager.log_metric(
            "mean_r2_abs_error_test_final", mean_r2_abs_error_test_final
        )

        # Save mean absolute errors to a CSV file
        mean_errors = {
            "metric": [
                "mae",
                "mse",
                "rmse",
                "r2",
            ],
            "mean_abs_error_val": [
                mean_mae_abs_error_val_final,
                mean_mse_abs_error_val_final,
                mean_rmse_abs_error_val_final,
                mean_r2_abs_error_val_final,
            ],
            "mean_abs_error_test": [
                mean_mae_abs_error_test_final,
                mean_mse_abs_error_test_final,
                mean_rmse_abs_error_test_final,
                mean_r2_abs_error_test_final,
            ],
        }

        mean_errors_df = pd.DataFrame(mean_errors)
        mean_errors_df.to_csv(
            "results/meta_model_mean_errors.csv", index=False
        )
        logging.info(
            "\nMean absolute errors saved to 'results/meta_model_mean_errors.csv'."
        )
        self.mlflow_manager.log_artifact("results/meta_model_mean_errors.csv")

        # End MLflow run
        self.mlflow_manager.end_run()

    def plot_and_log_comparisons(
        self, predictions_val_df, predictions_test_df
    ):
        # Combine validation and test data
        predictions_val_df["split"] = "Validation"
        predictions_test_df["split"] = "Test"
        combined_df = pd.concat(
            [predictions_val_df, predictions_test_df], ignore_index=True
        )

        # Metrics to compare
        metrics = ["mae", "mse", "rmse", "r2"]

        # Iterate through each dataset
        for dataset in combined_df["dataset_name"].unique():
            dataset_df = combined_df[combined_df["dataset_name"] == dataset]

            # Iterate through each split
            for split in ["Validation", "Test"]:
                split_df = dataset_df[dataset_df["split"] == split]

                # Iterate through each metric
                for metric in metrics:
                    # Prepare data
                    actual_metric = f"{metric}_val" if split == "Validation" else f"{metric}_test"
                    predicted_metric = f"predicted_{metric}"

                    # Aggregate data: calculate mean actual and predicted metrics per model
                    merged = (
                        split_df.groupby("model_name")
                        .agg({actual_metric: "mean", predicted_metric: "mean"})
                        .reset_index()
                    )

                    # Reshape the data for seaborn
                    reshaped = merged.melt(
                        id_vars="model_name",
                        value_vars=[actual_metric, predicted_metric],
                        var_name="Metric_Type",
                        value_name="Value",
                    )

                    # Rename metric types for clarity
                    reshaped["Metric_Type"] = reshaped["Metric_Type"].map(
                        {
                            actual_metric: "Actual",
                            predicted_metric: "Predicted",
                        }
                    )

                    # Set up the plot
                    plt.figure(figsize=(12, 8))
                    sns.set(style="whitegrid")

                    # Create the barplot with hue for Actual and Predicted
                    sns.barplot(
                        x="model_name",
                        y="Value",
                        hue="Metric_Type",
                        data=reshaped,
                        palette={"Actual": "skyblue", "Predicted": "salmon"},
                        edgecolor="white",
                        linewidth=1,
                    )

                    # Add difference annotations
                    for idx, row in merged.iterrows():
                        diff = row[predicted_metric] - row[actual_metric]
                        # For MAE, MSE, RMSE: lower is better; for R2: higher is better
                        if metric == "r2":
                            color = "green" if diff >= 0 else "red"
                        else:
                            color = "green" if diff <= 0 else "red"
                        plt.text(
                            idx,
                            max(row[actual_metric], row[predicted_metric])
                            + 0.01 * max(row[actual_metric], row[predicted_metric]),
                            f"Δ: {diff:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color=color,
                            fontweight="bold",
                        )

                    # Customize the plot
                    plt.title(
                        f"Actual vs Predicted {metric.upper()} for {dataset} ({split} Set)",
                        fontsize=16,
                    )
                    plt.xlabel("Model", fontsize=14)
                    plt.ylabel(f"{metric.upper()}", fontsize=14)
                    plt.xticks(rotation=45, ha="right", fontsize=12)
                    plt.legend(title="Metric Type", fontsize=12)
                    plt.tight_layout()
                    os.makedirs("figures", exist_ok=True)
                    # Save the plot
                    plot_filename = (
                        f"figures/{dataset}_{split}_{metric}_comparison.png"
                    )
                    plt.savefig(plot_filename)
                    plt.close()

                    # Log the plot as an artifact in MLflow
                    self.mlflow_manager.log_artifact(plot_filename)

                    logging.info(
                        f"Actual vs Predicted {metric.upper()} comparison plot for {dataset} ({split} set) saved and logged to MLflow."
                    )


if __name__ == "__main__":
    # Configure logging
    os.makedirs("results", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("results/meta_learning_pipeline.log"),
        ],
    )

    try:
        pipeline = MetaLearningPipeline()
        pipeline.start()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
