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
import mlflow.lightgbm
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
import seaborn as sns
from keras.datasets import fashion_mnist
from pymfe.mfe import MFE
from scipy.stats import entropy, kurtosis, skew
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (  # Imported for RandomizedSearch
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

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
        experiment_name="Meta-Model",
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


class DatasetLoader:
    def __init__(self):
        self.datasets = []

    def load_digits_dataset(self):
        logging.info("Loading Digits dataset...")
        try:
            from sklearn.datasets import load_digits

            digits = load_digits()
            # Check if feature names exist, else create generic names
            if hasattr(digits, "feature_names"):
                feature_names = digits.feature_names
            else:
                feature_names = [
                    f"pixel_{i}" for i in range(digits.data.shape[1])
                ]
            X_digits = pd.DataFrame(digits.data, columns=feature_names)
            y_digits = pd.Series(digits.target)
            # Ensure multiclass classification
            if y_digits.nunique() < 3:
                logging.warning("Digits dataset is not multiclass.")
                return
            self.datasets.append(("Digits", X_digits, y_digits))
            logging.info("Digits dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Digits dataset: {e}")

    def load_wine_dataset(self):
        logging.info("Loading Wine dataset...")
        try:
            from sklearn.datasets import load_wine

            wine = load_wine()
            X_wine = pd.DataFrame(wine.data, columns=wine.feature_names)
            y_wine = pd.Series(wine.target)
            # Ensure multiclass classification
            if y_wine.nunique() < 3:
                logging.warning("Wine dataset is not multiclass.")
                return
            self.datasets.append(("Wine", X_wine, y_wine))
            logging.info("Wine dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Wine dataset: {e}")

    def load_fashion_mnist(self, sample_size=10000, random_state=42):
        logging.info("Loading Fashion MNIST dataset...")
        try:
            (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
            X = np.concatenate((X_train, X_test), axis=0)
            y = np.concatenate((y_train, y_test), axis=0)
            y = pd.Series(y)  # Convert to pandas Series before using nunique()

            if y.nunique() < 3:
                logging.warning("Fashion MNIST dataset is not multiclass.")
                return

            # Flatten images
            X = X.reshape(X.shape[0], -1)
            X_fashion = pd.DataFrame(
                X, columns=[f"pixel_{i}" for i in range(X.shape[1])]
            )

            # Sample if necessary
            if len(X_fashion) > sample_size:
                X_fashion = X_fashion.sample(
                    n=sample_size, random_state=random_state
                )
                y_fashion = y[X_fashion.index]
            else:
                y_fashion = y

            self.datasets.append(("Fashion_MNIST", X_fashion, y_fashion))
            logging.info(
                "Fashion MNIST dataset loaded and sampled successfully."
            )
        except Exception as e:
            logging.error(f"Error loading Fashion MNIST dataset: {e}")

    def load_all_datasets(self):
        self.load_digits_dataset()
        self.load_wine_dataset()
        self.load_fashion_mnist()
        if not self.datasets:
            logging.error(
                "No valid multiclass classification datasets loaded."
            )
            sys.exit(1)
        logging.info("\nAll datasets loaded.\n")


class ModelTrainer:
    def __init__(self, models):
        self.models = models
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy="mean")

    def preprocess(self, X):
        # Handle categorical variables
        if isinstance(X, pd.DataFrame):
            categorical_cols = X.select_dtypes(
                include=["object", "category"]
            ).columns
            if len(categorical_cols) > 0:
                X = pd.get_dummies(
                    X, columns=categorical_cols, drop_first=True
                )
        # Impute missing values
        X_imputed = self.imputer.fit_transform(X)
        # Scale features
        X_scaled = self.scaler.fit_transform(X_imputed)
        return X_scaled

    def split_data(self, X, y, random_state=42):
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=random_state
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=0.25,  # 0.25 x 0.8 = 0.2
            stratify=y_temp,
            random_state=random_state,
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

        # Handle class imbalance
        if hasattr(model_clone, "class_weight"):
            model_clone.set_params(class_weight="balanced")
            logging.info(
                f"    - Set class_weight='balanced' for {model_name}."
            )
        elif isinstance(model_clone, XGBClassifier):
            # Calculate scale_pos_weight for each class
            classes = np.unique(y_train)
            class_counts = np.bincount(y_train)
            scale_pos_weights = class_counts[0] / class_counts[1:]
            # Note: XGBoost handles multiclass via 'multi:softprob', so scale_pos_weight is not directly applicable
            logging.info(
                f"    - Note: scale_pos_weight is not directly applicable for multiclass XGBClassifier."
            )

        try:
            model_clone.fit(X_train, y_train)
        except Exception as e:
            logging.error(
                f"Failed to train {model_name} on {dataset_name}: {e}"
            )
            return None

        # Evaluate on validation set
        y_pred_val = model_clone.predict(X_val)
        if hasattr(model_clone, "predict_proba"):
            y_proba_val = model_clone.predict_proba(X_val)
        elif hasattr(model_clone, "decision_function"):
            y_proba_val = model_clone.decision_function(X_val)
            # Scale decision function to [0,1]
            y_proba_val = (y_proba_val - y_proba_val.min()) / (
                y_proba_val.max() - y_proba_val.min() + 1e-10
            )
        else:
            y_proba_val = y_pred_val  # Fallback

        acc_val = accuracy_score(y_val, y_pred_val)
        precision_val = precision_score(
            y_val, y_pred_val, average="macro", zero_division=0
        )
        recall_val = recall_score(
            y_val, y_pred_val, average="macro", zero_division=0
        )
        f1_val = f1_score(y_val, y_pred_val, average="macro", zero_division=0)
        try:
            if len(np.unique(y_val)) < 2:
                auc_val = np.nan
            else:
                auc_val = roc_auc_score(
                    y_val, y_proba_val, multi_class="ovr", average="macro"
                )
        except ValueError:
            auc_val = np.nan

        # Evaluate on test set
        y_pred_test = model_clone.predict(X_test)
        if hasattr(model_clone, "predict_proba"):
            y_proba_test = model_clone.predict_proba(X_test)
        elif hasattr(model_clone, "decision_function"):
            y_proba_test = model_clone.decision_function(X_test)
            y_proba_test = (y_proba_test - y_proba_test.min()) / (
                y_proba_test.max() - y_proba_test.min() + 1e-10
            )
        else:
            y_proba_test = y_pred_test

        acc_test = accuracy_score(y_test, y_pred_test)
        precision_test = precision_score(
            y_test, y_pred_test, average="macro", zero_division=0
        )
        recall_test = recall_score(
            y_test, y_pred_test, average="macro", zero_division=0
        )
        f1_test = f1_score(
            y_test, y_pred_test, average="macro", zero_division=0
        )
        try:
            if len(np.unique(y_test)) < 2:
                auc_test = np.nan
            else:
                auc_test = roc_auc_score(
                    y_test, y_proba_test, multi_class="ovr", average="macro"
                )
        except ValueError:
            auc_test = np.nan

        logging.info(f"    - Validation Accuracy: {acc_val:.4f}")
        logging.info(f"    - Validation Precision: {precision_val:.4f}")
        logging.info(f"    - Validation Recall: {recall_val:.4f}")
        logging.info(f"    - Validation F1-Score: {f1_val:.4f}")
        logging.info(f"    - Validation AUC-ROC: {auc_val:.4f}")
        logging.info(f"    - Test Accuracy: {acc_test:.4f}")
        logging.info(f"    - Test Precision: {precision_test:.4f}")
        logging.info(f"    - Test Recall: {recall_test:.4f}")
        logging.info(f"    - Test F1-Score: {f1_test:.4f}")
        logging.info(f"    - Test AUC-ROC: {auc_test:.4f}")

        # Log metrics
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_accuracy_val", acc_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_precision_val", precision_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_recall_val", recall_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_f1_score_val", f1_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_auc_roc_val", auc_val
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_accuracy_test", acc_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_precision_test", precision_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_recall_test", recall_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_f1_score_test", f1_test
        )
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_auc_roc_test", auc_test
        )

        # Save model
        model_artifact_path = f"{dataset_name}_{model_name}_model"
        mlflow_manager.log_model(model_clone, model_artifact_path)

        performance = {
            "accuracy_val": acc_val,
            "precision_val": precision_val,
            "recall_val": recall_val,
            "f1_score_val": f1_val,
            "auc_roc_val": auc_val,
            "accuracy_test": acc_test,
            "precision_test": precision_test,
            "recall_test": recall_test,
            "f1_score_test": f1_test,
            "auc_roc_test": auc_test,
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
        meta_features["class_balance"] = len(np.unique(y)) / X.shape[0]
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
        mi = mutual_info_classif(
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
        # Define hyperparameter grids for different models with reduced complexity
        param_distributions = {}
        models = {}

        if model_type == "xgb":
            models["xgb"] = XGBRegressor(
                random_state=42, objective="reg:squarederror", n_jobs=4
            )
            param_distributions["xgb"] = {
                "n_estimators": [100, 300, 500],
                "max_depth": [5, 7, 9],
                "subsample": [0.8, 1.0],
                "learning_rate": [0.05, 0.1],
            }
        elif model_type == "rf":
            models["rf"] = RandomForestRegressor(random_state=42, n_jobs=4)
            param_distributions["rf"] = {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 10, 20],
                "max_features": ["auto", "sqrt"],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [1, 2],
                "bootstrap": [True, False],
            }

        elif model_type == "svr":
            models["svr"] = SVR()
            param_distributions["svr"] = {
                "C": [1, 10],
                "gamma": ["scale", "auto", 0.01],
                "kernel": ["linear", "rbf"],
                "degree": [2, 3],  # Relevant for 'poly' kernel
                "epsilon": [0.1, 0.5],
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
                cv=3,  # Reduced number of folds for speed
                n_jobs=4,
                verbose=0,
                n_iter=20,  # Reduced number of iterations
                random_state=41,  # For reproducibility
            )
            randomized_search.fit(X_train, y_train)
            if -randomized_search.best_score_ < best_score:
                best_score = -randomized_search.best_score_
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
            cv=3,  # Reduced number of folds for speed
            scoring="neg_mean_absolute_error",
            n_jobs=4,
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
            "Logistic Regression": LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=200,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=4,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=42,
                n_jobs=4,
            ),
            "Decision Tree": DecisionTreeClassifier(
                criterion="gini",
                splitter="best",
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                random_state=42,
            ),
            "K-Nearest Neighbors": KNeighborsClassifier(
                n_neighbors=5, algorithm="auto", leaf_size=30, p=2, n_jobs=4
            ),
            "AdaBoost": AdaBoostClassifier(
                n_estimators=50, learning_rate=1.0, random_state=42
            ),
            "Bagging Classifier": BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                random_state=42,
                n_jobs=4,
            ),
            "Stacking Classifier": StackingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=10, random_state=42
                        ),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=3, n_jobs=4)),
                ],
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=4,
            ),
            # "XGBoost": XGBClassifier(
            #     use_label_encoder=False,
            #     eval_metric="mlogloss",
            #     learning_rate=0.1,
            #     n_estimators=100,
            #     max_depth=6,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=42,
            # ),
            "Voting Classifier": VotingClassifier(
                estimators=[  # Define base models for voting
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=200,
                            solver="lbfgs",
                            multi_class="multinomial",
                            n_jobs=4,
                        ),
                    ),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=42, n_jobs=4
                        ),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=4)),
                ],
                voting="soft",  # Set to "hard" for majority voting or "soft" for probability-based voting
                n_jobs=4,  # Utilize all available cores
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
                    "accuracy": performance["accuracy_val"],
                    "precision": performance["precision_val"],
                    "recall": performance["recall_val"],
                    "f1_score": performance["f1_score_val"],
                    "auc_roc": performance["auc_roc_val"],
                }
                self.performance_list_val.append(performance_entry_val)

                # Append performance metrics for test set
                performance_entry_test = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy": performance["accuracy_test"],
                    "precision": performance["precision_test"],
                    "recall": performance["recall_test"],
                    "f1_score": performance["f1_score_test"],
                    "auc_roc": performance["auc_roc_test"],
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
            [
                "dataset_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
            ],
            axis=1,
        )
        X_meta_test = meta_dataset_test.drop(
            [
                "dataset_name",
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
            ],
            axis=1,
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
        y_meta_acc_val = meta_dataset_val["accuracy"].values
        y_meta_precision_val = meta_dataset_val["precision"].values
        y_meta_recall_val = meta_dataset_val["recall"].values
        y_meta_f1_val = meta_dataset_val["f1_score"].values
        y_meta_auc_val = (
            meta_dataset_val["auc_roc"]
            .fillna(meta_dataset_val["auc_roc"].mean())
            .values
        )

        # Targets for meta-model testing (using test split)
        y_meta_acc_test = meta_dataset_test["accuracy"].values
        y_meta_precision_test = meta_dataset_test["precision"].values
        y_meta_recall_test = meta_dataset_test["recall"].values
        y_meta_f1_test = meta_dataset_test["f1_score"].values
        y_meta_auc_test = (
            meta_dataset_test["auc_roc"]
            .fillna(meta_dataset_test["auc_roc"].mean())
            .values
        )

        # Train and evaluate meta-models on validation metrics
        logging.info(
            "\nTraining and evaluating meta-models for validation metrics."
        )
        meta_model_manager = self.meta_model_manager

        # Accuracy Meta-Model
        xgb_model_acc_final, error_train_acc, error_test_acc = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_acc_val,
                X_meta_test_processed,
                y_meta_acc_test,
                model_type="xgb",
                metric_name="Accuracy",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # Precision Meta-Model
        (
            xgb_model_precision_final,
            error_train_precision,
            error_test_precision,
        ) = meta_model_manager.train_and_evaluate_meta_model(
            X_meta_val_processed,
            y_meta_precision_val,
            X_meta_test_processed,
            y_meta_precision_test,
            model_type="xgb",
            metric_name="Precision",
            mlflow_manager=self.mlflow_manager,
        )

        # Recall Meta-Model
        xgb_model_recall_final, error_train_recall, error_test_recall = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_recall_val,
                X_meta_test_processed,
                y_meta_recall_test,
                model_type="xgb",
                metric_name="Recall",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # F1-Score Meta-Model
        xgb_model_f1_final, error_train_f1, error_test_f1 = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_f1_val,
                X_meta_test_processed,
                y_meta_f1_test,
                model_type="xgb",
                metric_name="F1_Score",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # AUC-ROC Meta-Model
        xgb_model_auc_final, error_train_auc, error_test_auc = (
            meta_model_manager.train_and_evaluate_meta_model(
                X_meta_val_processed,
                y_meta_auc_val,
                X_meta_test_processed,
                y_meta_auc_test,
                model_type="xgb",
                metric_name="AUC_ROC",
                mlflow_manager=self.mlflow_manager,
            )
        )

        # Save the final meta-models
        if xgb_model_acc_final:
            self.mlflow_manager.log_model(
                xgb_model_acc_final, "final_meta_model_accuracy_xgb"
            )
        if xgb_model_precision_final:
            self.mlflow_manager.log_model(
                xgb_model_precision_final, "final_meta_model_precision_xgb"
            )
        if xgb_model_recall_final:
            self.mlflow_manager.log_model(
                xgb_model_recall_final, "final_meta_model_recall_xgb"
            )
        if xgb_model_f1_final:
            self.mlflow_manager.log_model(
                xgb_model_f1_final, "final_meta_model_f1_score_xgb"
            )
        if xgb_model_auc_final:
            self.mlflow_manager.log_model(
                xgb_model_auc_final, "final_meta_model_auc_roc_xgb"
            )
        logging.info("Meta-models for validation metrics logged to MLflow.")

        # Predicting on validation and test sets
        logging.info(
            "\nPredicting validation and test metrics using meta-models."
        )
        if xgb_model_acc_final:
            predicted_acc_val = xgb_model_acc_final.predict(
                X_meta_val_processed
            )
            predicted_acc_test = xgb_model_acc_final.predict(
                X_meta_test_processed
            )
        else:
            predicted_acc_val = np.zeros_like(y_meta_acc_val)
            predicted_acc_test = np.zeros_like(y_meta_acc_test)

        if xgb_model_precision_final:
            predicted_precision_val = xgb_model_precision_final.predict(
                X_meta_val_processed
            )
            predicted_precision_test = xgb_model_precision_final.predict(
                X_meta_test_processed
            )
        else:
            predicted_precision_val = np.zeros_like(y_meta_precision_val)
            predicted_precision_test = np.zeros_like(y_meta_precision_test)

        if xgb_model_recall_final:
            predicted_recall_val = xgb_model_recall_final.predict(
                X_meta_val_processed
            )
            predicted_recall_test = xgb_model_recall_final.predict(
                X_meta_test_processed
            )
        else:
            predicted_recall_val = np.zeros_like(y_meta_recall_val)
            predicted_recall_test = np.zeros_like(y_meta_recall_test)

        if xgb_model_f1_final:
            predicted_f1_val = xgb_model_f1_final.predict(X_meta_val_processed)
            predicted_f1_test = xgb_model_f1_final.predict(
                X_meta_test_processed
            )
        else:
            predicted_f1_val = np.zeros_like(y_meta_f1_val)
            predicted_f1_test = np.zeros_like(y_meta_f1_test)

        if xgb_model_auc_final:
            predicted_auc_val = xgb_model_auc_final.predict(
                X_meta_val_processed
            )
            predicted_auc_test = xgb_model_auc_final.predict(
                X_meta_test_processed
            )
        else:
            predicted_auc_val = np.zeros_like(y_meta_auc_val)
            predicted_auc_test = np.zeros_like(y_meta_auc_test)

        # Create DataFrames for predictions
        predictions_val_df = meta_dataset_val.copy()
        predictions_val_df["predicted_accuracy"] = predicted_acc_val
        predictions_val_df["predicted_precision"] = predicted_precision_val
        predictions_val_df["predicted_recall"] = predicted_recall_val
        predictions_val_df["predicted_f1_score"] = predicted_f1_val
        predictions_val_df["predicted_auc_roc"] = predicted_auc_val

        predictions_test_df = meta_dataset_test.copy()
        predictions_test_df["predicted_accuracy"] = predicted_acc_test
        predictions_test_df["predicted_precision"] = predicted_precision_test
        predictions_test_df["predicted_recall"] = predicted_recall_test
        predictions_test_df["predicted_f1_score"] = predicted_f1_test
        predictions_test_df["predicted_auc_roc"] = predicted_auc_test

        # Print out comparison of predicted metrics and real metrics
        logging.info(
            "\nComparison of Predicted and Actual Metrics on Validation Set:"
        )
        logging.info(
            predictions_val_df[
                [
                    "dataset_name",
                    "model_name",
                    "accuracy",
                    "predicted_accuracy",
                    "precision",
                    "predicted_precision",
                    "recall",
                    "predicted_recall",
                    "f1_score",
                    "predicted_f1_score",
                    "auc_roc",
                    "predicted_auc_roc",
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
                    "accuracy",
                    "predicted_accuracy",
                    "precision",
                    "predicted_precision",
                    "recall",
                    "predicted_recall",
                    "f1_score",
                    "predicted_f1_score",
                    "auc_roc",
                    "predicted_auc_roc",
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
        predictions_val_df["acc_abs_error"] = abs(
            predictions_val_df["predicted_accuracy"]
            - predictions_val_df["accuracy"]
        )
        predictions_val_df["precision_abs_error"] = abs(
            predictions_val_df["predicted_precision"]
            - predictions_val_df["precision"]
        )
        predictions_val_df["recall_abs_error"] = abs(
            predictions_val_df["predicted_recall"]
            - predictions_val_df["recall"]
        )
        predictions_val_df["f1_abs_error"] = abs(
            predictions_val_df["predicted_f1_score"]
            - predictions_val_df["f1_score"]
        )
        predictions_val_df["auc_abs_error"] = abs(
            predictions_val_df["predicted_auc_roc"]
            - predictions_val_df["auc_roc"]
        )

        # For test set
        predictions_test_df["acc_abs_error"] = abs(
            predictions_test_df["predicted_accuracy"]
            - predictions_test_df["accuracy"]
        )
        predictions_test_df["precision_abs_error"] = abs(
            predictions_test_df["predicted_precision"]
            - predictions_test_df["precision"]
        )
        predictions_test_df["recall_abs_error"] = abs(
            predictions_test_df["predicted_recall"]
            - predictions_test_df["recall"]
        )
        predictions_test_df["f1_abs_error"] = abs(
            predictions_test_df["predicted_f1_score"]
            - predictions_test_df["f1_score"]
        )
        predictions_test_df["auc_abs_error"] = abs(
            predictions_test_df["predicted_auc_roc"]
            - predictions_test_df["auc_roc"]
        )

        mean_acc_abs_error_val_final = predictions_val_df[
            "acc_abs_error"
        ].mean()
        mean_precision_abs_error_val_final = predictions_val_df[
            "precision_abs_error"
        ].mean()
        mean_recall_abs_error_val_final = predictions_val_df[
            "recall_abs_error"
        ].mean()
        mean_f1_abs_error_val_final = predictions_val_df["f1_abs_error"].mean()
        mean_auc_abs_error_val_final = predictions_val_df[
            "auc_abs_error"
        ].mean()

        mean_acc_abs_error_test_final = predictions_test_df[
            "acc_abs_error"
        ].mean()
        mean_precision_abs_error_test_final = predictions_test_df[
            "precision_abs_error"
        ].mean()
        mean_recall_abs_error_test_final = predictions_test_df[
            "recall_abs_error"
        ].mean()
        mean_f1_abs_error_test_final = predictions_test_df[
            "f1_abs_error"
        ].mean()
        mean_auc_abs_error_test_final = predictions_test_df[
            "auc_abs_error"
        ].mean()

        logging.info(
            f"\nMean Absolute Error of Accuracy Meta-Model on Validation Set: {mean_acc_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Precision Meta-Model on Validation Set: {mean_precision_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Recall Meta-Model on Validation Set: {mean_recall_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of F1-Score Meta-Model on Validation Set: {mean_f1_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of AUC-ROC Meta-Model on Validation Set: {mean_auc_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Accuracy Meta-Model on Test Set: {mean_acc_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Precision Meta-Model on Test Set: {mean_precision_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Recall Meta-Model on Test Set: {mean_recall_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of F1-Score Meta-Model on Test Set: {mean_f1_abs_error_test_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of AUC-ROC Meta-Model on Test Set: {mean_auc_abs_error_test_final:.4f}"
        )

        # Log evaluation metrics
        self.mlflow_manager.log_metric(
            "mean_acc_abs_error_val_final", mean_acc_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_precision_abs_error_val_final",
            mean_precision_abs_error_val_final,
        )
        self.mlflow_manager.log_metric(
            "mean_recall_abs_error_val_final", mean_recall_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_f1_abs_error_val_final", mean_f1_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_auc_abs_error_val_final", mean_auc_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_acc_abs_error_test_final", mean_acc_abs_error_test_final
        )
        self.mlflow_manager.log_metric(
            "mean_precision_abs_error_test_final",
            mean_precision_abs_error_test_final,
        )
        self.mlflow_manager.log_metric(
            "mean_recall_abs_error_test_final",
            mean_recall_abs_error_test_final,
        )
        self.mlflow_manager.log_metric(
            "mean_f1_abs_error_test_final", mean_f1_abs_error_test_final
        )
        self.mlflow_manager.log_metric(
            "mean_auc_abs_error_test_final", mean_auc_abs_error_test_final
        )

        # Save mean absolute errors to a CSV file
        mean_errors = {
            "metric": [
                "accuracy",
                "precision",
                "recall",
                "f1_score",
                "auc_roc",
            ],
            "mean_abs_error_val": [
                mean_acc_abs_error_val_final,
                mean_precision_abs_error_val_final,
                mean_recall_abs_error_val_final,
                mean_f1_abs_error_val_final,
                mean_auc_abs_error_val_final,
            ],
            "mean_abs_error_test": [
                mean_acc_abs_error_test_final,
                mean_precision_abs_error_test_final,
                mean_recall_abs_error_test_final,
                mean_f1_abs_error_test_final,
                mean_auc_abs_error_test_final,
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
        metrics = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]

        # Iterate through each dataset
        for dataset in combined_df["dataset_name"].unique():
            dataset_df = combined_df[combined_df["dataset_name"] == dataset]

            # Iterate through each split
            for split in ["Validation", "Test"]:
                split_df = dataset_df[dataset_df["split"] == split]

                # Iterate through each metric
                for metric in metrics:
                    # Prepare data
                    actual_metric = metric
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
                        color = "green" if diff >= 0 else "red"
                        plt.text(
                            idx,
                            max(row[actual_metric], row[predicted_metric])
                            + 0.01
                            * max(row[actual_metric], row[predicted_metric]),
                            f"Δ: {diff:.2f}",
                            ha="center",
                            va="bottom",
                            fontsize=10,
                            color=color,
                            fontweight="bold",
                        )

                    # Customize the plot
                    plt.title(
                        f"Actual vs Predicted {metric.capitalize()} for {dataset} ({split} Set)",
                        fontsize=16,
                    )
                    plt.xlabel("Model", fontsize=14)
                    plt.ylabel(f"{metric.capitalize()}", fontsize=14)
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
                        f"Actual vs Predicted {metric.capitalize()} comparison plot for {dataset} ({split} set) saved and logged to MLflow."
                    )


if __name__ == "__main__":
    try:
        pipeline = MetaLearningPipeline()
        pipeline.start()
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
