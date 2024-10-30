# meta_model_manager.py

import logging
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
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
