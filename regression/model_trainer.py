# model_trainer.py

import logging
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib
import pandas as pd
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
