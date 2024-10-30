# meta_learning_pipeline/model_trainer.py

import logging
import os
import sys

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor


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
            # Calculate scale_pos_weight
            num_neg = np.sum(y_train == 0)
            num_pos = np.sum(y_train == 1)
            if num_pos == 0:
                scale_pos_weight = 1
            else:
                scale_pos_weight = num_neg / num_pos
            model_clone.set_params(scale_pos_weight=scale_pos_weight)
            logging.info(
                f"    - Set scale_pos_weight={scale_pos_weight:.2f} for XGBClassifier."
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
            y_proba_val = model_clone.predict_proba(X_val)[:, 1]
        elif hasattr(model_clone, "decision_function"):
            y_proba_val = model_clone.decision_function(X_val)
            # Scale decision function to [0,1]
            y_proba_val = (y_proba_val - y_proba_val.min()) / (
                y_proba_val.max() - y_proba_val.min() + 1e-10
            )
        else:
            y_proba_val = y_pred_val  # Fallback

        acc_val = accuracy_score(y_val, y_pred_val)
        f1_val = f1_score(y_val, y_pred_val, zero_division=0)
        try:
            if len(np.unique(y_val)) == 1:
                auc_val = np.nan
            else:
                auc_val = roc_auc_score(y_val, y_proba_val)
        except ValueError:
            auc_val = np.nan

        # Evaluate on test set
        y_pred_test = model_clone.predict(X_test)
        if hasattr(model_clone, "predict_proba"):
            y_proba_test = model_clone.predict_proba(X_test)[:, 1]
        elif hasattr(model_clone, "decision_function"):
            y_proba_test = model_clone.decision_function(X_test)
            y_proba_test = (y_proba_test - y_proba_test.min()) / (
                y_proba_test.max() - y_proba_test.min() + 1e-10
            )
        else:
            y_proba_test = y_pred_test

        acc_test = accuracy_score(y_test, y_pred_test)
        f1_test = f1_score(y_test, y_pred_test, zero_division=0)
        try:
            if len(np.unique(y_test)) == 1:
                auc_test = np.nan
            else:
                auc_test = roc_auc_score(y_test, y_proba_test)
        except ValueError:
            auc_test = np.nan

        logging.info(f"    - Validation Accuracy: {acc_val:.4f}")
        logging.info(f"    - Validation F1-Score: {f1_val:.4f}")
        logging.info(f"    - Validation AUC-ROC: {auc_val:.4f}")
        logging.info(f"    - Test Accuracy: {acc_test:.4f}")
        logging.info(f"    - Test F1-Score: {f1_test:.4f}")
        logging.info(f"    - Test AUC-ROC: {auc_test:.4f}")

        # Log metrics
        mlflow_manager.log_metric(
            f"{dataset_name}_{model_name}_accuracy_val", acc_val
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
            "f1_score_val": f1_val,
            "auc_roc_val": auc_val,
            "accuracy_test": acc_test,
            "f1_score_test": f1_test,
            "auc_roc_test": auc_test,
        }

        return performance
