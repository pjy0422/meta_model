# meta_learning_pipeline/pipeline.py

import logging
import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    StackingRegressor,
    VotingRegressor,
)
import mlflow
import matplotlib
import mlflow.sklearn
import mlflow.xgboost
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from dataset_loader import DatasetLoader
from model_trainer import ModelTrainer
from meta_feature_extractor import MetaFeatureExtractor
from meta_model_manager import MetaModelManager
from mlflow_manager import MLflowManager
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
matplotlib.use("Agg")
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
