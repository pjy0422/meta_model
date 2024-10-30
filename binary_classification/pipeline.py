# meta_learning_pipeline/pipeline.py

import logging
import os
import sys
import warnings
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset_loader import DatasetLoader
from meta_feature_extractor import MetaFeatureExtractor
from meta_model_manager import MetaModelManager
from mlflow_manager import MLflowManager
from model_trainer import ModelTrainer
from sklearn.ensemble import (
    BaggingClassifier,
    RandomForestClassifier,
    StackingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
warnings.filterwarnings("ignore")


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
                penalty="l2", C=1.0, max_iter=200, solver="lbfgs", n_jobs=-1
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
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
                n_neighbors=5, algorithm="auto", leaf_size=30, p=2, n_jobs=-1
            ),
            "Bagging Classifier": BaggingClassifier(
                estimator=DecisionTreeClassifier(),
                n_estimators=10,
                max_samples=1.0,
                max_features=1.0,
                bootstrap=True,
                random_state=42,
                n_jobs=-1,
            ),
            "Stacking Classifier": StackingClassifier(
                estimators=[
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=10, random_state=42
                        ),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=3, n_jobs=-1)),
                ],
                final_estimator=LogisticRegression(),
                cv=5,
                n_jobs=-1,
            ),
            "XGBoost": XGBClassifier(
                use_label_encoder=False,
                eval_metric="logloss",
                learning_rate=0.1,
                n_estimators=100,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
            ),
            "Voting Classifier": VotingClassifier(
                estimators=[  # Define base models for voting
                    (
                        "lr",
                        LogisticRegression(
                            max_iter=200, solver="lbfgs", n_jobs=-1
                        ),
                    ),
                    (
                        "rf",
                        RandomForestClassifier(
                            n_estimators=100, random_state=42, n_jobs=-1
                        ),
                    ),
                    ("knn", KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
                ],
                voting="soft",  # Set to "hard" for majority voting or "soft" for probability-based voting
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
                    "accuracy": performance["accuracy_val"],
                    "f1_score": performance["f1_score_val"],
                    "auc_roc": performance["auc_roc_val"],
                }
                self.performance_list_val.append(performance_entry_val)

                # Append performance metrics for test set
                performance_entry_test = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "accuracy": performance["accuracy_test"],
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
            ["dataset_name", "accuracy", "f1_score", "auc_roc"], axis=1
        )
        X_meta_test = meta_dataset_test.drop(
            ["dataset_name", "accuracy", "f1_score", "auc_roc"], axis=1
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
        y_meta_f1_val = meta_dataset_val["f1_score"].values
        y_meta_auc_val = (
            meta_dataset_val["auc_roc"]
            .fillna(meta_dataset_val["auc_roc"].mean())
            .values
        )

        # Targets for meta-model testing (using test split)
        y_meta_acc_test = meta_dataset_test["accuracy"].values
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
        predictions_val_df["predicted_f1_score"] = predicted_f1_val
        predictions_val_df["predicted_auc_roc"] = predicted_auc_val

        predictions_test_df = meta_dataset_test.copy()
        predictions_test_df["predicted_accuracy"] = predicted_acc_test
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
        mean_f1_abs_error_val_final = predictions_val_df["f1_abs_error"].mean()
        mean_auc_abs_error_val_final = predictions_val_df[
            "auc_abs_error"
        ].mean()
        mean_acc_abs_error_test_final = predictions_test_df[
            "acc_abs_error"
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
            f"Mean Absolute Error of F1-Score Meta-Model on Validation Set: {mean_f1_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of AUC-ROC Meta-Model on Validation Set: {mean_auc_abs_error_val_final:.4f}"
        )
        logging.info(
            f"Mean Absolute Error of Accuracy Meta-Model on Test Set: {mean_acc_abs_error_test_final:.4f}"
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
            "mean_f1_abs_error_val_final", mean_f1_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_auc_abs_error_val_final", mean_auc_abs_error_val_final
        )
        self.mlflow_manager.log_metric(
            "mean_acc_abs_error_test_final", mean_acc_abs_error_test_final
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
                "f1_score",
                "auc_roc",
            ],
            "mean_abs_error_val": [
                mean_acc_abs_error_val_final,
                mean_f1_abs_error_val_final,
                mean_auc_abs_error_val_final,
            ],
            "mean_abs_error_test": [
                mean_acc_abs_error_test_final,
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
        metrics = ["accuracy", "f1_score", "auc_roc"]

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
