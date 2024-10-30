# meta_learning_pipeline/dataset_loader.py

import logging
import os
import sys

import pandas as pd
from sklearn.datasets import load_breast_cancer


class DatasetLoader:
    def __init__(self):
        self.datasets = []

    def load_credit_card_fraud(
        self, filepath="creditcard.csv", sample_size=10000, random_state=42
    ):
        logging.info("Loading Credit Card Fraud Detection dataset...")
        try:
            cc_data = pd.read_csv(filepath)
            if "Class" not in cc_data.columns:
                logging.warning(
                    "Credit Card Fraud Detection dataset missing 'Class' column."
                )
                return
            # Ensure binary classification
            if cc_data["Class"].nunique() != 2:
                logging.warning(
                    "Credit Card Fraud Detection dataset is not binary."
                )
                return
            if len(cc_data) < sample_size:
                logging.warning(
                    f"Requested sample size {sample_size} exceeds dataset size {len(cc_data)}. Using entire dataset."
                )
                sample_size = len(cc_data)
            cc_data_sampled = cc_data.sample(
                n=sample_size, random_state=random_state
            )
            X_cc = cc_data_sampled.drop("Class", axis=1)
            y_cc = cc_data_sampled["Class"]
            self.datasets.append(("Credit Card Fraud", X_cc, y_cc))
            logging.info(
                "Credit Card Fraud Detection dataset loaded and sampled successfully."
            )
        except FileNotFoundError:
            logging.error(
                f"Credit Card Fraud Detection dataset not found at '{filepath}'."
            )
        except Exception as e:
            logging.error(
                f"Error loading Credit Card Fraud Detection dataset: {e}"
            )

    def load_breast_cancer_dataset(self):
        logging.info("Loading Breast Cancer dataset...")
        try:
            breast_cancer = load_breast_cancer()
            X_bc = pd.DataFrame(
                breast_cancer.data, columns=breast_cancer.feature_names
            )
            y_bc = pd.Series(breast_cancer.target)
            # Ensure binary classification
            if y_bc.nunique() != 2:
                logging.warning("Breast Cancer dataset is not binary.")
                return
            self.datasets.append(("Breast Cancer", X_bc, y_bc))
            logging.info("Breast Cancer dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Breast Cancer dataset: {e}")

    def load_diabetes(self, filepath="diabetes.csv"):
        logging.info("Loading Diabetes dataset...")
        try:
            diabetes = pd.read_csv(filepath)
            if "Outcome" not in diabetes.columns:
                logging.warning("Diabetes dataset missing 'Outcome' column.")
                return
            y_diabetes = diabetes["Outcome"]
            # Ensure binary classification
            if y_diabetes.nunique() != 2:
                logging.warning("Diabetes dataset is not binary.")
                return
            X_diabetes = diabetes.drop("Outcome", axis=1)
            self.datasets.append(("Diabetes", X_diabetes, y_diabetes))
            logging.info("Diabetes dataset loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Diabetes dataset not found at '{filepath}'.")
        except Exception as e:
            logging.error(f"Error loading Diabetes dataset: {e}")

    def load_titanic(
        self,
        url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    ):
        logging.info("Downloading and loading Titanic dataset...")
        try:
            titanic = pd.read_csv(url)
            required_columns = ["Survived", "Sex", "Embarked", "Age"]
            if not all(col in titanic.columns for col in required_columns):
                logging.warning("Titanic dataset is missing required columns.")
                return
            titanic = titanic.drop(
                ["PassengerId", "Name", "Ticket", "Cabin"],
                axis=1,
                errors="ignore",
            )
            titanic["Age"].fillna(titanic["Age"].median(), inplace=True)
            titanic["Embarked"].fillna(
                titanic["Embarked"].mode()[0], inplace=True
            )
            titanic = pd.get_dummies(
                titanic, columns=["Sex", "Embarked"], drop_first=True
            )
            if "Survived" not in titanic.columns:
                logging.warning("Titanic dataset missing 'Survived' column.")
                return
            y_titanic = titanic["Survived"]
            # Ensure binary classification
            if y_titanic.nunique() != 2:
                logging.warning("Titanic dataset is not binary.")
                return
            X_titanic = titanic.drop("Survived", axis=1)
            self.datasets.append(("Titanic", X_titanic, y_titanic))
            logging.info(
                "Titanic dataset loaded and preprocessed successfully."
            )
        except Exception as e:
            logging.error(f"Error loading Titanic dataset: {e}")

    def load_adult_income(
        self,
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    ):
        logging.info("Downloading and loading Adult Income dataset...")
        try:
            adult_column_names = [
                "age",
                "workclass",
                "fnlwgt",
                "education",
                "education-num",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "capital-gain",
                "capital-loss",
                "hours-per-week",
                "native-country",
                "income",
            ]
            adult = pd.read_csv(
                url,
                names=adult_column_names,
                na_values=" ?",
                skipinitialspace=True,
            )
            adult.dropna(inplace=True)
            adult = adult.drop("fnlwgt", axis=1, errors="ignore")
            categorical_cols = [
                "workclass",
                "education",
                "marital-status",
                "occupation",
                "relationship",
                "race",
                "sex",
                "native-country",
            ]
            adult = pd.get_dummies(
                adult, columns=categorical_cols, drop_first=True
            )
            if "income" not in adult.columns:
                logging.warning(
                    "Adult Income dataset missing 'income' column."
                )
                return
            adult["income"] = adult["income"].apply(
                lambda x: 1 if x.strip() == ">50K" else 0
            )
            y_adult = adult["income"]
            # Ensure binary classification
            if y_adult.nunique() != 2:
                logging.warning("Adult Income dataset is not binary.")
                return
            X_adult = adult.drop("income", axis=1)
            self.datasets.append(("Adult Income", X_adult, y_adult))
            logging.info(
                "Adult Income dataset loaded and preprocessed successfully."
            )
        except Exception as e:
            logging.error(f"Error loading Adult Income dataset: {e}")

    def load_heart_disease(self, filepath="heart.csv"):
        logging.info("Loading Heart Disease dataset...")
        try:
            heart = pd.read_csv(filepath)
            if "target" not in heart.columns:
                logging.warning(
                    "Heart Disease dataset missing 'target' column."
                )
                return
            y_heart = heart["target"]
            # Ensure binary classification
            if y_heart.nunique() != 2:
                logging.warning("Heart Disease dataset is not binary.")
                return
            X_heart = heart.drop("target", axis=1)
            self.datasets.append(("Heart Disease", X_heart, y_heart))
            logging.info("Heart Disease dataset loaded successfully.")
        except FileNotFoundError:
            logging.error(f"Heart Disease dataset not found at '{filepath}'.")
        except Exception as e:
            logging.error(f"Error loading Heart Disease dataset: {e}")

    def load_all_datasets(self):
        self.load_credit_card_fraud()
        self.load_breast_cancer_dataset()
        self.load_diabetes()
        self.load_titanic()
        self.load_adult_income()
        self.load_heart_disease()
        if not self.datasets:
            logging.error("No valid binary classification datasets loaded.")
            sys.exit(1)
        logging.info("\nAll datasets loaded.\n")
