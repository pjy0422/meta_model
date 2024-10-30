# dataset_loader.py

import logging
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_digits, load_wine, load_iris

class DatasetLoader:
    def __init__(self):
        self.datasets = []

    def load_digits_dataset(self):
        logging.info("Loading Digits dataset...")
        try:
            digits = load_digits()
            # Check if feature names exist, else create generic names
            if hasattr(digits, "feature_names"):
                feature_names = digits.feature_names
            else:
                feature_names = [f"pixel_{i}" for i in range(digits.data.shape[1])]
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

    def load_iris_dataset(self):
        logging.info("Loading Iris dataset...")
        try:
            iris = load_iris()
            X_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
            y_iris = pd.Series(iris.target)
            # Ensure multiclass classification
            if y_iris.nunique() < 3:
                logging.warning("Iris dataset is not multiclass.")
                return
            self.datasets.append(("Iris", X_iris, y_iris))
            logging.info("Iris dataset loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading Iris dataset: {e}")




    def load_all_datasets(self):
        self.load_digits_dataset()
        self.load_wine_dataset()
        self.load_iris_dataset()

        if not self.datasets:
            logging.error(
                "No valid multiclass classification datasets loaded."
            )
            sys.exit(1)
        logging.info("\nAll datasets loaded successfully.\n")

        # Debugging: Print the number of loaded datasets
        logging.info(f"Total datasets loaded: {len(self.datasets)}")
        for name, X, y in self.datasets:
            logging.info(f"Dataset: {name}")
            logging.info(f"Features shape: {X.shape}")
            logging.info(f"Target distribution:\n{y.value_counts()}\n")
