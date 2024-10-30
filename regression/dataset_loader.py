# dataset_loader.py

import os
import sys
import pandas as pd
from sklearn.datasets import fetch_california_housing, fetch_openml

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
        # self.load_concrete_compressive_strength()  # Uncomment if needed
        self.load_energy_efficiency()
        self.load_auto_mpg()
        if not self.datasets:
            print("No valid regression datasets loaded.")
            sys.exit(1)
        print("\nAll datasets loaded successfully.\n")
