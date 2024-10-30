# meta_learning_pipeline/meta_feature_extractor.py

import logging

import numpy as np
import pandas as pd
from pymfe.mfe import MFE
from scipy.stats import entropy, kurtosis, skew
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


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
        meta_features["class_balance"] = y.mean()
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
