"""Ensemble model training for multi-condition medical diagnosis."""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.multioutput import MultiOutputClassifier

logger = logging.getLogger(__name__)


class MedicalEnsembleClassifier:
    """
    Multi-label ensemble classifier for medical conditions.

    Uses a heterogeneous ensemble of Random Forest and Gradient Boosting
    models wrapped in MultiOutputClassifier for multi-label prediction.
    """

    CONDITION_NAMES = ["diabetes", "heart_disease", "liver_condition"]

    def __init__(
        self,
        rf_params: Optional[Dict[str, Any]] = None,
        gb_params: Optional[Dict[str, Any]] = None,
        ensemble_weights: Optional[Tuple[float, float]] = None,
        random_state: int = 42,
    ) -> None:
        """
        Initialize the ensemble.

        Args:
            rf_params: Parameters for RandomForestClassifier.
            gb_params: Parameters for GradientBoostingClassifier.
            ensemble_weights: Weights for (RF, GB) averaging. Default equal.
            random_state: Reproducibility seed.
        """
        self.random_state = random_state
        self.rf_params = rf_params or {
            "n_estimators": 200,
            "max_depth": 12,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced_subsample",
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.gb_params = gb_params or {
            "n_estimators": 150,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 10,
            "random_state": random_state,
        }
        self.ensemble_weights = ensemble_weights or (0.5, 0.5)

        self.rf_model_: Optional[MultiOutputClassifier] = None
        self.gb_model_: Optional[MultiOutputClassifier] = None
        self.is_fitted_ = False

    def _create_base_classifiers(self) -> Tuple[MultiOutputClassifier, MultiOutputClassifier]:
        """Create wrapped multi-output classifiers."""
        rf = MultiOutputClassifier(RandomForestClassifier(**self.rf_params))
        gb = MultiOutputClassifier(GradientBoostingClassifier(**self.gb_params))
        return rf, gb

    def fit(self, X: np.ndarray, y: np.ndarray) -> MedicalEnsembleClassifier:
        """
        Fit both ensemble members on the full training data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples, n_conditions).
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        if y.ndim != 2:
            raise ValueError("y must be 2D array of shape (n_samples, n_conditions)")

        self.rf_model_, self.gb_model_ = self._create_base_classifiers()

        logger.info("Training Random Forest ensemble member...")
        self.rf_model_.fit(X, y)

        logger.info("Training Gradient Boosting ensemble member...")
        self.gb_model_.fit(X, y)

        self.is_fitted_ = True
        logger.info("Ensemble fitting complete. Features: %d, Conditions: %d", X.shape[1], y.shape[1])
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels using averaged probability threshold at 0.5."""
        probs = self.predict_proba(X)
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities by weighted ensemble averaging.

        Returns:
            Array of shape (n_samples, n_conditions) with positive-class probabilities.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        rf_probs = self._extract_positive_proba(self.rf_model_, X)
        gb_probs = self._extract_positive_proba(self.gb_model_, X)

        w_rf, w_gb = self.ensemble_weights
        ensemble = w_rf * rf_probs + w_gb * gb_probs
        return ensemble

    @staticmethod
    def _extract_positive_proba(
        model: MultiOutputClassifier, X: np.ndarray
    ) -> np.ndarray:
        """Extract P(class=1) from each multi-output estimator's predict_proba."""
        probas = model.predict_proba(X)
        # probas is list of arrays, one per output
        positive = np.stack([p[:, 1] for p in probas], axis=1)
        return positive

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stratified cross-validation per condition.

        Uses the first condition for stratification; this is a pragmatic
        simplification for multi-label data.

        Returns:
            Nested dict of metrics per condition.
        """
        logger.info("Running %d-fold cross-validation...", cv_folds)
        # Use KFold for multi-label since stratification is not natively supported
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        rf, gb = self._create_base_classifiers()

        rf_probs = cross_val_predict(
            rf, X, y, cv=kf, method="predict_proba"
        )
        gb_probs = cross_val_predict(
            gb, X, y, cv=kf, method="predict_proba"
        )

        # Extract positive class probabilities
        rf_pos = np.stack([rf_probs[i][:, 1] for i in range(len(rf_probs))], axis=1)
        gb_pos = np.stack([gb_probs[i][:, 1] for i in range(len(gb_probs))], axis=1)

        w_rf, w_gb = self.ensemble_weights
        ensemble_probs = w_rf * rf_pos + w_gb * gb_pos
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)

        results: Dict[str, Dict[str, float]] = {}
        for i, name in enumerate(self.CONDITION_NAMES):
            if i >= y.shape[1]:
                break
            results[name] = {
                "accuracy": accuracy_score(y[:, i], ensemble_preds[:, i]),
                "precision": precision_score(y[:, i], ensemble_preds[:, i], zero_division=0),
                "recall": recall_score(y[:, i], ensemble_preds[:, i], zero_division=0),
                "f1": f1_score(y[:, i], ensemble_preds[:, i], zero_division=0),
                "roc_auc": roc_auc_score(y[:, i], ensemble_probs[:, i]),
            }

        logger.info("Cross-validation complete")
        return results

    def feature_importances(
        self, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Aggregate feature importances across ensemble members and conditions.

        Returns:
            DataFrame with importance per feature per model/condition.
        """
        if not self.is_fitted_:
            raise RuntimeError("Model must be fitted first")

        records: List[Dict[str, Any]] = []

        # Random Forest importances
        for j, estimator in enumerate(self.rf_model_.estimators_):
            if j >= len(self.CONDITION_NAMES):
                break
            cond = self.CONDITION_NAMES[j]
            importances = estimator.feature_importances_
            for fi, fn in enumerate(feature_names[: len(importances)]):
                records.append({
                    "feature": fn,
                    "condition": cond,
                    "model": "random_forest",
                    "importance": importances[fi],
                })

        # Gradient Boosting importances
        for j, estimator in enumerate(self.gb_model_.estimators_):
            if j >= len(self.CONDITION_NAMES):
                break
            cond = self.CONDITION_NAMES[j]
            importances = estimator.feature_importances_
            for fi, fn in enumerate(feature_names[: len(importances)]):
                records.append({
                    "feature": fn,
                    "condition": cond,
                    "model": "gradient_boosting",
                    "importance": importances[fi],
                })

        return pd.DataFrame(records)

    def save(self, path: str | Path) -> None:
        """Serialize ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "rf": self.rf_model_,
                "gb": self.gb_model_,
                "weights": self.ensemble_weights,
                "is_fitted": self.is_fitted_,
                "rf_params": self.rf_params,
                "gb_params": self.gb_params,
            }, f)
        logger.info("Saved ensemble model to %s", path)

    def load(self, path: str | Path) -> MedicalEnsembleClassifier:
        """Deserialize ensemble from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.rf_model_ = data["rf"]
        self.gb_model_ = data["gb"]
        self.ensemble_weights = data["weights"]
        self.is_fitted_ = data["is_fitted"]
        self.rf_params = data["rf_params"]
        self.gb_params = data["gb_params"]
        logger.info("Loaded ensemble model from %s", path)
        return self
