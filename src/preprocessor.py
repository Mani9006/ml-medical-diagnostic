"""Data preprocessing pipeline for medical diagnostic features."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Custom transformer for medical-specific feature engineering."""

    def __init__(self) -> None:
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> MedicalFeatureEngineer:
        """Fit the transformer (stateless)."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer medical composite features.

        Adds:
        - bmi_category_score: Normal=0, Overweight=1, Obese=2
        - glucose_hba1c_ratio: Combined glucose metabolism indicator
        - liver_enzyme_ratio: AST/ALT ratio (marker of liver damage pattern)
        - cholesterol_ldl_ratio: LDL proportion of total cholesterol
        - metabolic_risk_score: Composite of BMI, glucose, BP
        - cardio_risk_score: Composite of cholesterol, LDL, BP, age
        - liver_risk_score: Composite of bilirubin, enzymes, alcohol
        """
        X = X.copy()

        # BMI category
        if "bmi" in X.columns:
            bmi = X["bmi"].values
            bmi_cat = np.zeros(len(X))
            bmi_cat[(bmi >= 25) & (bmi < 30)] = 1
            bmi_cat[bmi >= 30] = 2
            X["bmi_category_score"] = bmi_cat

        # Glucose-HbA1c synergy
        if "glucose" in X.columns and "hba1c" in X.columns:
            X["glucose_hba1c_ratio"] = (X["glucose"] / 100.0) * (X["hba1c"] / 5.0)

        # Liver enzyme ratio (De Ritis ratio approximation)
        if "ast" in X.columns and "alt" in X.columns:
            ast = X["ast"].values
            alt = X["alt"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(alt > 0, ast / alt, 0)
            ratio = np.clip(ratio, 0, 10)
            X["liver_enzyme_ratio"] = ratio.round(3)

        # Cholesterol profile
        if "cholesterol" in X.columns and "ldl" in X.columns:
            chol = X["cholesterol"].values
            ldl = X["ldl"].values
            with np.errstate(divide="ignore", invalid="ignore"):
                ldl_ratio = np.where(chol > 0, ldl / chol, 0)
            X["cholesterol_ldl_ratio"] = np.clip(ldl_ratio, 0, 1).round(3)

        # Composite risk scores
        if all(c in X.columns for c in ["bmi", "glucose", "blood_pressure"]):
            X["metabolic_risk_score"] = (
                0.3 * np.clip((X["bmi"] - 25) / 15, 0, 1)
                + 0.4 * np.clip((X["glucose"] - 100) / 150, 0, 1)
                + 0.3 * np.clip((X["blood_pressure"] - 120) / 100, 0, 1)
            ).round(3)

        if all(c in X.columns for c in ["cholesterol", "ldl", "blood_pressure", "age"]):
            X["cardio_risk_score"] = (
                0.25 * np.clip((X["cholesterol"] - 180) / 200, 0, 1)
                + 0.25 * np.clip((X["ldl"] - 100) / 150, 0, 1)
                + 0.25 * np.clip((X["blood_pressure"] - 120) / 100, 0, 1)
                + 0.25 * np.clip((X["age"] - 40) / 50, 0, 1)
            ).round(3)

        if all(c in X.columns for c in ["bilirubin", "ast", "alt", "alcohol_units"]):
            X["liver_risk_score"] = (
                0.25 * np.clip((X["bilirubin"] - 0.3) / 6, 0, 1)
                + 0.25 * np.clip((X["ast"] - 20) / 150, 0, 1)
                + 0.25 * np.clip((X["alt"] - 20) / 150, 0, 1)
                + 0.25 * np.clip(X["alcohol_units"] / 20, 0, 1)
            ).round(3)

        self.feature_names_ = list(X.columns)
        logger.debug("Engineered %d features", len(self.feature_names_))
        return X

    def get_feature_names_out(self, input_features: Any = None) -> List[str]:
        """Return output feature names."""
        return self.feature_names_


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip extreme outliers using IQR method."""

    def __init__(self, factor: float = 3.0) -> None:
        self.factor = factor
        self.bounds_: Dict[str, Tuple[float, float]] = {}

    def fit(self, X: pd.DataFrame, y: Any = None) -> OutlierClipper:
        """Compute IQR-based bounds for each column."""
        for col in X.columns:
            q1 = X[col].quantile(0.25)
            q3 = X[col].quantile(0.75)
            iqr = q3 - q1
            self.bounds_[col] = (q1 - self.factor * iqr, q3 + self.factor * iqr)
        logger.debug("Computed outlier bounds for %d columns", len(self.bounds_))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Clip values to computed bounds."""
        X = X.copy()
        clipped = 0
        for col, (lower, upper) in self.bounds_.items():
            if col in X.columns:
                before = (X[col] < lower) | (X[col] > upper)
                X[col] = X[col].clip(lower, upper)
                clipped += before.sum()
        if clipped > 0:
            logger.info("Clipped %d outlier values", int(clipped))
        return X


class MedicalDataPreprocessor:
    """End-to-end preprocessing pipeline for medical diagnostic data."""

    IDENTIFIER_COLS = ["patient_id"]
    TARGET_COLS = ["diabetes", "heart_disease", "liver_condition"]

    def __init__(
        self,
        scale_method: str = "robust",
        clip_outliers: bool = True,
        engineer_features: bool = True,
    ) -> None:
        """
        Initialize preprocessor.

        Args:
            scale_method: 'standard', 'robust', or 'none'.
            clip_outliers: Whether to clip extreme values.
            engineer_features: Whether to add engineered medical features.
        """
        self.scale_method = scale_method
        self.clip_outliers = clip_outliers
        self.engineer_features = engineer_features
        self.pipeline_: Optional[Pipeline] = None
        self.feature_names_: List[str] = []
        self.raw_feature_names_: List[str] = []

    def _build_pipeline(self) -> Pipeline:
        """Construct sklearn Pipeline with configurable steps."""
        steps: List[Tuple[str, Any]] = []

        if self.engineer_features:
            steps.append(("engineer", MedicalFeatureEngineer()))

        if self.clip_outliers:
            steps.append(("clip", OutlierClipper(factor=3.0)))

        # Imputation
        steps.append(("impute", SimpleImputer(strategy="median")))

        # Scaling
        if self.scale_method == "standard":
            steps.append(("scale", StandardScaler()))
        elif self.scale_method == "robust":
            steps.append(("scale", RobustScaler()))

        return Pipeline(steps)

    def fit_transform(
        self,
        data: pd.DataFrame,
        target_cols: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Fit preprocessor and transform data.

        Returns:
            (features_array, targets_array, feature_names)
        """
        if data.empty:
            raise ValueError("Cannot preprocess empty data")

        target_cols = target_cols or self.TARGET_COLS
        id_cols = self.IDENTIFIER_COLS

        y = None
        available_targets = [c for c in target_cols if c in data.columns]
        if available_targets:
            y = data[available_targets].values
            logger.info("Extracted targets: %s", available_targets)

        # Drop ID and target columns for features
        drop_cols = id_cols + available_targets
        X_df = data.drop(columns=[c for c in drop_cols if c in data.columns])

        # Ensure numeric
        X_df = X_df.select_dtypes(include=[np.number])

        # Store raw column names before pipeline modifies them
        self.raw_feature_names_ = list(X_df.columns)

        self.pipeline_ = self._build_pipeline()
        X_array = self.pipeline_.fit_transform(X_df)

        # Extract final feature names
        if self.engineer_features:
            engineer_step = self.pipeline_.named_steps.get("engineer")
            if isinstance(engineer_step, MedicalFeatureEngineer):
                self.feature_names_ = engineer_step.get_feature_names_out()
            else:
                self.feature_names_ = list(X_df.columns)
        else:
            self.feature_names_ = list(X_df.columns)

        logger.info("Preprocessed shape: X=%s, y=%s", X_array.shape, y.shape if y is not None else None)
        return X_array, y, self.feature_names_

    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline."""
        if self.pipeline_ is None:
            raise RuntimeError("Preprocessor has not been fitted. Call fit_transform first.")

        id_cols = self.IDENTIFIER_COLS
        target_cols = self.TARGET_COLS
        drop_cols = id_cols + [c for c in target_cols if c in data.columns]
        X_df = data.drop(columns=[c for c in drop_cols if c in data.columns])
        X_df = X_df.select_dtypes(include=[np.number])

        # Check that all raw columns are present
        missing = set(self.raw_feature_names_) - set(X_df.columns)
        if missing:
            raise ValueError(f"Missing expected columns: {missing}")

        # Reorder to match training column order
        X_df = X_df[self.raw_feature_names_]
        return self.pipeline_.transform(X_df)

    def get_feature_names(self) -> List[str]:
        """Return processed feature names."""
        return self.feature_names_
