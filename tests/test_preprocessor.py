"""Tests for data preprocessing pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import (
    MedicalDataPreprocessor,
    MedicalFeatureEngineer,
    OutlierClipper,
)


class TestMedicalFeatureEngineer:
    """Test feature engineering transformer."""

    def test_fit_transform_adds_features(self):
        df = pd.DataFrame({
            "bmi": [22.0, 28.0, 35.0],
            "glucose": [90, 120, 160],
            "blood_pressure": [110, 130, 150],
            "hba1c": [5.0, 6.0, 7.5],
            "ast": [20, 35, 50],
            "alt": [18, 30, 45],
            "cholesterol": [180, 220, 280],
            "ldl": [100, 140, 180],
            "bilirubin": [0.5, 1.0, 2.0],
            "alcohol_units": [2, 8, 15],
            "age": [30, 50, 70],
        })
        eng = MedicalFeatureEngineer()
        result = eng.fit_transform(df)
        assert "bmi_category_score" in result.columns
        assert "glucose_hba1c_ratio" in result.columns
        assert "liver_enzyme_ratio" in result.columns
        assert "metabolic_risk_score" in result.columns
        assert "cardio_risk_score" in result.columns

    def test_bmi_categories(self):
        df = pd.DataFrame({
            "bmi": [18.0, 27.0, 32.0],
            "glucose": [90, 90, 90],
            "blood_pressure": [120, 120, 120],
        })
        eng = MedicalFeatureEngineer()
        result = eng.fit_transform(df)
        assert result["bmi_category_score"][0] == 0  # normal
        assert result["bmi_category_score"][1] == 1  # overweight
        assert result["bmi_category_score"][2] == 2  # obese

    def test_liver_ratio_computation(self):
        df = pd.DataFrame({
            "ast": [20, 40],
            "alt": [20, 20],
        })
        eng = MedicalFeatureEngineer()
        result = eng.fit_transform(df)
        assert result["liver_enzyme_ratio"][0] == 1.0
        assert result["liver_enzyme_ratio"][1] == 2.0


class TestOutlierClipper:
    """Test outlier clipping."""

    def test_clips_extreme_values(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 1000],  # 1000 is outlier
            "b": [10, 11, 12, 13],
        })
        clipper = OutlierClipper(factor=1.5)
        result = clipper.fit_transform(df)
        # The extreme value should be clipped
        assert result["a"].max() < 1000

    def test_leaves_normal_values(self):
        df = pd.DataFrame({
            "a": list(range(20)),
        })
        clipper = OutlierClipper(factor=3.0)
        result = clipper.fit_transform(df)
        # Normal values should be unchanged
        assert result["a"][0] == 0
        assert result["a"][10] == 10


class TestMedicalDataPreprocessor:
    """Test end-to-end preprocessing."""

    def test_fit_transform_returns_arrays(self):
        df = pd.DataFrame({
            "patient_id": range(100),
            "age": np.random.randint(20, 80, 100),
            "gender": np.random.randint(0, 2, 100),
            "bmi": np.random.uniform(20, 35, 100),
            "glucose": np.random.uniform(70, 200, 100),
            "blood_pressure": np.random.uniform(90, 160, 100),
            "insulin": np.random.uniform(5, 50, 100),
            "hba1c": np.random.uniform(4, 10, 100),
            "cholesterol": np.random.uniform(150, 300, 100),
            "ldl": np.random.uniform(80, 180, 100),
            "stress_level": np.random.randint(1, 10, 100),
            "bilirubin": np.random.uniform(0.2, 2.0, 100),
            "albumin": np.random.uniform(3, 5, 100),
            "ast": np.random.uniform(15, 60, 100),
            "alt": np.random.uniform(15, 55, 100),
            "alcohol_units": np.random.uniform(0, 15, 100),
            "exercise_hours": np.random.uniform(0, 10, 100),
            "smoking": np.random.randint(0, 2, 100),
            "family_history": np.random.randint(0, 2, 100),
            "hepatitis_history": np.random.randint(0, 2, 100),
            "diabetes": np.random.randint(0, 2, 100),
            "heart_disease": np.random.randint(0, 2, 100),
            "liver_condition": np.random.randint(0, 2, 100),
        })
        preprocessor = MedicalDataPreprocessor(scale_method="robust")
        X, y, names = preprocessor.fit_transform(df)
        assert isinstance(X, np.ndarray)
        assert isinstance(y, np.ndarray)
        assert len(names) > 0
        assert X.shape[0] == 100
        assert y.shape == (100, 3)

    def test_different_scale_methods(self):
        df = pd.DataFrame({
            "age": [30, 50, 70],
            "bmi": [22, 28, 35],
            "glucose": [90, 120, 150],
            "blood_pressure": [110, 130, 150],
            "diabetes": [0, 1, 0],
        })
        for method in ["standard", "robust", "none"]:
            prep = MedicalDataPreprocessor(scale_method=method)
            X, y, _ = prep.fit_transform(df)
            assert X.shape[0] == 3

    def test_transform_new_data(self):
        train = pd.DataFrame({
            "age": [30, 40, 50, 60],
            "bmi": [22, 25, 30, 35],
            "glucose": [90, 110, 130, 150],
            "blood_pressure": [110, 120, 130, 140],
            "diabetes": [0, 0, 1, 1],
        })
        prep = MedicalDataPreprocessor(scale_method="robust")
        prep.fit_transform(train)

        test = pd.DataFrame({
            "age": [45],
            "bmi": [28],
            "glucose": [115],
            "blood_pressure": [125],
        })
        X_test = prep.transform(test)
        # Raw training had 4 features + engineered features (bmi_category_score, metabolic_risk_score)
        assert X_test.shape[0] == 1
        assert X_test.shape[1] >= 4  # At least as many as raw, likely more with engineering

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="empty"):
            prep = MedicalDataPreprocessor()
            prep.fit_transform(pd.DataFrame())
