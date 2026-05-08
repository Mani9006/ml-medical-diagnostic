"""Tests for ensemble model training and prediction."""

import numpy as np
import pandas as pd
import pytest

from src.ensemble_model import MedicalEnsembleClassifier


class TestMedicalEnsembleClassifier:
    """Test ensemble classifier."""

    @pytest.fixture
    def sample_data(self):
        """Generate small synthetic dataset."""
        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 10)
        # Create some correlation with targets
        y = np.zeros((n, 3))
        for i in range(3):
            y[:, i] = (X[:, i] + np.random.randn(n) * 0.5) > 0.5
        return X.astype(np.float64), y.astype(int)

    def test_init_default_params(self):
        model = MedicalEnsembleClassifier(random_state=42)
        assert model.rf_params["random_state"] == 42
        assert model.gb_params["random_state"] == 42

    def test_fit(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        model.fit(X, y)
        assert model.is_fitted_
        assert model.rf_model_ is not None
        assert model.gb_model_ is not None

    def test_fit_shape_mismatch(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        with pytest.raises(ValueError, match="same number"):
            model.fit(X[:-10], y)

    def test_fit_1d_y_raises(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        with pytest.raises(ValueError, match="2D"):
            model.fit(X, y[:, 0])

    def test_predict(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_not_fitted(self, sample_data):
        X, _ = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(X)

    def test_predict_proba(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        model.fit(X, y)
        probs = model.predict_proba(X)
        assert probs.shape == (X.shape[0], 3)
        assert np.all((probs >= 0) & (probs <= 1))

    def test_cross_validate(self, sample_data):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        # Reduce estimators for speed
        model.rf_params["n_estimators"] = 10
        model.gb_params["n_estimators"] = 10
        model.fit(X, y)
        results = model.cross_validate(X, y, cv_folds=3)
        assert "diabetes" in results
        assert "heart_disease" in results
        assert "liver_condition" in results
        for cond, metrics in results.items():
            assert "accuracy" in metrics
            assert "roc_auc" in metrics
            assert 0 <= metrics["accuracy"] <= 1

    def test_feature_importances(self, sample_data):
        X, y = sample_data
        feature_names = [f"feat_{i}" for i in range(X.shape[1])]
        model = MedicalEnsembleClassifier(random_state=1)
        model.rf_params["n_estimators"] = 10
        model.gb_params["n_estimators"] = 10
        model.fit(X, y)
        imp_df = model.feature_importances(feature_names)
        assert isinstance(imp_df, pd.DataFrame)
        assert "feature" in imp_df.columns
        assert "condition" in imp_df.columns
        assert "model" in imp_df.columns
        assert "importance" in imp_df.columns

    def test_save_and_load(self, sample_data, tmp_path):
        X, y = sample_data
        model = MedicalEnsembleClassifier(random_state=1)
        model.rf_params["n_estimators"] = 10
        model.gb_params["n_estimators"] = 10
        model.fit(X, y)

        model_path = tmp_path / "test_model.pkl"
        model.save(str(model_path))
        assert model_path.exists()

        loaded = MedicalEnsembleClassifier()
        loaded.load(str(model_path))
        assert loaded.is_fitted_

        # Predictions should match
        orig_preds = model.predict(X)
        loaded_preds = loaded.predict(X)
        np.testing.assert_array_equal(orig_preds, loaded_preds)
