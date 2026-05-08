"""Tests for synthetic patient data generation."""

import numpy as np
import pandas as pd
import pytest

from src.data_synthesis import (
    DIABETES_CONFIG,
    HEART_DISEASE_CONFIG,
    LIVER_CONDITION_CONFIG,
    ConditionConfig,
    PatientDataSynthesizer,
    load_dataset,
    save_dataset,
)


class TestConditionConfig:
    """Test condition configuration data class."""

    def test_config_attributes(self):
        config = ConditionConfig(
            name="test_condition",
            prevalence=0.1,
            risk_factors={"feature_a": 0.5},
            age_factor=0.01,
            gender_bias=0.0,
        )
        assert config.name == "test_condition"
        assert config.prevalence == 0.1
        assert config.risk_factors["feature_a"] == 0.5


class TestPatientDataSynthesizer:
    """Test patient data synthesizer."""

    def test_init_valid(self):
        synth = PatientDataSynthesizer(n_samples=100, random_state=42)
        assert synth.n_samples == 100

    def test_init_invalid(self):
        with pytest.raises(ValueError, match="positive"):
            PatientDataSynthesizer(n_samples=0)

    def test_generate_shape(self):
        synth = PatientDataSynthesizer(n_samples=200, random_state=1)
        data = synth.generate()
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 200
        assert "patient_id" in data.columns

    def test_generate_has_target_cols(self):
        synth = PatientDataSynthesizer(n_samples=100, random_state=2)
        data = synth.generate(include_labels=True)
        assert "diabetes" in data.columns
        assert "heart_disease" in data.columns
        assert "liver_condition" in data.columns

    def test_generate_without_labels(self):
        synth = PatientDataSynthesizer(n_samples=100, random_state=3)
        data = synth.generate(include_labels=False)
        assert "diabetes" not in data.columns
        assert "heart_disease" not in data.columns

    def test_target_binary(self):
        synth = PatientDataSynthesizer(n_samples=500, random_state=4)
        data = synth.generate()
        assert set(data["diabetes"].unique()).issubset({0, 1})
        assert set(data["heart_disease"].unique()).issubset({0, 1})
        assert set(data["liver_condition"].unique()).issubset({0, 1})

    def test_feature_ranges(self):
        synth = PatientDataSynthesizer(n_samples=1000, random_state=5)
        data = synth.generate(include_labels=False)
        assert data["age"].between(18, 95).all()
        assert data["gender"].between(0, 1).all()
        assert data["bmi"].between(15, 55).all()
        assert data["glucose"].between(60, 300).all()

    def test_random_state_reproducibility(self):
        s1 = PatientDataSynthesizer(n_samples=100, random_state=99)
        s2 = PatientDataSynthesizer(n_samples=100, random_state=99)
        d1 = s1.generate(include_labels=False)
        d2 = s2.generate(include_labels=False)
        pd.testing.assert_frame_equal(d1, d2)

    def test_prevalence_reasonable(self):
        synth = PatientDataSynthesizer(n_samples=5000, random_state=7)
        data = synth.generate()
        for col in ["diabetes", "heart_disease", "liver_condition"]:
            prev = data[col].mean()
            assert 0.01 < prev < 0.50, f"{col} prevalence {prev} out of reasonable range"


class TestSaveLoad:
    """Test dataset serialization."""

    def test_save_and_load(self, tmp_path):
        synth = PatientDataSynthesizer(n_samples=50, random_state=8)
        data = synth.generate()
        path = tmp_path / "test_data.csv"
        save_dataset(data, str(path))
        assert path.exists()

        loaded = load_dataset(str(path))
        pd.testing.assert_frame_equal(data, loaded)

    def test_save_empty_raises(self, tmp_path):
        empty = pd.DataFrame()
        with pytest.raises(ValueError, match="empty"):
            save_dataset(empty, str(tmp_path / "empty.csv"))
