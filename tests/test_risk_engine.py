"""Tests for risk stratification engine."""

import numpy as np
import pytest

from src.risk_engine import (
    ConditionRisk,
    PatientRiskProfile,
    RiskLevel,
    RiskStratificationEngine,
)


class TestRiskLevel:
    """Test RiskLevel enum."""

    def test_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"

    def test_colors(self):
        assert RiskLevel.LOW.color == "#27ae60"
        assert RiskLevel.HIGH.color == "#e74c3c"

    def test_recommendations(self):
        assert "routine" in RiskLevel.LOW.recommendation.lower()
        assert "urgent" in RiskLevel.CRITICAL.recommendation.lower()


class TestConditionRisk:
    """Test ConditionRisk dataclass."""

    def test_creation(self):
        cr = ConditionRisk(
            condition="diabetes",
            probability=0.6,
            risk_level=RiskLevel.HIGH,
            contributing_factors=["High glucose"],
        )
        assert cr.condition == "diabetes"
        assert cr.probability == 0.6


class TestPatientRiskProfile:
    """Test PatientRiskProfile."""

    def test_to_dict(self):
        profile = PatientRiskProfile(
            patient_id=1,
            overall_risk=RiskLevel.MEDIUM,
            overall_score=0.35,
            condition_risks=[
                ConditionRisk("diabetes", 0.4, RiskLevel.MEDIUM, ["BMI"]),
            ],
        )
        d = profile.to_dict()
        assert d["patient_id"] == 1
        assert d["overall_risk"] == "medium"
        assert "conditions" in d
        assert len(d["conditions"]) == 1


class TestRiskStratificationEngine:
    """Test risk stratification engine."""

    def test_init(self):
        engine = RiskStratificationEngine()
        assert "diabetes" in engine.thresholds

    def test_assess_low_risk(self):
        engine = RiskStratificationEngine()
        probs = np.array([0.05, 0.08, 0.03])
        profile = engine.assess(probs)
        assert profile.overall_risk == RiskLevel.LOW

    def test_assess_medium_risk(self):
        engine = RiskStratificationEngine()
        # One medium risk only; others low to avoid multi-condition escalation
        probs = np.array([0.30, 0.05, 0.05])
        profile = engine.assess(probs)
        assert profile.overall_risk == RiskLevel.MEDIUM

    def test_assess_high_risk(self):
        engine = RiskStratificationEngine()
        # One high risk; others low to test single-condition high classification
        probs = np.array([0.55, 0.05, 0.05])
        profile = engine.assess(probs)
        assert profile.overall_risk == RiskLevel.HIGH

    def test_assess_critical_risk(self):
        engine = RiskStratificationEngine()
        probs = np.array([0.75, 0.60, 0.55])
        profile = engine.assess(probs)
        assert profile.overall_risk == RiskLevel.CRITICAL

    def test_assess_with_features(self):
        engine = RiskStratificationEngine()
        probs = np.array([0.5, 0.3, 0.2])
        features = {
            "glucose": 150,
            "hba1c": 7.0,
            "bmi": 32,
            "family_history": 1,
            "metabolic_risk_score": 0.5,
        }
        profile = engine.assess(probs, features=features)
        # Should have contributing factors
        diabetes_risk = [cr for cr in profile.condition_risks if cr.condition == "diabetes"][0]
        assert len(diabetes_risk.contributing_factors) > 0

    def test_assess_multi_condition_escalation(self):
        engine = RiskStratificationEngine(multi_condition_escalation=True)
        # Two medium risks should elevate overall
        probs = np.array([0.35, 0.35, 0.05])
        profile = engine.assess(probs)
        assert profile.overall_risk in (RiskLevel.MEDIUM, RiskLevel.HIGH)

    def test_assess_invalid_dimensions(self):
        engine = RiskStratificationEngine()
        with pytest.raises(ValueError, match="1D"):
            engine.assess(np.array([[0.5, 0.3]]))

    def test_batch_assess(self):
        engine = RiskStratificationEngine()
        probs = np.array([
            [0.05, 0.08, 0.03],
            [0.50, 0.30, 0.20],
            [0.75, 0.60, 0.55],
        ])
        profiles = engine.batch_assess(probs)
        assert len(profiles) == 3
        assert profiles[0].overall_risk == RiskLevel.LOW
        assert profiles[1].overall_risk == RiskLevel.CRITICAL
        assert profiles[2].overall_risk == RiskLevel.CRITICAL

    def test_batch_assess_invalid_input(self):
        engine = RiskStratificationEngine()
        with pytest.raises(ValueError, match="2D"):
            engine.batch_assess(np.array([0.5, 0.3, 0.2]))

    def test_custom_thresholds(self):
        custom = {
            "diabetes": (0.10, 0.25, 0.50),
            "heart_disease": (0.10, 0.25, 0.50),
            "liver_condition": (0.10, 0.25, 0.50),
        }
        engine = RiskStratificationEngine(thresholds=custom)
        probs = np.array([0.20, 0.05, 0.05])
        profile = engine.assess(probs)
        # With custom thresholds, 0.20 is medium (was low in defaults)
        diabetes_risk = [cr for cr in profile.condition_risks if cr.condition == "diabetes"][0]
        assert diabetes_risk.risk_level == RiskLevel.MEDIUM

    def test_contributing_factors_heart_disease(self):
        engine = RiskStratificationEngine()
        probs = np.array([0.1, 0.5, 0.1])
        features = {
            "cholesterol": 260,
            "ldl": 180,
            "blood_pressure": 150,
            "smoking": 1,
            "age": 60,
        }
        profile = engine.assess(probs, features=features)
        heart = [cr for cr in profile.condition_risks if cr.condition == "heart_disease"][0]
        factors = heart.contributing_factors
        assert any("cholesterol" in f.lower() for f in factors)

    def test_contributing_factors_liver(self):
        engine = RiskStratificationEngine()
        probs = np.array([0.1, 0.1, 0.5])
        features = {
            "alcohol_units": 20,
            "ast": 60,
            "alt": 55,
            "bilirubin": 2.0,
            "hepatitis_history": 1,
        }
        profile = engine.assess(probs, features=features)
        liver = [cr for cr in profile.condition_risks if cr.condition == "liver_condition"][0]
        factors = liver.contributing_factors
        assert any("alcohol" in f.lower() for f in factors)
