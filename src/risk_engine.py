"""Risk stratification engine for medical diagnostic predictions."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk stratification levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def color(self) -> str:
        """Return display color for this risk level."""
        return {
            RiskLevel.LOW: "#27ae60",
            RiskLevel.MEDIUM: "#f39c12",
            RiskLevel.HIGH: "#e74c3c",
            RiskLevel.CRITICAL: "#8e44ad",
        }[self]

    @property
    def recommendation(self) -> str:
        """Return clinical recommendation for this risk level."""
        return {
            RiskLevel.LOW: "Continue routine monitoring and healthy lifestyle.",
            RiskLevel.MEDIUM: "Schedule follow-up within 3 months. Consider lifestyle modifications.",
            RiskLevel.HIGH: "Schedule follow-up within 1 month. Additional diagnostic tests recommended.",
            RiskLevel.CRITICAL: "Urgent referral to specialist within 1-2 weeks. Comprehensive workup needed.",
        }[self]


@dataclass
class ConditionRisk:
    """Risk assessment for a single condition."""

    condition: str
    probability: float
    risk_level: RiskLevel
    contributing_factors: List[str] = field(default_factory=list)


@dataclass
class PatientRiskProfile:
    """Complete risk profile for a patient."""

    patient_id: Optional[int]
    overall_risk: RiskLevel
    overall_score: float
    condition_risks: List[ConditionRisk]
    composite_scores: Dict[str, float] = field(default_factory=dict)
    summary_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert profile to serializable dict."""
        return {
            "patient_id": self.patient_id,
            "overall_risk": self.overall_risk.value,
            "overall_risk_color": self.overall_risk.color,
            "overall_score": round(self.overall_score, 4),
            "summary": self.summary_text,
            "composite_scores": {k: round(v, 4) for k, v in self.composite_scores.items()},
            "conditions": [
                {
                    "condition": cr.condition,
                    "probability": round(cr.probability, 4),
                    "risk_level": cr.risk_level.value,
                    "risk_color": cr.risk_level.color,
                    "recommendation": cr.risk_level.recommendation,
                    "contributing_factors": cr.contributing_factors,
                }
                for cr in self.condition_risks
            ],
        }


class RiskStratificationEngine:
    """
    Stratifies patient risk based on model predictions and raw features.

    Uses configurable thresholds per condition and combines with
    composite feature-based risk scores for robust assessment.
    """

    CONDITION_NAMES = ["diabetes", "heart_disease", "liver_condition"]

    # Default probability thresholds per condition
    DEFAULT_THRESHOLDS: Dict[str, Tuple[float, float, float]] = {
        "diabetes": (0.20, 0.45, 0.70),
        "heart_disease": (0.15, 0.40, 0.65),
        "liver_condition": (0.10, 0.35, 0.60),
    }

    def __init__(
        self,
        thresholds: Optional[Dict[str, Tuple[float, float, float]]] = None,
        multi_condition_escalation: bool = True,
    ) -> None:
        """
        Initialize risk engine.

        Args:
            thresholds: Dict mapping condition to (low/med, med/high, high/critical) thresholds.
            multi_condition_escalation: If True, having multiple medium+ risks escalates overall.
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.multi_condition_escalation = multi_condition_escalation

    def _probability_to_level(self, prob: float, condition: str) -> RiskLevel:
        """Map a probability to a risk level using condition-specific thresholds."""
        t = self.thresholds.get(condition, (0.2, 0.45, 0.70))
        low_med, med_high, high_crit = t
        if prob < low_med:
            return RiskLevel.LOW
        if prob < med_high:
            return RiskLevel.MEDIUM
        if prob < high_crit:
            return RiskLevel.HIGH
        return RiskLevel.CRITICAL

    def _identify_contributing_factors(
        self, condition: str, features: Dict[str, float]
    ) -> List[str]:
        """Identify feature values that contribute to elevated risk."""
        factors: List[str] = []

        if condition == "diabetes":
            if features.get("glucose", 0) > 140:
                factors.append(f"Elevated glucose: {features['glucose']:.0f} mg/dL")
            if features.get("hba1c", 0) > 6.5:
                factors.append(f"Elevated HbA1c: {features['hba1c']:.1f}%")
            if features.get("bmi", 0) > 30:
                factors.append(f"Obesity: BMI {features['bmi']:.1f}")
            if features.get("family_history", 0) > 0.5:
                factors.append("Family history of diabetes")

        elif condition == "heart_disease":
            if features.get("cholesterol", 0) > 240:
                factors.append(f"High cholesterol: {features['cholesterol']:.0f} mg/dL")
            if features.get("ldl", 0) > 160:
                factors.append(f"Elevated LDL: {features['ldl']:.0f} mg/dL")
            if features.get("blood_pressure", 0) > 140:
                factors.append(f"Hypertension: {features['blood_pressure']:.0f} mmHg")
            if features.get("smoking", 0) > 0.5:
                factors.append("Active smoking")
            if features.get("age", 0) > 55:
                factors.append(f"Age {features['age']:.0f} (risk increases with age)")

        elif condition == "liver_condition":
            if features.get("alcohol_units", 0) > 14:
                factors.append(f"High alcohol: {features['alcohol_units']:.0f} units/week")
            if features.get("ast", 0) > 40:
                factors.append(f"Elevated AST: {features['ast']:.0f} U/L")
            if features.get("alt", 0) > 40:
                factors.append(f"Elevated ALT: {features['alt']:.0f} U/L")
            if features.get("bilirubin", 0) > 1.5:
                factors.append(f"Elevated bilirubin: {features['bilirubin']:.1f} mg/dL")
            if features.get("hepatitis_history", 0) > 0.5:
                factors.append("History of hepatitis")

        return factors if factors else ["No single major risk factor identified"]

    def assess(
        self,
        probabilities: np.ndarray,
        features: Optional[Dict[str, float]] = None,
        patient_id: Optional[int] = None,
    ) -> PatientRiskProfile:
        """
        Assess patient risk from predicted probabilities.

        Args:
            probabilities: Array of positive-class probabilities per condition.
            features: Optional raw feature dict for contributing factor analysis.
            patient_id: Optional patient identifier.

        Returns:
            PatientRiskProfile with full stratification.
        """
        if probabilities.ndim != 1:
            raise ValueError("probabilities must be 1D array")

        features = features or {}
        condition_risks: List[ConditionRisk] = []

        for i, name in enumerate(self.CONDITION_NAMES):
            if i >= len(probabilities):
                break
            prob = float(probabilities[i])
            level = self._probability_to_level(prob, name)
            factors = self._identify_contributing_factors(name, features)
            condition_risks.append(ConditionRisk(name, prob, level, factors))

        # Compute overall score: weighted max + count penalty
        max_prob = max((cr.probability for cr in condition_risks), default=0.0)
        medium_high_count = sum(
            1 for cr in condition_risks if cr.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)
        )
        overall_score = max_prob + 0.05 * medium_high_count

        # Determine overall risk
        if max_prob >= 0.70 or medium_high_count >= 3:
            overall_level = RiskLevel.CRITICAL
        elif max_prob >= 0.45 or medium_high_count >= 2:
            overall_level = RiskLevel.HIGH
        elif max_prob >= 0.20 or medium_high_count >= 1:
            overall_level = RiskLevel.MEDIUM
        else:
            overall_level = RiskLevel.LOW

        if self.multi_condition_escalation and medium_high_count >= 2 and overall_level == RiskLevel.LOW:
            overall_level = RiskLevel.MEDIUM

        # Composite scores from features
        composite: Dict[str, float] = {}
        if features:
            composite["metabolic_risk"] = features.get("metabolic_risk_score", 0.0)
            composite["cardio_risk"] = features.get("cardio_risk_score", 0.0)
            composite["liver_risk"] = features.get("liver_risk_score", 0.0)

        summary = self._generate_summary(condition_risks, overall_level)

        profile = PatientRiskProfile(
            patient_id=patient_id,
            overall_risk=overall_level,
            overall_score=overall_score,
            condition_risks=condition_risks,
            composite_scores=composite,
            summary_text=summary,
        )

        logger.info(
            "Risk assessment for patient %s: overall=%s, score=%.3f",
            patient_id, overall_level.value, overall_score,
        )
        return profile

    def _generate_summary(
        self, risks: List[ConditionRisk], overall: RiskLevel
    ) -> str:
        """Generate human-readable summary text."""
        lines = [f"Overall Risk: {overall.value.upper()}"]
        for cr in sorted(risks, key=lambda x: x.probability, reverse=True):
            lines.append(
                f"  - {cr.condition}: {cr.probability:.1%} ({cr.risk_level.value})"
            )
        lines.append(f"  Recommendation: {overall.recommendation}")
        return "\n".join(lines)

    def batch_assess(
        self,
        probabilities: np.ndarray,
        feature_list: Optional[List[Dict[str, float]]] = None,
    ) -> List[PatientRiskProfile]:
        """
        Assess multiple patients.

        Args:
            probabilities: Array of shape (n_patients, n_conditions).
            feature_list: Optional list of feature dicts per patient.

        Returns:
            List of PatientRiskProfiles.
        """
        if probabilities.ndim != 2:
            raise ValueError("probabilities must be 2D array (n_patients, n_conditions)")

        feature_list = feature_list or [{} for _ in range(len(probabilities))]
        profiles: List[PatientRiskProfile] = []
        for i, probs in enumerate(probabilities):
            profile = self.assess(
                np.array(probs),
                features=feature_list[i] if i < len(feature_list) else None,
                patient_id=i,
            )
            profiles.append(profile)

        logger.info("Batch assessed %d patients", len(profiles))
        return profiles
