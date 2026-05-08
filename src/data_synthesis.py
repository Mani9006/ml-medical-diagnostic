"""Synthetic patient data generation with realistic correlations."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEED = 42
np.random.seed(SEED)


@dataclass
class ConditionConfig:
    """Configuration for a medical condition's synthetic data profile."""

    name: str
    prevalence: float
    risk_factors: Dict[str, float]
    age_factor: float
    gender_bias: float  # positive = more common in males, negative = females


# Pre-defined condition configurations based on approximate epidemiology
DIABETES_CONFIG = ConditionConfig(
    name="diabetes",
    prevalence=0.11,
    risk_factors={
        "bmi": 0.08,
        "glucose": 0.35,
        "blood_pressure": 0.05,
        "insulin": 0.25,
        "hba1c": 0.30,
        "family_history": 0.40,
        "exercise_hours": -0.10,
        "smoking": 0.15,
    },
    age_factor=0.02,
    gender_bias=0.0,
)

HEART_DISEASE_CONFIG = ConditionConfig(
    name="heart_disease",
    prevalence=0.14,
    risk_factors={
        "cholesterol": 0.30,
        "blood_pressure": 0.25,
        "bmi": 0.12,
        "smoking": 0.40,
        "exercise_hours": -0.15,
        "family_history": 0.35,
        "stress_level": 0.20,
        "ldl": 0.28,
    },
    age_factor=0.04,
    gender_bias=0.15,
)

LIVER_CONDITION_CONFIG = ConditionConfig(
    name="liver_condition",
    prevalence=0.07,
    risk_factors={
        "bmi": 0.10,
        "alcohol_units": 0.45,
        "bilirubin": 0.30,
        "albumin": -0.25,
        "ast": 0.35,
        "alt": 0.35,
        "smoking": 0.10,
        "hepatitis_history": 0.50,
    },
    age_factor=0.01,
    gender_bias=0.05,
)

ALL_CONDITIONS: List[ConditionConfig] = [
    DIABETES_CONFIG,
    HEART_DISEASE_CONFIG,
    LIVER_CONDITION_CONFIG,
]


class PatientDataSynthesizer:
    """Generates synthetic patient records with realistic feature correlations."""

    # Feature ranges for validation
    FEATURE_RANGES: Dict[str, Tuple[float, float]] = {
        "age": (18, 95),
        "gender": (0, 1),  # 0 = female, 1 = male
        "bmi": (15.0, 55.0),
        "glucose": (60, 300),
        "blood_pressure": (80, 220),
        "insulin": (5, 100),
        "hba1c": (4.0, 14.0),
        "family_history": (0, 1),
        "exercise_hours": (0, 14),
        "smoking": (0, 1),
        "cholesterol": (120, 450),
        "stress_level": (1, 10),
        "ldl": (60, 250),
        "alcohol_units": (0, 40),
        "bilirubin": (0.1, 8.0),
        "albumin": (2.0, 5.5),
        "ast": (10, 200),
        "alt": (10, 200),
        "hepatitis_history": (0, 1),
    }

    def __init__(self, n_samples: int = 5000, random_state: int | None = None) -> None:
        """
        Initialize the synthesizer.

        Args:
            n_samples: Number of patient records to generate.
            random_state: Optional seed for reproducibility.
        """
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        self.n_samples = n_samples
        self.rng = np.random.RandomState(random_state)
        logger.info("Initialized PatientDataSynthesizer for %d samples", n_samples)

    def _generate_base_features(self) -> pd.DataFrame:
        """Generate base demographic and lifestyle features."""
        age = self.rng.normal(52, 16, self.n_samples)
        age = np.clip(age, *self.FEATURE_RANGES["age"])

        gender = self.rng.binomial(1, 0.51, self.n_samples)

        # BMI correlated with age and gender
        bmi = (
            self.rng.normal(26, 5, self.n_samples)
            + 0.05 * age
            + 1.2 * gender
        )
        bmi = np.clip(bmi, *self.FEATURE_RANGES["bmi"])

        # Exercise inversely correlated with BMI
        exercise_hours = (
            self.rng.gamma(2.5, 1.2, self.n_samples)
            - 0.08 * bmi
            + 0.3 * gender
        )
        exercise_hours = np.clip(exercise_hours, *self.FEATURE_RANGES["exercise_hours"])

        # Smoking correlated with age and gender
        smoking_prob = 1 / (1 + np.exp(-(-2.5 + 0.02 * age + 0.4 * gender)))
        smoking = self.rng.binomial(1, smoking_prob)

        # Family history ~20% prevalence
        family_history = self.rng.binomial(1, 0.20, self.n_samples)

        # Stress level correlated with age and smoking
        stress_level = (
            self.rng.poisson(5, self.n_samples)
            + 0.5 * smoking
            + 0.02 * age
        )
        stress_level = np.clip(stress_level, *self.FEATURE_RANGES["stress_level"])

        data = pd.DataFrame({
            "patient_id": range(self.n_samples),
            "age": age.round(1),
            "gender": gender,
            "bmi": bmi.round(2),
            "exercise_hours": exercise_hours.round(2),
            "smoking": smoking,
            "family_history": family_history,
            "stress_level": stress_level.astype(int),
        })

        logger.debug("Generated %d base feature records", self.n_samples)
        return data

    def _generate_diabetes_features(self, base: pd.DataFrame) -> pd.DataFrame:
        """Generate diabetes-related biomarkers."""
        age = base["age"].values
        bmi = base["bmi"].values
        exercise = base["exercise_hours"].values
        family_hist = base["family_history"].values
        smoking = base["smoking"].values

        glucose = (
            self.rng.normal(95, 12, self.n_samples)
            + 0.8 * bmi
            - 1.5 * exercise
            + 3.0 * family_hist
            + 5.0 * smoking
            + 0.1 * age
        )
        glucose = np.clip(glucose, *self.FEATURE_RANGES["glucose"])

        hba1c = (
            self.rng.normal(5.0, 0.5, self.n_samples)
            + 0.03 * (glucose - 95)
            + 0.02 * bmi
            - 0.05 * exercise
        )
        hba1c = np.clip(hba1c, *self.FEATURE_RANGES["hba1c"])

        insulin = (
            self.rng.lognormal(2.8, 0.6, self.n_samples)
            + 0.15 * bmi
            - 0.8 * exercise
            + 2.0 * family_hist
        )
        insulin = np.clip(insulin, *self.FEATURE_RANGES["insulin"])

        blood_pressure = (
            self.rng.normal(120, 15, self.n_samples)
            + 0.4 * bmi
            + 0.15 * age
            + 5.0 * smoking
        )
        blood_pressure = np.clip(blood_pressure, *self.FEATURE_RANGES["blood_pressure"])

        return pd.DataFrame({
            "glucose": glucose.round(1),
            "hba1c": hba1c.round(2),
            "insulin": insulin.round(2),
            "blood_pressure": blood_pressure.round(1),
        })

    def _generate_heart_features(self, base: pd.DataFrame) -> pd.DataFrame:
        """Generate heart disease-related biomarkers."""
        age = base["age"].values
        bmi = base["bmi"].values
        exercise = base["exercise_hours"].values
        family_hist = base["family_history"].values
        smoking = base["smoking"].values
        stress = base["stress_level"].values

        cholesterol = (
            self.rng.normal(190, 35, self.n_samples)
            + 1.2 * bmi
            + 0.4 * age
            + 15.0 * smoking
            + 3.0 * stress
            + 20.0 * family_hist
        )
        cholesterol = np.clip(cholesterol, *self.FEATURE_RANGES["cholesterol"])

        ldl = (
            self.rng.normal(110, 28, self.n_samples)
            + 0.6 * bmi
            + 0.25 * age
            + 8.0 * smoking
            + 0.5 * (cholesterol - 190)
        )
        ldl = np.clip(ldl, *self.FEATURE_RANGES["ldl"])

        # Blood pressure shared with diabetes but add heart-specific influence
        # We'll return additional heart-specific features only
        return pd.DataFrame({
            "cholesterol": cholesterol.round(1),
            "ldl": ldl.round(1),
        })

    def _generate_liver_features(self, base: pd.DataFrame) -> pd.DataFrame:
        """Generate liver condition-related biomarkers."""
        age = base["age"].values
        bmi = base["bmi"].values
        smoking = base["smoking"].values

        # Alcohol units correlated with age, gender, smoking
        alcohol_units = (
            self.rng.gamma(1.8, 2.5, self.n_samples)
            + 0.04 * age
            + 0.8 * smoking
        )
        alcohol_units = np.clip(alcohol_units, *self.FEATURE_RANGES["alcohol_units"])

        # Hepatitis history ~6% prevalence, higher with age
        hep_prob = 1 / (1 + np.exp(-(-3.0 + 0.02 * age)))
        hepatitis_history = self.rng.binomial(1, hep_prob)

        # Bilirubin elevated with alcohol and hepatitis
        bilirubin = (
            self.rng.lognormal(-0.5, 0.5, self.n_samples)
            + 0.04 * alcohol_units
            + 0.5 * hepatitis_history
        )
        bilirubin = np.clip(bilirubin, *self.FEATURE_RANGES["bilirubin"])

        # Albumin lower with liver issues
        albumin = (
            self.rng.normal(4.3, 0.5, self.n_samples)
            - 0.015 * alcohol_units
            - 0.3 * hepatitis_history
            - 0.005 * age
        )
        albumin = np.clip(albumin, *self.FEATURE_RANGES["albumin"])

        # AST/ALT liver enzymes
        ast = (
            self.rng.lognormal(3.2, 0.4, self.n_samples)
            + 0.8 * alcohol_units
            + 8.0 * hepatitis_history
            + 0.1 * bmi
        )
        ast = np.clip(ast, *self.FEATURE_RANGES["ast"])

        alt = (
            self.rng.lognormal(3.0, 0.45, self.n_samples)
            + 0.9 * alcohol_units
            + 10.0 * hepatitis_history
            + 0.15 * bmi
        )
        alt = np.clip(alt, *self.FEATURE_RANGES["alt"])

        return pd.DataFrame({
            "alcohol_units": alcohol_units.round(1),
            "hepatitis_history": hepatitis_history,
            "bilirubin": bilirubin.round(2),
            "albumin": albumin.round(2),
            "ast": ast.round(1),
            "alt": alt.round(1),
        })

    def _assign_conditions(self, features: pd.DataFrame) -> pd.DataFrame:
        """Assign condition labels using logistic threshold with risk factor correlations."""
        labels = pd.DataFrame({"patient_id": features["patient_id"]})

        for config in ALL_CONDITIONS:
            logits = np.zeros(self.n_samples)

            # Age contribution
            age_vals = features["age"].values
            logits += config.age_factor * (age_vals - 50)

            # Gender bias
            gender_vals = features["gender"].values
            logits += config.gender_bias * (gender_vals - 0.5)

            # Risk factors
            for feature, weight in config.risk_factors.items():
                if feature in features.columns:
                    vals = features[feature].values
                    # Normalize to ~zero mean for logistic computation
                    normalized = (vals - np.mean(vals)) / (np.std(vals) + 1e-6)
                    logits += weight * normalized

            # Adjust prevalence to target
            prob = 1 / (1 + np.exp(-logits))

            # Calibrate probability to match target prevalence
            target_prev = config.prevalence
            current_median = np.median(prob)
            if current_median > 0:
                calibration = target_prev / (current_median + 1e-6)
                prob = np.clip(prob * calibration, 0, 1)

            labels[config.name] = self.rng.binomial(1, prob)

        # Add a small fraction of comorbid cases by ensuring some overlap
        comorbid_idx = self.rng.choice(
            self.n_samples,
            size=int(self.n_samples * 0.03),
            replace=False,
        )
        for idx in comorbid_idx:
            # Make this patient positive for all conditions with 60% chance each
            for config in ALL_CONDITIONS:
                if self.rng.random() < 0.6:
                    labels.at[idx, config.name] = 1

        logger.info(
            "Condition prevalences: diabetes=%.3f, heart_disease=%.3f, liver_condition=%.3f",
            labels["diabetes"].mean(),
            labels["heart_disease"].mean(),
            labels["liver_condition"].mean(),
        )
        return labels

    def generate(self, include_labels: bool = True) -> pd.DataFrame:
        """
        Generate complete synthetic patient dataset.

        Args:
            include_labels: If True, include condition label columns.

        Returns:
            DataFrame with patient features and optionally labels.
        """
        base = self._generate_base_features()
        diabetes = self._generate_diabetes_features(base)
        heart = self._generate_heart_features(base)
        liver = self._generate_liver_features(base)

        features = pd.concat([base, diabetes, heart, liver], axis=1)

        if include_labels:
            labels = self._assign_conditions(features)
            data = pd.merge(features, labels, on="patient_id")
        else:
            data = features

        # Reorder columns logically
        col_order = [
            "patient_id", "age", "gender",
            "bmi", "glucose", "blood_pressure", "insulin", "hba1c",
            "cholesterol", "ldl", "stress_level",
            "bilirubin", "albumin", "ast", "alt",
            "alcohol_units", "exercise_hours",
            "smoking", "family_history", "hepatitis_history",
        ]
        if include_labels:
            col_order += ["diabetes", "heart_disease", "liver_condition"]

        data = data[[c for c in col_order if c in data.columns]]
        logger.info("Generated dataset with shape %s", data.shape)
        return data


def save_dataset(data: pd.DataFrame, path: str) -> None:
    """Save dataset to CSV with validation."""
    if data.empty:
        raise ValueError("Cannot save empty dataset")
    data.to_csv(path, index=False)
    logger.info("Saved dataset to %s", path)


def load_dataset(path: str) -> pd.DataFrame:
    """Load dataset from CSV."""
    df = pd.read_csv(path)
    logger.info("Loaded dataset from %s with shape %s", path, df.shape)
    return df
