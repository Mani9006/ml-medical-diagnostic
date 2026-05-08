"""Streamlit interactive dashboard for medical diagnostic predictions."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_synthesis import PatientDataSynthesizer
from src.ensemble_model import MedicalEnsembleClassifier
from src.evaluator import ModelEvaluator
from src.preprocessor import MedicalDataPreprocessor
from src.risk_engine import RiskStratificationEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

st.set_page_config(
    page_title="Medical Diagnostic Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_or_train_model():
    """Load pre-trained model or train a fresh one if not available."""
    model_path = PROJECT_ROOT / "models" / "ensemble.pkl"
    if model_path.exists():
        ensemble = MedicalEnsembleClassifier()
        ensemble.load(model_path)
        return ensemble, None, None

    with st.spinner("Training model on synthetic data... This may take a moment."):
        synthesizer = PatientDataSynthesizer(n_samples=5000, random_state=42)
        data = synthesizer.generate()

        preprocessor = MedicalDataPreprocessor(scale_method="robust")
        X, y, feature_names = preprocessor.fit_transform(data)

        ensemble = MedicalEnsembleClassifier(random_state=42)
        ensemble.fit(X, y)

        # Save
        model_path.parent.mkdir(parents=True, exist_ok=True)
        ensemble.save(model_path)
        st.success("Model trained and saved successfully!")

        return ensemble, preprocessor, feature_names


def render_sidebar():
    """Render sidebar with patient input controls."""
    st.sidebar.header("Patient Information")

    age = st.sidebar.slider("Age", 18, 95, 45)
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    gender_val = 1 if gender == "Male" else 0

    bmi = st.sidebar.slider("BMI", 15.0, 55.0, 26.0, step=0.5)
    exercise_hours = st.sidebar.slider("Exercise (hours/week)", 0.0, 14.0, 3.0, step=0.5)
    smoking = 1 if st.sidebar.checkbox("Smoking") else 0
    family_history = 1 if st.sidebar.checkbox("Family History") else 0
    stress_level = st.sidebar.slider("Stress Level (1-10)", 1, 10, 4)

    st.sidebar.header("Blood Markers")
    glucose = st.sidebar.slider("Glucose (mg/dL)", 60, 300, 95)
    hba1c = st.sidebar.slider("HbA1c (%)", 4.0, 14.0, 5.2, step=0.1)
    insulin = st.sidebar.slider("Insulin", 5.0, 100.0, 15.0, step=1.0)
    blood_pressure = st.sidebar.slider("Blood Pressure (mmHg)", 80, 220, 120)
    cholesterol = st.sidebar.slider("Cholesterol (mg/dL)", 120, 450, 190)
    ldl = st.sidebar.slider("LDL (mg/dL)", 60, 250, 110)

    st.sidebar.header("Liver Markers")
    bilirubin = st.sidebar.slider("Bilirubin (mg/dL)", 0.1, 8.0, 0.8, step=0.1)
    albumin = st.sidebar.slider("Albumin (g/dL)", 2.0, 5.5, 4.2, step=0.1)
    ast = st.sidebar.slider("AST (U/L)", 10, 200, 25)
    alt = st.sidebar.slider("ALT (U/L)", 10, 200, 22)
    alcohol_units = st.sidebar.slider("Alcohol (units/week)", 0, 40, 4)
    hepatitis_history = 1 if st.sidebar.checkbox("Hepatitis History") else 0

    patient_data = {
        "age": age,
        "gender": gender_val,
        "bmi": bmi,
        "glucose": glucose,
        "blood_pressure": blood_pressure,
        "insulin": insulin,
        "hba1c": hba1c,
        "cholesterol": cholesterol,
        "ldl": ldl,
        "stress_level": stress_level,
        "bilirubin": bilirubin,
        "albumin": albumin,
        "ast": ast,
        "alt": alt,
        "alcohol_units": alcohol_units,
        "exercise_hours": exercise_hours,
        "smoking": smoking,
        "family_history": family_history,
        "hepatitis_history": hepatitis_history,
    }

    return patient_data


def main():
    """Main dashboard entry point."""
    st.title("🏥 Medical Diagnostic Assistant")
    st.markdown(
        """
        This tool uses an ensemble machine learning model to estimate the risk
        of **diabetes**, **heart disease**, and **liver conditions** based on
        patient biomarkers and lifestyle factors.

        ⚠️ **Disclaimer**: This is a demonstration tool using synthetic data.
        It does not provide medical advice. Always consult a qualified healthcare professional.
        """
    )

    # Load model
    ensemble, preprocessor, feature_names = load_or_train_model()

    # Always create a fresh preprocessor for the dashboard
    synthesizer = PatientDataSynthesizer(n_samples=5000, random_state=42)
    train_data = synthesizer.generate()
    preprocessor = MedicalDataPreprocessor(scale_method="robust")
    _, _, feature_names = preprocessor.fit_transform(train_data)

    patient_data = render_sidebar()

    # Build input DataFrame
    input_df = pd.DataFrame([patient_data])

    # Preprocess
    try:
        X_input = preprocessor.transform(input_df)
    except (ValueError, KeyError) as e:
        st.error(f"Preprocessing error: {e}")
        return

    # Predict
    probs = ensemble.predict_proba(X_input)[0]
    preds = ensemble.predict(X_input)[0]

    # Risk stratification
    risk_engine = RiskStratificationEngine()
    profile = risk_engine.assess(
        probabilities=probs,
        features=patient_data,
    )

    # ---- Main Content ----
    col1, col2, col3 = st.columns(3)

    conditions = ["diabetes", "heart_disease", "liver_condition"]
    condition_display = ["Diabetes", "Heart Disease", "Liver Condition"]

    for col, cond, display, prob, pred in zip(
        [col1, col2, col3], conditions, condition_display, probs, preds
    ):
        with col:
            st.subheader(display)
            # Gauge-style display
            st.progress(min(float(prob), 1.0))
            st.metric("Risk Probability", f"{prob:.1%}")
            status = "⚠️ At Risk" if pred == 1 else "✅ Low Risk"
            st.write(f"Status: **{status}**")

    st.divider()

    # Risk Profile
    st.header("Risk Stratification Profile")
    profile_data = profile.to_dict()

    # Overall risk banner
    risk_color = profile_data["overall_risk_color"]
    risk_label = profile_data["overall_risk"].upper()
    st.markdown(
        f"""
        <div style="padding: 15px; border-radius: 8px; background-color: {risk_color}20;
                    border-left: 5px solid {risk_color};">
            <h4 style="margin:0; color: {risk_color};">Overall Risk: {risk_label}</h4>
            <p style="margin:5px 0 0 0;">{profile.overall_risk.recommendation}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Detailed condition breakdown
    st.subheader("Condition Breakdown")
    for cond_info in profile_data["conditions"]:
        c_color = cond_info["risk_color"]
        with st.expander(
            f"{cond_info['condition'].replace('_', ' ').title()} — "
            f"{cond_info['probability']:.1%} ({cond_info['risk_level'].upper()})"
        ):
            st.markdown(
                f"<div style='color:{c_color}; font-weight:bold;'>Risk Level: {cond_info['risk_level'].upper()}</div>",
                unsafe_allow_html=True,
            )
            st.write(f"**Probability:** {cond_info['probability']:.2%}")
            st.write(f"**Recommendation:** {cond_info['recommendation']}")
            if cond_info["contributing_factors"]:
                st.write("**Contributing Factors:**")
                for factor in cond_info["contributing_factors"]:
                    st.write(f"  - {factor}")

    # ---- Batch Analysis Tab ----
    st.divider()
    st.header("Batch Analysis")
    st.markdown(
        "Upload a CSV file with patient data for batch risk assessment. "
        "The file should contain columns matching the feature names."
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write(f"Loaded {len(batch_df)} records")

            if st.button("Run Batch Assessment"):
                with st.spinner("Processing..."):
                    X_batch = preprocessor.transform(batch_df)
                    batch_probs = ensemble.predict_proba(X_batch)

                    # Build results
                    results = []
                    for i, probs in enumerate(batch_probs):
                        profile = risk_engine.assess(
                            probs, patient_id=batch_df.iloc[i].get("patient_id", i)
                        )
                        pd_data = profile.to_dict()
                        row = {
                            "patient_id": pd_data["patient_id"],
                            "overall_risk": pd_data["overall_risk"],
                            "overall_score": pd_data["overall_score"],
                        }
                        for ci in pd_data["conditions"]:
                            row[f"{ci['condition']}_prob"] = ci["probability"]
                            row[f"{ci['condition']}_risk"] = ci["risk_level"]
                        results.append(row)

                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    # Summary counts
                    st.subheader("Risk Distribution")
                    risk_counts = results_df["overall_risk"].value_counts()
                    st.bar_chart(risk_counts)

                    # Download
                    csv = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download Results",
                        csv,
                        "risk_assessment_results.csv",
                        "text/csv",
                    )
        except Exception as e:
            st.error(f"Error processing batch: {e}")

    # ---- Model Info ----
    st.divider()
    st.header("Model Information")
    st.markdown(
        """
        **Architecture:** Heterogeneous Ensemble
        - Random Forest (200 estimators, max_depth=12)
        - Gradient Boosting (150 estimators, max_depth=5)
        - Weighted probability averaging (50/50)

        **Features:** 18 raw biomarkers + 7 engineered composite features

        **Risk Levels:**
        - 🟢 **Low** (<20%): Continue routine monitoring
        - 🟡 **Medium** (20-45%): Follow-up within 3 months
        - 🟠 **High** (45-70%): Follow-up within 1 month, additional tests
        - 🔴 **Critical** (>70%): Urgent specialist referral
        """
    )


if __name__ == "__main__":
    main()
