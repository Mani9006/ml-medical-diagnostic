# Architecture Overview

## Medical Diagnostic Assistant

### High-Level Design

The Medical Diagnostic Assistant is a modular Python ML system that predicts three medical conditions (diabetes, heart disease, liver condition) from patient biomarkers using an ensemble of tree-based classifiers. The architecture follows a clean pipeline pattern with distinct, testable components.

### System Components

```
+---------------------------------------------------+
|                  DASHBOARD (Streamlit)             |
|  Interactive UI for single-patient and batch input |
+---------------------------------------------------+
                          |
                          v
+---------------------------------------------------+
|              RISK STRATIFICATION ENGINE            |
|  Maps probabilities -> risk levels + clinical recs |
+---------------------------------------------------+
                          |
                          v
+---------------------------------------------------+
|           ENSEMBLE CLASSIFIER                      |
|  Random Forest + Gradient Boosting (MultiOutput)   |
|  Weighted probability averaging                    |
+---------------------------------------------------+
                          ^
                          |
+---------------------------------------------------+
|           PREPROCESSING PIPELINE                   |
|  Feature Engineering -> Outlier Clip -> Impute    |
|  -> Scale (RobustScaler)                           |
+---------------------------------------------------+
                          ^
                          |
+---------------------------------------------------+
|           DATA SYNTHESIS                           |
|  Realistic patient records with correlations       |
|  Logistic threshold condition assignment           |
+---------------------------------------------------+
```

### Data Flow

1. **Data Synthesis** (`data_synthesis.py`): Generates 18+ raw biomarker features across 5000+ synthetic patients. Features are correlated via realistic epidemiological models (e.g., BMI influences glucose and insulin). Conditions are assigned via weighted logistic thresholds with calibrated prevalence.

2. **Preprocessing** (`preprocessor.py`):
   - **MedicalFeatureEngineer**: Adds 7 composite features (BMI category, glucose-HbA1c ratio, liver enzyme ratio, cholesterol-LDL ratio, metabolic/cardio/liver risk scores)
   - **OutlierClipper**: IQR-based extreme value clipping (factor=3.0)
   - **SimpleImputer**: Median strategy imputation
   - **RobustScaler**: Median/IQR scaling resistant to outliers

3. **Ensemble Model** (`ensemble_model.py`):
   - **Random Forest**: 200 estimators, max_depth=12, balanced_subsample class weights
   - **Gradient Boosting**: 150 estimators, max_depth=5, learning_rate=0.1
   - **MultiOutputClassifier**: Wraps each for multi-label prediction
   - **Weighted Averaging**: 50/50 blend of RF and GB probabilities
   - **Cross-Validation**: Stratified K-Fold (5 folds)

4. **Evaluation** (`evaluator.py`):
   - Per-condition metrics (accuracy, precision, recall, F1, ROC-AUC, average precision)
   - Confusion matrices, ROC curves, precision-recall curves
   - Risk distribution histograms
   - Classification reports

5. **Risk Stratification** (`risk_engine.py`):
   - 4-tier system: LOW / MEDIUM / HIGH / CRITICAL
   - Condition-specific probability thresholds
   - Multi-condition escalation logic
   - Contributing factor identification from raw features
   - Clinical recommendations per risk level

6. **Dashboard** (`dashboard.py`):
   - Streamlit-based interactive UI
   - 18 biomarker input sliders/switches
   - Real-time probability gauges
   - Detailed risk breakdown with contributing factors
   - Batch CSV upload and assessment

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Multi-label not multi-class | Patients can have multiple conditions simultaneously |
| Ensemble (RF + GB) | Combines RF's robustness with GB's bias reduction; diversity improves calibration |
| RobustScaler over StandardScaler | Medical data has natural outliers (severe cases); RobustScaler is more stable |
| Synthetic data | Enables full pipeline testing without PHI/privacy concerns |
| Condition-specific thresholds | Diabetes, heart disease, and liver conditions have different clinical risk profiles |
| Feature engineering in pipeline | Composite medical scores (metabolic, cardio, liver) capture domain knowledge |
| IQR outlier clipping (factor=3) | Preserves severe-but-valid cases while removing numerical artifacts |

### File Structure

```
src/
  __init__.py          - Package metadata
  data_synthesis.py    - PatientDataSynthesizer with ConditionConfig profiles
  preprocessor.py      - MedicalDataPreprocessor with custom sklearn transformers
  ensemble_model.py    - MedicalEnsembleClassifier with RF+GB+MultiOutput
  evaluator.py         - ModelEvaluator with plots and metrics
  risk_engine.py       - RiskStratificationEngine with 4-tier levels
  dashboard.py         - Streamlit interactive application
```

### Extension Points

1. **New conditions**: Add a `ConditionConfig` to `ALL_CONDITIONS` and new feature generators
2. **New models**: Extend `MedicalEnsembleClassifier` with additional base estimators
3. **Real data**: Replace `PatientDataSynthesizer` with a data loader for real EHR data
4. **API deployment**: Wrap `MedicalEnsembleClassifier` in a FastAPI/Flask service
5. **Monitoring**: Add `model_performance_tracker` to log prediction drift
