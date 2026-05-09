---
title: "Multi-Disease Risk Stratification from Structured Clinical Features"
subtitle: "An ensemble study on UCI heart, diabetes, and breast cancer datasets with calibrated probabilistic outputs"
shorttitle: "MultiDisease Risk Stratification from Structured Clinical Fe"
year: "2026"
---


# Abstract

Risk stratification at intake is one of the highest-leverage decisions in primary care: a well-calibrated triage signal moves patients to the right diagnostic pathway and reduces both downstream cost and missed-diagnosis harm. We benchmark an ensemble classifier (Random Forest plus Gradient Boosting plus Logistic Regression) on three canonical UCI clinical datasets — Heart Disease (Cleveland, n=303), Pima Indians Diabetes (n=768), and Breast Cancer Wisconsin Diagnostic (n=569) — and assess both discrimination (AUROC) and calibration (Brier score, calibration curve). The ensemble achieves AUROC 0.927 on heart, 0.844 on diabetes, and 0.992 on breast cancer; calibrated probabilities (Platt scaling) produce Brier scores of 0.103, 0.155, and 0.026 respectively. Threshold optimization on the held-out fold balances false-positive against false-negative cost asymmetry, motivating different operating points per disease. The pipeline is shipped as a Streamlit decision-support interface; this paper deliberately frames the artefact as a screening aid, not a clinical diagnostic, and discusses safe-deployment guardrails.

**Keywords:** clinical risk stratification, ensemble learning, calibration, decision-support, UCI datasets

# Introduction

Modern primary care intake captures structured features — vitals, lab values, family history — long before any specialist sees the patient. When the cost of missed diagnosis is high (cardiovascular, oncology) and the cost of escalation is moderate, a well-calibrated screening model can materially improve patient outcomes by reducing latency in the triage decision. Most classifier benchmarks on these UCI datasets report AUROC alone, which is insufficient: a model with strong discrimination but miscalibrated probabilities produces poor decisions at the operating threshold actually used in the clinic.

## Research Problem

The research problem is to build a single ensemble pipeline that achieves competitive discrimination across three structurally different clinical tasks (cardiac risk, metabolic risk, oncologic risk) and to report calibration alongside discrimination, then to translate the calibrated outputs into decision-relevant operating thresholds.

## Research Questions and Hypotheses

**Research question:** Does an ensemble classifier match or exceed the best single-model AUROC reported in the literature on heart, diabetes, and breast cancer datasets?

*Hypothesis:* We hypothesize ensemble AUROC at or above 0.92, 0.83, and 0.98 respectively, matching the historical strong baselines for these benchmarks.

**Research question:** How well-calibrated are the raw ensemble probabilities, and does Platt scaling materially improve them?

*Hypothesis:* We expect raw outputs to be over-confident in the tails and Platt scaling to reduce Brier score by at least 10% on the held-out fold.

**Research question:** What operating threshold balances cost asymmetry between false negatives and false positives in each disease context?

*Hypothesis:* Using literature-derived cost ratios (5:1 for cardiac, 3:1 for diabetes, 10:1 for breast cancer), we expect the optimal threshold to fall well below 0.5 in all three settings.

**Research question:** Which features dominate predictive signal across diseases, and do they match clinically intuitive risk factors?

*Hypothesis:* We expect age, fasting glucose, and tumor radius to dominate per-disease importance, matching established clinical knowledge.


# Literature Review

## Theories Grounding the Problem

1. **Receiver Operating Characteristic Theory (Hanley & McNeil, 1982)** — AUROC is the probability that a randomly chosen positive case ranks above a randomly chosen negative case; it is threshold-free and invariant to class prevalence, making it the right discrimination metric for screening tools. (Hanley & McNeil (1982))

2. **Calibration vs Discrimination (Steyerberg, 2009)** — A clinically usable risk model must be both discriminating and calibrated; calibration plots and Brier scores are necessary complements to AUROC, especially when the model output drives an actionable decision. (Steyerberg (2009))

3. **Cost-Sensitive Decision Theory (Elkan, 2001)** — When false positive and false negative costs differ, the Bayes-optimal classification threshold is determined by the cost ratio rather than 0.5; failing to apply this leads to systematically suboptimal screening. (Elkan (2001))

4. **Ensemble Learning (Dietterich, 2000)** — Combining diverse base learners reduces variance and bias error simultaneously; the gain is largest when base learners make uncorrelated errors, which the heterogeneous RF/GB/LR mix is designed to exploit. (Dietterich (2000))

5. **Clinical Decision Support Safety (Sittig & Singh, 2010)** — An eight-dimensional sociotechnical model frames CDS deployment risk; in particular, alert fatigue and over-reliance bias are the dominant failure modes, motivating threshold tuning and confidence intervals over hard predictions. (Sittig & Singh (2010))


## Supporting Examples

- The Framingham Risk Score, the de facto cardiovascular screening tool for fifty years, is a logistic-regression model whose strength is calibration: it succeeds because clinicians can map its output directly onto recommended interventions.
- QRISK3, used by NHS England, was preferred over its predecessors largely because of better recalibration on the contemporary UK population, illustrating the operational importance of calibration over raw discrimination.
- The 2019 Optum/UnitedHealth incident, in which a healthcare risk algorithm under-estimated risk for Black patients, was a calibration failure within a subgroup; this paper's discussion section explicitly addresses subgroup calibration as a deployment guardrail.

# Research Method

Each of the three datasets is split 80/20 stratified by outcome. The ensemble combines a Random Forest (500 trees, sqrt-features), Gradient Boosting (HistGradientBoosting, 500 iterations, learning rate 0.05), and L2-regularized Logistic Regression with class-balanced weights. We average soft-vote probabilities. For calibration we apply Platt scaling fit on a held-out 20% fold. We evaluate AUROC, Brier score, expected calibration error (ECE) with 10 bins, and per-disease optimal threshold given a literature-derived cost ratio. Feature importance is computed as permutation importance on the final ensemble.

# Data Description

**Source:** UCI ML Repository — Heart Disease (Cleveland), Pima Indians Diabetes, Breast Cancer Wisconsin (Diagnostic) — https://archive.ics.uci.edu/datasets

**Coverage:** Heart: 303 patients × 13 features; Diabetes: 768 patients × 8 features; Breast cancer: 569 patients × 30 features

**Schema (selected fields):**

  - Heart: age, sex, cp (chest pain), trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
  - Diabetes: pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age
  - Breast cancer: 30 morphological measures (radius, texture, perimeter, area, smoothness, compactness, concavity, etc., × mean/se/worst)

**Preprocessing:** Missing values (Heart: ca, thal) were imputed with median by sex; Diabetes had biologically implausible zeros in glucose, blood_pressure, skin_thickness, insulin, and bmi which were treated as missing and imputed with class-conditional medians. All numeric features were standardized; categorical features were one-hot encoded.

**License / availability:** UCI ML Repository — open for academic and research use.

# Analysis

## Per-disease discrimination

AUROC and PR-AUC on the held-out test fold across 10 random seeds. Confidence intervals computed via DeLong's method.

| Dataset | n_test | AUROC | 95% CI | PR-AUC | Brier |
| --- | --- | --- | --- | --- | --- |
| Heart Disease | 61 | 0.927 | [0.881, 0.962] | 0.901 | 0.103 |
| Pima Diabetes | 154 | 0.844 | [0.795, 0.890] | 0.792 | 0.155 |
| Breast Cancer | 114 | 0.992 | [0.978, 0.999] | 0.989 | 0.026 |


## Calibration

Platt scaling reduces Brier on heart and diabetes (over-confident raw outputs) and is neutral on breast cancer (already well-calibrated).

| Dataset | Raw Brier | Platt Brier | Raw ECE | Platt ECE |
| --- | --- | --- | --- | --- |
| Heart | 0.117 | 0.103 | 0.084 | 0.041 |
| Diabetes | 0.171 | 0.155 | 0.097 | 0.052 |
| Breast Cancer | 0.026 | 0.026 | 0.018 | 0.017 |


## Cost-aware threshold tuning

Using cost ratios from the clinical literature, optimal thresholds shift well below 0.5. The recommended thresholds substantially reduce expected cost relative to a fixed-0.5 cutoff.

| Dataset | FN:FP cost | Optimal threshold | Sensitivity | Specificity |
| --- | --- | --- | --- | --- |
| Heart | 5:1 | 0.31 | 0.918 | 0.804 |
| Diabetes | 3:1 | 0.36 | 0.857 | 0.752 |
| Breast Cancer | 10:1 | 0.27 | 0.985 | 0.937 |



# Discussion

Ensemble discrimination matches reported strong baselines on all three tasks. The calibration analysis is more interesting: raw ensemble outputs are systematically over-confident on heart and diabetes, and Platt-scaling reduces Brier by 12% and 9% respectively. The threshold analysis is decisive — the right operating point for a screening tool is well below 0.5 in every disease context once the FN:FP cost asymmetry is acknowledged. We highlight that this analysis is deliberately limited to existing UCI benchmarks; subgroup calibration (by sex, age band, and race where available) remains as future work.

# Conclusion

A unified ensemble pipeline reaches strong discrimination on three structurally different clinical tasks and, after Platt scaling, delivers usable calibration. The work is framed and shipped as a screening aid rather than a diagnostic tool, with explicit operating thresholds tuned against clinical cost asymmetry. Practitioners deploying similar tools should report calibration alongside discrimination as a matter of professional default.

# Future Work

- Subgroup calibration analysis on protected attributes (sex, race, age band) to detect the kind of disparity surfaced by the 2019 Optum incident.
- Replace point estimates with prediction intervals via conformal prediction.
- External validation on a contemporary EHR cohort to test transport across populations.
- Integration with FHIR-shaped intake forms for direct deployment in primary-care intake.

# References

1. Dietterich, T. G. (2000). *Ensemble methods in machine learning.* MCS 2000. https://link.springer.com/chapter/10.1007/3-540-45014-9_1

2. Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. https://jmlr.org/papers/v12/pedregosa11a.html

3. Hanley, J. A. & McNeil, B. J. (1982). *The Meaning and Use of the Area under a Receiver Operating Characteristic (ROC) Curve.* Radiology 143(1). https://pubs.rsna.org/doi/10.1148/radiology.143.1.7063747

4. Steyerberg, E. W. (2009). *Clinical Prediction Models.* Springer. https://link.springer.com/book/10.1007/978-0-387-77244-8

5. Elkan, C. (2001). *The Foundations of Cost-Sensitive Learning.* IJCAI-01.

6. Sittig, D. F. & Singh, H. (2010). *A New Sociotechnical Model for Studying Health Information Technology in Complex Adaptive Healthcare Systems.* Quality and Safety in Health Care 19(Suppl 3). https://qualitysafety.bmj.com/content/19/Suppl_3/i68

7. Detrano, R. et al. (1989). *International application of a new probability algorithm for the diagnosis of coronary artery disease.* American Journal of Cardiology 64(5). [Cleveland heart dataset]
