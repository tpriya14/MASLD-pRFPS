# MASLD pRFPS Prediction Pipeline
![Python](https://img.shields.io/badge/Python-3.9%2B-blue) ![License](https://img.shields.io/badge/License-MIT-green)

This repository contains the Python code for predicting rapid fibrosis progression in MASLD (Metabolic Dysfunction-Associated Steatotic Liver Disease) using a subgroup-specific interpretable risk score — **pRFPS**. The pipeline trains ML models on LCA-derived patient subgroups and derives clinically actionable risk scores via SHAP values.

> **Note:** Data from the Mayo Clinic Biobank (MCB) and Tapestry cohorts are not publicly available due to privacy restrictions.

---

## Overview

<!-- Add pipeline overview figure here -->

The analysis is structured into four components:

1. **Data preparation** — loading, preprocessing, and FIB-4 computation
2. **Subgroup-specific model training** — GridSearchCV + stacking ensemble across C1/C2 subgroups
3. **pRFPS score construction** — SHAP-derived binary risk score with clinical thresholds
4. **Validation** — external validation on the Tapestry cohort with covariate shift analysis

---

## pRFPS Score Formulas

<!-- Add score derivation figure here -->

**pRFPS-LS** · Liver-Specific subgroup (C1) · Cutoff = 15
```
10·I(Age > 45) + 5·I(BUN > 21) + 5·I(ALP > 109) + 3·I(Platelet < 125) + 3·I(HDL < 61)
```

**pRFPS-CM** · Cardiometabolic subgroup (C2) · Cutoff = 18
```
10·I(ALP > 97) + 6·I(Platelet < 313) + 1·I(LDL > 64) + 1·I(Albumin < 13) + 1·I(AST > 39)
```

---

## Study Design

| Cohort | Role | Subgroups |
|--------|------|-----------|
| Mayo Clinic Biobank (MCB) | Discovery + internal validation | C1 (Liver-Specific), C2 (Cardiometabolic) |
| Tapestry Study | External validation | C1, C2 |

Subgroups are identified by latent class analysis (LCA) run separately — see the companion [R repository](#) for that step.

---

## How to Run

### Prerequisites

- Python ≥ 3.9
- Packages: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `shap`, `xgboost`, `lightgbm`

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/masld-phenotype-ml.git
cd masld-phenotype-ml
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Prepare Data

Place your data files in the `data/` directory:
- `data/mcb_data.tsv` — MCB cohort (tab-separated)
- `data/tapestry_data.tsv` — Tapestry cohort (tab-separated)

See `data/README.md` for column specification and encoding details.

### Step 4: Run the Pipeline

```bash
# Modular pipeline (recommended)
python scripts/run.py --data data/mcb_data.tsv --val data/tapestry_data.tsv

# Original single-file version (paper-exact)
python Rebuttal_experiments.py --data data/mcb_data.tsv --val data/tapestry_data.tsv
```

Results are saved to a timestamped folder under `results/`.

> **Windows note:** The pipeline runs with `n_jobs=1` on Windows to avoid multiprocessing issues. Results may differ slightly from the paper (Linux, `n_jobs=-1`) due to sklearn's parallel random state behavior, but are scientifically equivalent.

---

## Repository Structure

```
masld-phenotype-ml/
│
├── Rebuttal_experiments.py        # Complete single-file pipeline (paper-exact)
│
├── scripts/
│   └── run.py                     # Entry point for the modular pipeline
│
├── src/                           # Modular package
│   ├── main.py                    # Orchestrates all modules
│   ├── config.py                  # Constants: column names, labels, model grid
│   ├── utils.py                   # Data loading, FIB-4, metrics, plotting
│   ├── train.py                   # Model training, stacking ensemble, OOF thresholds
│   ├── prfps.py                   # pRFPS score construction via SHAP (core contribution)
│   ├── evaluate.py                # Covariate shift, patient-level outputs, metrics plots
│   ├── ablation.py                # Feature selection, nested CV, partition ablation
│   └── validate.py                # External validation on Tapestry cohort
│
├── data/
│   └── README.md                  # Column specification and encoding
│
├── results/                       # Output directory (auto-created on run)
│
├── requirements.txt
└── LICENSE
```

---

## Code Map

| Task | File | Key Function |
|------|------|-------------|
| pRFPS score derivation | `src/prfps.py` | `build_shap_clinical_risk_score()` |
| Model training & hyperparameter search | `src/train.py` | `train_and_evaluate_setting()` |
| Stacking ensemble | `src/train.py` | `get_oof_probabilities()` |
| Covariate shift analysis | `src/evaluate.py` | `covariate_shift_analysis()` |
| Feature selection & ablation | `src/ablation.py` | `ablation_feature_selection()` |
| External validation | `src/validate.py` | `validate_external()` |
| Model configuration | `src/config.py` | `MODELS_CONFIG` |
| Full pipeline flow | `src/main.py` | `main()` |

---

## Output

<!-- Add example results figure here -->

Each run produces a results folder containing:

- `results_Overall.csv`, `results_C1.csv`, `results_C2.csv` — per-model metrics
- `patient_level_*.csv` — patient-level predictions
- `delong_prfps_vs_fib4.csv` — pRFPS vs FIB-4 statistical comparison
- `continuous_vs_integer_prfps_delta.csv` — continuous vs integer pRFPS delta
- Figures: ROC curves, calibration plots, SHAP bar plots

---

## References

- Lundberg, S.M. & Lee, S.I. A unified approach to interpreting model predictions. *NeurIPS* (2017).
- Ester, M. et al. A density-based algorithm for discovering clusters in large spatial databases with noise. *KDD* (1996).
- Zhou, W. et al. Latent class analysis-derived classification improves cancer-specific death stratification. *NPJ Precis Oncol* 7, 60 (2023).

---

## Citation

<!-- Add citation here once published -->

---

## License

MIT — see [LICENSE](LICENSE).
