# Phenotype-Aware Machine Learning for Predicting Rapid Fibrosis Progression in MASLD

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Code for:

> **pRFPS: A Subgroup-Specific Interpretable Risk Score for Predicting Rapid
> Fibrosis Progression in Metabolic Dysfunction-Associated Steatotic Liver Disease**
> *Pacific Symposium on Biocomputing (PSB) 2027*

---

## Overview

Standard clinical tools like FIB-4 miss approximately 24% of rapid MASLD
progressors. This pipeline addresses that gap through three steps:

1. **Phenotyping** — Latent Class Analysis (LCA) identifies two biologically
   distinct subgroups: liver-specific (LS) and cardiometabolic (CM).
2. **Subgroup-specific modelling** — Ten ML architectures trained independently
   within each subgroup with nested feature selection.
3. **Clinical translation** — SHAP-based attribution derives the pRFPS,
   an integer-weighted bedside score from five routine laboratory values.

**Cohorts:**
- Discovery: Mayo Clinic Biobank (MCB), n=5,747
- Validation: Tapestry Study, n=7,480

---

## Repository Structure

```
masld-phenotype-ml/
├── Rebuttal_experiments.py   # Complete pipeline (single-file, run this)
├── run.py                    # Entry point wrapper
├── config.py                 # All constants, hyperparameter grids
├── train.py                  # Training utilities (modular reference)
├── evaluate.py               # Evaluation: DeLong, calibration, DCA, stratified
├── ablation.py               # Ablation: feature selection × partitioning
├── prfps.py                  # pRFPS derivation + continuous variant comparison
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/[your-username]/masld-phenotype-ml.git
cd masld-phenotype-ml
pip install -r requirements.txt
```

Python 3.9 or higher required.

---

## Usage

### Full pipeline

```bash
python run.py \
    --data path/to/mcb_data.tsv \
    --val  path/to/tapestry_data.tsv \
    --n_boot 1000
```

### Quick test (fast, 50 bootstrap resamples)

```bash
python run.py \
    --data path/to/mcb_data.tsv \
    --val  path/to/tapestry_data.tsv \
    --n_boot 50
```

---

## Data

The MCB and Tapestry datasets contain protected health information and are
not publicly available. Access to the Mayo Clinic Biobank can be requested
through the [Mayo Clinic Biobank](https://www.mayo.edu/research/centers-programs/mayo-clinic-biobank/overview).

Expected input columns:

| Column | Description |
|--------|-------------|
| `overall_progression_category2` | Target label |
| `Subgroup_5` | LCA subgroup assignment |
| `diagnosis_age` | Age at MASLD diagnosis (years) |
| `Avg_BMI` | Mean BMI (kg/m²) |
| `ALP`, `AST.x`, `ALT.x` | Liver enzymes (U/L) |
| `PLATELET_COUNT` | Platelet count (×10⁹/L) |
| `ALBUMIN` | Albumin (g/dL) |
| `BUN` | Blood urea nitrogen (mg/dL) |
| `HDL`, `LDL`, `TRIGLYCERIDE` | Lipid panel |
| `first_FIB4` | Baseline FIB-4 score |
| `Sex` | Sex (0=Female, 1=Male) |
| `PNPLA3`, `TM6SF2`, `HSD17B13` | Genetic variant dosages |

---

## Key Outputs

Results saved to `MASLD_Comprehensive_Analysis_*/`:

| Directory | Contents |
|-----------|----------|
| `rebuttal_outputs/` | Master results table, DeLong comparisons |
| `clinical_risk_score/` | pRFPS formula, feature tables |
| `best_model_shap/` | SHAP importance plots |
| `patient_level/` | Per-patient predictions with pRFPS and FIB-4 |
| `patient_level/*/stratified_eval/` | Sex- and age-stratified metrics |
| `prfps_continuous_variants/` | Binary vs continuous I() comparison |
| `calibration_plots/` | Reliability diagrams |
| `dca_plots/` | Decision curve analysis |
| `lca_analysis/` | LCA BIC/entropy plots, posterior distributions |

---

## pRFPS Formulas

**Liver-specific (LS / C1)** — cutoff = 15:
```
pRFPS_LS = 10·I(Age > 45) + 5·I(BUN > 21) + 5·I(ALP > 109)
         + 3·I(Platelet < 125) + 3·I(HDL < 61)
```

**Cardiometabolic (CM / C2)** — cutoff = 18:
```
pRFPS_CM = 10·I(ALP > 97) + 6·I(Platelet < 313) + 1·I(LDL > 64)
         + 1·I(Albumin < 13) + 1·I(AST > 39)
```

I(·) = 1 if condition is true, 0 otherwise. Thresholds derived by
bootstrap-stabilised Youden's J on MCB training data (50 resamples).

---

## Methods Summary

| Step | Method | Key design choice |
|------|--------|-------------------|
| Feature construction | Mean of pre-index measurements | Temporal separation from outcomes |
| Subgroup discovery | LCA, BIC-optimal k=2 | Entropy=0.749, mean PP=0.89 |
| Feature selection | Consensus of 5 methods | Nested inside training fold only |
| Model training | GridSearchCV, 5-fold CV | 10 architectures per subgroup |
| Threshold selection | Youden's J on OOF predictions | No test-set leakage |
| SHAP attribution | Mean absolute \|SHAP\| (training set) | Sign-cancellation-free |
| pRFPS weights | Normalised mean \|SHAP\| → integer | Bedside computability |
| Validation | Tapestry, locked models, no refitting | Independent IRB |

---

## Reproducibility

All random states fixed at `seed=42`. Results directory names encode the
configuration. See `requirements.txt` for exact package versions.

---

## Citation

```bibtex
@inproceedings{priya2027prfps,
  title     = {pRFPS: A Subgroup-Specific Interpretable Risk Score for
               Predicting Rapid Fibrosis Progression in MASLD},
  author    = {Priya, T.S. and others},
  booktitle = {Pacific Symposium on Biocomputing},
  year      = {2027}
}
```

---

## Conflict of Interest

A.J.A. has received grants or contracts from Rhythm Pharmaceuticals,
Boehringer Ingelheim, Vivus Inc., Vivus Pharmaceuticals, Regeneron, and
Novo Nordisk (paid to institute); holds royalties or licenses from Phenomix
Sciences; has received personal consultancy fees from Boehringer Ingelheim,
Regeneron, Currax, and Structure Pharmaceuticals, and consultancy fees
(paid to institute) from Amgen, RareStone, and Bausch Health; has received
personal honoraria from Eli Lilly and Boehringer Ingelheim, and honoraria
(paid to institute) from Vivus Pharmaceuticals; has received travel support
from Currax (paid to institute); owns 10 patents, of which 3 have been issued;
has served on Data Safety Monitoring Boards or Advisory Boards for Amgen,
Boehringer Ingelheim, Currax, Structure Pharmaceuticals, and Regeneron; and
holds stock options in Gila Therapeutics and Phenomix Sciences.
All other authors declare no conflicts of interest.

## LLM Disclosure

Large language model assistance was used in the preparation of this manuscript
and codebase, including support with writing, editing, and code review.
All scientific content, analyses, results, and conclusions were generated,
verified, and approved by the authors.

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Contact

[Author Name] · [email@mayo.edu] · Mayo Clinic
