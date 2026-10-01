"""
config.py — Global constants, model configurations, and paths for the MASLD pipeline.
"""

import os

# ── Output directories ────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR    = os.path.join(BASE_DIR, "data")

# ── Label constants ───────────────────────────────────────────────────────────
POSITIVE_LABEL = "Rapid Progression"
NEGATIVE_LABEL = "No Progression"
TARGET_COL     = "rapid_progression"   # binary 0/1 outcome column in data

# ── LCA subgroup labels ───────────────────────────────────────────────────────
SUBGROUP_LS = "C1"   # Liver-Specific subgroup
SUBGROUP_CM = "C2"   # Cardiometabolic subgroup

# ── pRFPS formulas (derived from SHAP analysis) ───────────────────────────────
# Liver-Specific (LS / C1) subgroup
PRFPS_LS_FORMULA = {
    "features": ["Age", "BUN", "ALP", "Platelet", "HDL"],
    "weights":  [10,    5,     5,     3,           3   ],
    "directions": [">45", ">21", ">109", "<125", "<61"],
    "cutoff": 15,
    "label": "pRFPS_LS",
}

# Cardiometabolic (CM / C2) subgroup
PRFPS_CM_FORMULA = {
    "features": ["ALP", "Platelet", "LDL", "Albumin", "AST"],
    "weights":  [10,    6,          1,     1,          1  ],
    "directions": [">97", "<313", ">64", "<13", ">39"],
    "cutoff": 18,
    "label": "pRFPS_CM",
}

# ── Cross-validation ──────────────────────────────────────────────────────────
N_SPLITS    = 5
N_REPEATS   = 3
RANDOM_SEED = 42

# ── Model hyperparameter grids ────────────────────────────────────────────────
# Defined in utils.py (MODELS_CONFIG) — do not duplicate here.
