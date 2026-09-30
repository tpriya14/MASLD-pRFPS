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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False

MODELS_CONFIG = {
    "RandomForest": {
        "class": RandomForestClassifier,
        "params": {"n_estimators": 200, "max_depth": 6, "random_state": RANDOM_SEED,
                   "class_weight": "balanced"},
        "grid": {
            "n_estimators": [100, 200],
            "max_depth":    [4, 6, 8],
            "min_samples_split": [2, 5],
        },
    },
    "MLP": {
        "class": MLPClassifier,
        "params": {"hidden_layer_sizes": (64, 32), "max_iter": 500,
                   "random_state": RANDOM_SEED},
        "grid": {
            "hidden_layer_sizes": [(64,), (64, 32), (128, 64)],
            "alpha": [1e-4, 1e-3],
        },
    },
    "GradientBoosting": {
        "class": GradientBoostingClassifier,
        "params": {"n_estimators": 150, "max_depth": 4, "random_state": RANDOM_SEED},
        "grid": {
            "n_estimators": [100, 150],
            "max_depth":    [3, 4, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "SVM": {
        "class": SVC,
        "params": {"probability": True, "kernel": "rbf", "random_state": RANDOM_SEED},
        "grid": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    },
    "LogisticRegression": {
        "class": LogisticRegression,
        "params": {"max_iter": 1000, "class_weight": "balanced",
                   "random_state": RANDOM_SEED},
        "grid": {"C": [0.01, 0.1, 1, 10]},
    },
    "KNN": {
        "class": KNeighborsClassifier,
        "params": {"n_neighbors": 7},
        "grid": {"n_neighbors": [3, 5, 7, 11]},
    },
    "DecisionTree": {
        "class": DecisionTreeClassifier,
        "params": {"max_depth": 5, "class_weight": "balanced",
                   "random_state": RANDOM_SEED},
        "grid": {"max_depth": [3, 5, 7], "min_samples_leaf": [1, 2, 5]},
    },
}

if _HAS_XGB:
    MODELS_CONFIG["XGBoost"] = {
        "class": XGBClassifier,
        "params": {"n_estimators": 150, "max_depth": 4, "use_label_encoder": False,
                   "eval_metric": "logloss", "random_state": RANDOM_SEED,
                   "scale_pos_weight": 3},
        "grid": {
            "n_estimators": [100, 150],
            "max_depth":    [3, 4],
            "learning_rate": [0.05, 0.1],
        },
    }

if _HAS_LGB:
    MODELS_CONFIG["LightGBM"] = {
        "class": LGBMClassifier,
        "params": {"n_estimators": 150, "max_depth": 4, "class_weight": "balanced",
                   "random_state": RANDOM_SEED, "verbose": -1},
        "grid": {
            "n_estimators": [100, 150],
            "max_depth":    [3, 4],
            "learning_rate": [0.05, 0.1],
        },
    }
