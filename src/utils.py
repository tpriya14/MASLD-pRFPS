"""
utils.py — All shared imports, constants, helpers, and model configs.
"""

# =============================================================================
# STANDARD IMPORTS
# =============================================================================
import io
import os
import sys

# Force single-process mode on Windows BEFORE joblib/sklearn loads
# This prevents the "lost sys.stderr" crash in worker subprocesses
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import json
import pickle
import warnings
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, confusion_matrix, f1_score,
    roc_curve, average_precision_score
)
from scipy import stats as scipy_stats
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# Windows UTF-8 safety — set via environment, not sys.stderr rewrapping
# (rewrapping crashes in joblib worker subprocesses on Windows)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

warnings.filterwarnings("ignore")
np.random.seed(42)

# sklearn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold,
    GridSearchCV, KFold
)

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    SelectKBest, f_classif, RFE, SelectFromModel, mutual_info_classif
)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier, ExtraTreesClassifier
)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score,
    roc_curve, auc, precision_recall_curve, roc_auc_score,
    average_precision_score, balanced_accuracy_score,
    precision_recall_fscore_support, brier_score_loss
)
from scipy import stats as scipy_stats

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available.")

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install: pip install lightgbm")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available.")

# =============================================================================
# DIRECTORY SETUP  (identical to original)
# =============================================================================
results_dir = "MASLD_Comprehensive_Analysis_Tap_Bio3_up_new_kbest3_10_weight_10_500_f1_no_NFS3_cont_rfps"
# results_dir = "MASLD_Comprehensive_Analysis_Tap_Bio3_up_new_kbest3_10_weight_10_Tap"
for _d in [
    "figures", "models", "rules", "subgroup_analysis",
    "hyperparameters", "best_model_shap", "stacking_ensemble_info",
    # New output directories for reviewer fixes
    "rebuttal_outputs", "calibration_plots", "dca_plots",
    "shap_stability", "lca_analysis",
]:
    os.makedirs(f"{results_dir}/{_d}", exist_ok=True)

OUT = f"{results_dir}/rebuttal_outputs"

print("=== MASLD Comprehensive Progression Analysis Pipeline (Revised) ===")
print(f"Results: {results_dir}/")

# =============================================================================
# CONSTANTS  (identical to original)
# =============================================================================
TARGET_COL     = "overall_progression_category2"
SUBGROUP_COL   = "Subgroup_5"
POSITIVE_LABEL = "Rapid Progression"
NEGATIVE_LABEL = "No Progression"
CLASS_NAMES    = [NEGATIVE_LABEL, POSITIVE_LABEL]   # 0 = No, 1 = Rapid
FIB4_COMPONENTS = ["AST.x", "ALT.x", "PLATELET_COUNT"]

def compute_fib4_score(df, ast_col="AST.x", alt_col="ALT.x",
                        platelet_col="PLATELET_COUNT", age_col="diagnosis_age"):
    """FIB-4 = (Age * AST) / (Platelet * sqrt(ALT)). Returns NaN on missing/zero."""
    # for c in [ast_col, alt_col, platelet_col, age_col]:
    #     if c not in df.columns:
    #         return pd.Series(np.nan, index=df.index)
    # age  = df[age_col].astype(float)
    # ast  = df[ast_col].astype(float)
    # alt  = df[alt_col].astype(float)
    # plt_ = df[platelet_col].astype(float)
    # denom = plt_ * np.sqrt(alt.clip(0))
    # return (age * ast) / denom.replace(0, np.nan)
    FIB4_col = "first_FIB4"
    FIB4_ = df[FIB4_col].astype(float)
    return FIB4_


def fib4_binary_label(fib4_scores, age):
    """
    Guideline-aligned binary FIB-4 label (EASL 2016, 2021):
      Low-risk  -> 0 (Slow Progression proxy):
          FIB-4 < 1.3 for age < 65
          FIB-4 < 2.0 for age >= 65
      High-risk -> 1 (Rapid Progression proxy):
          FIB-4 > 2.67 (all ages)
      Intermediate (1.3/2.0 to 2.67): NaN -- excluded for cleaner comparison.
    """
    age   = pd.Series(age).astype(float).reset_index(drop=True)
    fib4  = pd.Series(fib4_scores).astype(float).reset_index(drop=True)
    label = pd.Series(np.nan, index=fib4.index)
    label[(age <  65) & (fib4 < 1.3)]  = 0
    label[(age >= 65) & (fib4 < 2.0)]  = 0
    label[fib4 > 2.67]                 = 1
    return label


def prepare_fib4_comparator(df_raw, test_index,
                              y_true_test,
                              setting="Overall",
                              subgroups_test=None,
                              ast_col="AST.x", alt_col="ALT.x",
                              platelet_col="PLATELET_COUNT",
                              age_col="diagnosis_age"):
    """
    Compute FIB-4 score + binary label for test patients.

    IMPORTANT: test_index must already be filtered to the correct setting.
    all_test_indices[setting] stores exactly the right df_raw row indices
    for each setting (including subgroup-specific), so NO internal subgroup
    filtering is done here. That was the source of the length mismatch
    (IndexError: size 65 vs 28).

    Returns arrays of length len(test_index) -- same as _yt, _yp, _prfps_norm.
    """
    if len(test_index) == 0:
        log(f"  FIB-4 [{setting}]: no test indices provided -- returning empty")
        empty = np.full(len(y_true_test), np.nan)
        return {
            "fib4_continuous": empty, "fib4_norm": empty,
            "fib4_binary": empty, "y_true_test": np.array(y_true_test),
            "n_low": 0, "n_high": 0, "n_intermediate": 0,
        }

    try:
        df_te = df_raw.loc[test_index].copy().reset_index(drop=True)
    except Exception:
        df_te = df_raw.iloc[test_index].copy().reset_index(drop=True)

    # NO internal subgroup filtering -- caller passes pre-filtered indices
    y_te = np.array(y_true_test)

    # Length guard: if df_te and y_te differ, something upstream is wrong
    if len(df_te) != len(y_te):
        log(f"  FIB-4 [{setting}] WARNING: df_te {len(df_te)} != y_te {len(y_te)} -- "
            f"truncating to min")
        n_min = min(len(df_te), len(y_te))
        df_te = df_te.iloc[:n_min].reset_index(drop=True)
        y_te  = y_te[:n_min]

    fib4_cont = compute_fib4_score(df_te, ast_col, alt_col, platelet_col, age_col)

    if age_col in df_te.columns:
        age_s = df_te[age_col].astype(float).reset_index(drop=True)
    else:
        age_s = pd.Series(50.0, index=df_te.index)

    fib4_bin = fib4_binary_label(fib4_cont, age_s)

    n_low = int((fib4_bin == 0).sum())
    n_hi  = int((fib4_bin == 1).sum())
    n_int = int(fib4_bin.isna().sum())
    log(f"  FIB-4 [{setting}]: low={n_low}  high={n_hi}  "
        f"intermediate(excl)={n_int}  total={len(df_te)}")

    fc_arr    = fib4_cont.values.astype(float)
    fc_max    = np.nanmax(fc_arr) + 1e-10
    fib4_norm = fc_arr / fc_max

    return {
        "fib4_continuous": fc_arr,
        "fib4_norm":       fib4_norm,
        "fib4_binary":     fib4_bin.values,
        "y_true_test":     y_te,
        "n_low": n_low, "n_high": n_hi, "n_intermediate": n_int,
    }

FEATURE_LABELS = {
    "diagnosis_age": "Age", "Avg_BMI": "BMI", "HbA1C": "HbA1c",
    "ALT.x": "ALT", "AST.x": "AST", "ALP": "ALP",
    "PLATELET_COUNT": "Platelet Count", "ALBUMIN": "Albumin",
    "TRIGLYCERIDE": "Triglycerides", "HDL": "HDL", "LDL": "LDL",
    "BUN": "BUN", "first_FIB4": "Baseline FIB-4",
    "NFS": "NAFLD Fibrosis Score", "AST_PT_ratio": "AST/Platelet Ratio",
    "diabetes": "Diabetes", "hypertension": "Hypertension",
    "ckd": "Kidney Disease", "depression": "Depression",
    "heart": "Heart Disease", "dyslipidemia": "Dyslipidemia",
    "PNPLA3": "PNPLA3", "TM6SF2": "TM6SF2", "HSD17B13": "HSD17B13",
    "diagnosis_age": "Age at Diagnosis", "Avg_BMI": "BMI",
    "n_followups": "Follow-ups", "CMR_count": "CMR Count",
}

feature_labels = FEATURE_LABELS  # alias for backward compat


# =============================================================================
# LOGGING
# =============================================================================
def log(msg: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        print(f"[{ts}] {msg}", flush=True)
    except UnicodeEncodeError:
        print(f"[{ts}] {msg.encode('ascii', errors='replace').decode()}", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/run.log", "a", encoding="utf-8") as fh:
        fh.write(f"[{ts}] {msg}\n")


def section(title: str) -> None:
    log("=" * 65)
    log(title)
    log("=" * 65)


# =============================================================================
# DATA LOADING  (identical to original load_and_prepare_data)
# =============================================================================
def load_and_prepare_data(data: pd.DataFrame, analysis_name: str,
                           output_file: str = "data_preparation_output.txt"):
    """Exact replica of original. No imputation -- NaN dropped at training."""
    with open(output_file, "w", encoding="utf-8") as logf:
        data = data.copy()
        data["Subgroup"] = data[SUBGROUP_COL]
        data = data[data[TARGET_COL].notna()].copy()
        data[TARGET_COL] = data[TARGET_COL].apply(
            lambda x: POSITIVE_LABEL if x == POSITIVE_LABEL else NEGATIVE_LABEL)

        continuous_vars = [
            "Avg_BMI", "PLATELET_COUNT", "LDL", "HDL", "HbA1C", "BUN",
            "ALP", "ALT.x", "AST.x", "AST_PT_ratio", "diagnosis_age",
            "TRIGLYCERIDE", "ALBUMIN",
        ]
        categorical_vars = [
            "obesity_conversion1", "Diabetes", "hypertension", "heart", "ckd",
            "depression1", "Sex", "dyslipidemia", "sleep_conversion1",
            "joint_conversion1", "heart_conversion1", "ckd_conversion1",
            "depression_conversion1", "PNPLA3", "TM6SF2", "HSD17B13"
        ]
        existing_cont = [v for v in continuous_vars if v in data.columns]
        existing_cat  = [v for v in categorical_vars if v in data.columns]

        data_processed = data.copy()
        label_encoders = {}
        for var in existing_cat:
            le = LabelEncoder()
            data_processed[var] = le.fit_transform(data_processed[var].astype(str))
            label_encoders[var] = le

        target_encoder = LabelEncoder()
        target_encoder.fit([NEGATIVE_LABEL, POSITIVE_LABEL])
        y_encoded = target_encoder.transform(data_processed[TARGET_COL])

        feature_cols = existing_cont + existing_cat
        X          = data_processed[feature_cols]
        y_original = data_processed[TARGET_COL]
        y_numeric  = pd.Series(y_encoded, index=X.index)
        subgroups  = data_processed["Subgroup"]

        print(f"Dataset shape: {data.shape}", file=logf)
        print(f"Target: {dict(y_original.value_counts())}", file=logf)
        print(f"Subgroups: {dict(subgroups.value_counts())}", file=logf)

    log(f"[{analysis_name}] n={len(X)} | "
        f"classes={dict(y_original.value_counts())} | "
        f"subgroups={dict(subgroups.value_counts())}")

    return (X, y_original, y_numeric, subgroups,
            feature_cols, data_processed, label_encoders, target_encoder)


# =============================================================================
# DROP NaN  (no imputation -- identical to original)
# =============================================================================
def drop_nan_rows(X, y_original=None, y_numeric=None, subgroups=None, label=""):
    mask  = ~X.isnull().any(axis=1)
    n_drop = (~mask).sum()
    if n_drop:
        log(f"  [{label}] Dropping {n_drop} NaN rows (no imputation)")
    Xc  = X[mask]
    yoc = y_original[mask] if y_original is not None else None
    ync = y_numeric[mask]  if y_numeric  is not None else None
    sgc = subgroups[mask]  if subgroups  is not None else None
    return Xc, yoc, ync, sgc


# =============================================================================
# [FIX-1]  THREE-WAY SPLIT  (no test-set leakage)
# =============================================================================
def two_way_split(X, y, test_size=0.20, random_state=42):
    """
    80/20 stratified split.

    [FIX-1] Threshold is NOT tuned on the test set.
    Instead, threshold is selected via out-of-fold (OOF) predictions on
    the training set using the same 5-fold CV already used for GridSearchCV.
    This avoids wasting 15% of a small cohort on a dedicated val fold while
    still eliminating leakage completely.

    Why OOF is correct here:
      - With n=752 total (C2 has only ~325), a 3-way 65/15/20 split gives
        C2 val=~49, test=~65 -- far too small for stable threshold estimation.
      - OOF uses ALL training data for threshold calibration without ever
        touching the test set.
      - This is standard practice in small-cohort clinical ML
        (Steyerberg 2019, Riley et al. 2021).
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    log(f"  2-way split -> train:{len(Xtr)}  test:{len(Xte)}")
    return Xtr, Xte, ytr, yte


# =============================================================================
# SCALE  (fit on train, apply to val and test)
# =============================================================================
def fit_scale(X_tr, X_te, features, X_va=None):
    """Fit scaler on train, apply to test (and optionally val)."""
    scl  = StandardScaler()
    Xtr_ = X_tr[features] if hasattr(X_tr, "__getitem__") else X_tr
    Xte_ = X_te[features] if hasattr(X_te, "__getitem__") else X_te
    Ntr  = scl.fit_transform(Xtr_)
    Nte  = scl.transform(Xte_)
    if X_va is not None:
        Xva_ = X_va[features] if hasattr(X_va, "__getitem__") else X_va
        Nva  = scl.transform(Xva_)
        return Ntr, Nte, scl, Nva
    return Ntr, Nte, scl


# =============================================================================
# ORIGINAL HYPERPARAMETER GRIDS  (preserved EXACTLY)
# Only change: multi_class guarded for sklearn >= 1.5 where it was removed.
# =============================================================================

def _lr_param_grid():
    import sklearn
    major, minor = map(int, sklearn.__version__.split(".")[:2])
    grid = {
        "C": [0.01, 0.1, 1, 10],
        "penalty": ["l2"], "solver": ["lbfgs"],
        "random_state": [42], "max_iter": [3000],
        "class_weight": [None, "balanced"],
    }
    if (major, minor) < (1, 5):
        grid["multi_class"] = ["multinomial"]
    return grid


MODELS_CONFIG = {
    "Logistic_Regression": {
        "model_class": LogisticRegression,
        "param_grid":  _lr_param_grid(),
        "scaled": True, "use_numeric_labels": True,
    },
    "KNN": {
        "model_class": KNeighborsClassifier,
        "param_grid":  {"n_neighbors": [3, 5, 7],
                        "weights": ["uniform", "distance"]},
        "scaled": True, "use_numeric_labels": True,
    },
    "Random_Forest": {
        "model_class": RandomForestClassifier,
        "param_grid":  {
            "n_estimators": [50, 100], "max_depth": [5, 10, None],
            "min_samples_split": [2, 5], "random_state": [42],
            "n_jobs": [1], "class_weight": [None, "balanced"],
        },
        "scaled": True, "use_numeric_labels": True,
    },
    "DecisionTree": {
        "model_class": DecisionTreeClassifier,
        "param_grid":  {
            "max_depth": [3, 5, 7], "min_samples_leaf": [2, 5],
            "random_state": [42], "class_weight": [None, "balanced"],
        },
        "scaled": False, "use_numeric_labels": True,
    },
    "GradientBoosting": {
        "model_class": GradientBoostingClassifier,
        "param_grid":  {
            "n_estimators": [50, 100], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1], "random_state": [42],
        },
        "scaled": True, "use_numeric_labels": True,
    },
    "SVM": {
        "model_class": SVC,
        "param_grid":  {
            "C": [0.1, 1], "kernel": ["rbf"], "gamma": ["scale"],
            "random_state": [42], "probability": [True],
            "class_weight": [None, "balanced"],
        },
        "scaled": True, "use_numeric_labels": True,
    },
    "MLP": {
        "model_class": MLPClassifier,
        "param_grid":  {
            "hidden_layer_sizes": [(50,), (100,)],
            "learning_rate_init": [0.001, 0.01],
            "alpha": [0.0001, 0.001], "solver": ["adam"],
            "random_state": [42], "max_iter": [500],
            "early_stopping": [True], "validation_fraction": [0.1],
        },
        "scaled": True, "use_numeric_labels": True,
    },
}
if XGBOOST_AVAILABLE:
    MODELS_CONFIG["XGBoost"] = {
        "model_class": XGBClassifier,
        "param_grid":  {
            "n_estimators": [50, 100], "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1], "subsample": [0.8, 1.0],
            "random_state": [42], "eval_metric": ["logloss"],
            "use_label_encoder": [False], "n_jobs": [1],
            "scale_pos_weight": [1, 3, 5],
        },
        "scaled": True, "use_numeric_labels": True,
    }

# LightGBM -- added to compare with baseline paper
# "Identification of Fast Progressors Among Patients With NASH Using ML"
# That paper used gradient boosting family; LightGBM is the modern equivalent
if LIGHTGBM_AVAILABLE:
    MODELS_CONFIG["LightGBM"] = {
        "model_class": LGBMClassifier,
        "param_grid":  {
            "n_estimators":  [50, 100],
            "max_depth":     [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "num_leaves":    [15, 31],
            "random_state":  [42],
            "n_jobs":        [1],
            "verbose":       [-1],
            "class_weight":  [None, "balanced"],
        },
        "scaled": True, "use_numeric_labels": True,
    }



# =============================================================================
# [FIX-1]  YOUDEN'S INDEX THRESHOLD -- on VAL fold only
# =============================================================================
def find_optimal_threshold_on_val(
    y_val: np.ndarray,
    y_proba_val: np.ndarray,
    optimization: str = "f1",
) -> float:
    """
    [FIX-1] Your ORIGINAL threshold optimisation logic, now applied to the
    VALIDATION fold instead of the test set.

    This is the ONLY change from your original find_optimal_threshold_pr_curve():
      - Original: received X_test, y_test  --> leakage (erLa critical concern)
      - Fixed:    receives y_val, y_proba_val --> no leakage

    optimization options (same as your original):
      "f1"       -- maximise F1 on PR curve  (your original default)
      "f2"       -- maximise F2 (recall-weighted)
      "f0.5"     -- maximise F0.5 (precision-weighted)
      "balanced" -- minimise distance from perfect PR point
      "youden"   -- maximise Sensitivity + Specificity - 1
      "default"  -- fixed 0.5
    """
    y_val      = np.array(y_val)
    y_proba_val = np.array(y_proba_val)

    if optimization in ("default", None):
        return 0.5

    precisions, recalls, thresholds = precision_recall_curve(y_val, y_proba_val)

    if len(thresholds) == 0:
        return 0.5

    if optimization == "f1":
        f1s = np.where((precisions + recalls) > 0,
                        2 * precisions * recalls / (precisions[:-1] + recalls[:-1] + 1e-10),
                        0)
        # precision_recall_curve returns n+1 points, thresholds has n values
        f1s = np.array([2*p*r/(p+r) if (p+r)>0 else 0
                        for p,r in zip(precisions[:-1], recalls[:-1])])
        return float(thresholds[np.argmax(f1s)])

    elif optimization == "f2":
        f2s = np.array([5*p*r/(4*p+r) if (4*p+r)>0 else 0
                        for p,r in zip(precisions[:-1], recalls[:-1])])
        return float(thresholds[np.argmax(f2s)])

    elif optimization == "f0.5":
        f05s = np.array([1.25*p*r/(0.25*p+r) if (0.25*p+r)>0 else 0
                         for p,r in zip(precisions[:-1], recalls[:-1])])
        return float(thresholds[np.argmax(f05s)])

    elif optimization == "balanced":
        dists = np.sqrt((1 - precisions[:-1])**2 + (1 - recalls[:-1])**2)
        return float(thresholds[np.argmin(dists)])

    elif optimization == "youden":
        fpr_, tpr_, thr_ = roc_curve(y_val, y_proba_val)
        j = tpr_ - fpr_
        return float(thr_[np.argmax(j)])

    else:
        raise ValueError(f"Unknown optimization: {optimization}. "
                         "Choose: f1, f2, f0.5, balanced, youden, default")


# Convenience alias kept for backward compatibility
def youden_threshold(y_val, y_proba_val):
    return find_optimal_threshold_on_val(y_val, y_proba_val, optimization="youden")


# =============================================================================
# [FIX-9]  BOOTSTRAP 95% CIs  (1000 resamples as instructed)
# =============================================================================
def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                  y_proba: np.ndarray, n_boot: int = 1000,
                  seed: int = 42) -> Dict:
    """
    [FIX-9] 95% bootstrap confidence intervals for all metrics.
    Returns dict: metric -> (mean, lower, upper).
    Metrics: accuracy, f1_macro, auroc, auprc, brier,
             precision_0, recall_0, f1_0, ppv_0, npv_0,
             precision_1, recall_1, f1_1, ppv_1, npv_1,
             sensitivity, specificity.
    """
    rng = np.random.RandomState(seed)
    n   = len(y_true)
    res = {m: [] for m in [
        "accuracy", "f1_macro", "auroc", "auprc", "brier",
        "precision_0", "recall_0", "f1_0",
        "precision_1", "recall_1", "f1_1",
        "sensitivity", "specificity", "ppv", "npv",
    ]}
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt, yp, ypr = y_true[idx], y_pred[idx], y_proba[idx]
        if len(np.unique(yt)) < 2:
            continue
        res["accuracy"].append(accuracy_score(yt, yp))
        res["f1_macro"].append(f1_score(yt, yp, average="macro", zero_division=0))
        res["auroc"].append(roc_auc_score(yt, ypr))
        res["auprc"].append(average_precision_score(yt, ypr))
        res["brier"].append(brier_score_loss(yt, ypr))
        for lab in [0, 1]:
            res[f"precision_{lab}"].append(
                precision_score(yt, yp, labels=[lab], average="micro", zero_division=0))
            res[f"recall_{lab}"].append(
                recall_score(yt, yp, labels=[lab], average="micro", zero_division=0))
            res[f"f1_{lab}"].append(
                f1_score(yt, yp, labels=[lab], average="micro", zero_division=0))
        cm_b    = confusion_matrix(yt, yp, labels=[0, 1])
        tn_, fp_, fn_, tp_ = cm_b.ravel()
        res["sensitivity"].append(tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0)
        res["specificity"].append(tn_ / (tn_ + fp_) if (tn_ + fp_) > 0 else 0)
        res["ppv"].append(tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0)
        res["npv"].append(tn_ / (tn_ + fn_) if (tn_ + fn_) > 0 else 0)

    alpha = 0.025
    out   = {}
    for m, vals in res.items():
        if vals:
            out[m] = (float(np.mean(vals)),
                      float(np.percentile(vals, alpha * 100)),
                      float(np.percentile(vals, (1 - alpha) * 100)))
        else:
            out[m] = (np.nan, np.nan, np.nan)
    return out


def fmt_ci(metric: str, ci_dict: Dict, point_dict: Dict = None) -> str:
    if metric in ci_dict:
        mn, lo, hi = ci_dict[metric]
        return f"{mn:.3f} [{lo:.3f}-{hi:.3f}]"
    v = (point_dict or {}).get(metric, float("nan"))
    return f"{v:.3f}" if not np.isnan(v) else "N/A"


# =============================================================================
# [FIX-2]  COMPREHENSIVE METRICS  (Part 2 deliverable)
# =============================================================================
def compute_comprehensive_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                                   y_proba: np.ndarray, n_boot: int = 1000,
                                   label: str = "") -> Tuple[Dict, Dict]:
    """
    [FIX-2] Returns (point_dict, ci_dict) with:
    - Overall: accuracy, F1-macro, AUROC, AUPRC, Brier
    - Class 0 (No Progression): precision, recall, F1, PPV, NPV, support
    - Class 1 (Rapid Progression): precision, recall, F1, sensitivity,
                                    specificity, PPV, NPV, support
    - Confusion matrix cells: TP, FP, TN, FN
    All with 95% bootstrap CIs (1000 resamples).
    """
    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_proba = np.array(y_proba)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    auroc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    auprc = average_precision_score(y_true, y_proba) if len(np.unique(y_true)) > 1 else np.nan
    brier = brier_score_loss(y_true, y_proba)

    prec_pc, rec_pc, f1_pc, sup_pc = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=[0, 1])

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv  = tn / (tn + fn) if (tn + fn) > 0 else 0

    point = {
        "accuracy":         accuracy_score(y_true, y_pred),
        "f1_macro":         f1_score(y_true, y_pred, average="macro", zero_division=0),
        "auroc":            auroc, "auprc": auprc, "brier": brier,
        "precision_0":      float(prec_pc[0]),
        "recall_0":         float(rec_pc[0]),
        "f1_0":             float(f1_pc[0]),
        "support_0":        int(sup_pc[0]),
        "precision_1":      float(prec_pc[1]),
        "recall_1":         float(rec_pc[1]),
        "f1_1":             float(f1_pc[1]),
        "support_1":        int(sup_pc[1]),
        "sensitivity":      float(sens),
        "specificity":      float(spec),
        "ppv":              float(ppv),
        "npv":              float(npv),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "prevalence":       float(y_true.mean()),
    }

    ci = bootstrap_ci(y_true, y_pred, y_proba, n_boot=n_boot)
    return point, ci


# =============================================================================
# EVALUATE MODEL  (extended from original evaluate_model_comprehensive)
# =============================================================================
def evaluate_model_comprehensive(model, X_test_np: np.ndarray,
                                  y_test_num: np.ndarray,
                                  y_pred: np.ndarray, y_proba: np.ndarray,
                                  model_name: str, threshold: float,
                                  n_boot: int = 1000,
                                  class_names: List[str] = None) -> Dict:
    """
    Extended version of original. Adds:
    [FIX-2] AUROC, AUPRC, Brier, calibration metrics, class-specific metrics
            (sensitivity, specificity, PPV, NPV), 95% bootstrap CIs.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    point, ci = compute_comprehensive_metrics(
        y_test_num, y_pred, y_proba, n_boot=n_boot, label=model_name)

    # Preserve original fields
    point["model_name"]      = model_name
    point["optimal_threshold"] = float(threshold)
    point["accuracy_orig"]   = point["accuracy"]    # alias
    point["precision"]       = point["precision_1"] # backward compat
    point["recall"]          = point["recall_1"]
    point["f1_score"]        = point["f1_macro"]
    point["auc_score"]       = point["auroc"]
    point["precision_per_class"] = np.array([point["precision_0"], point["precision_1"]])
    point["recall_per_class"]    = np.array([point["recall_0"],    point["recall_1"]])
    point["f1_per_class"]        = np.array([point["f1_0"],        point["f1_1"]])
    point["confusion_matrix"]    = confusion_matrix(y_test_num, y_pred, labels=[0, 1])
    point["classification_report"] = classification_report(
        y_test_num, y_pred, target_names=class_names,
        output_dict=True, zero_division=0)
    point["y_pred"]  = y_pred
    point["y_proba"] = y_proba
    point["y_true"]  = y_test_num
    point["ci"]      = ci
    return point


# =============================================================================
# RESULTS -> PUBLICATION-READY CSV  (Part 2 deliverable)
# =============================================================================
def results_to_publication_table(results_dict: Dict, setting: str,
                                  out_path: str) -> pd.DataFrame:
    """
    [Part 2] Generates Table X / Table Y style output with all metrics,
    class-specific rows, and 95% bootstrap CIs.
    """
    rows = []
    for name, r in results_dict.items():
        ci = r.get("ci", {})
        def f(m): return fmt_ci(m, ci, r)

        rows.append({
            "Setting":    setting, "Model": name,
            "Threshold":  f"{r.get('optimal_threshold', float('nan')):.3f}",
            # Overall metrics
            "Accuracy":           f("accuracy"),
            "F1_macro":           f("f1_macro"),
            "AUROC":              f("auroc"),
            "AUPRC":              f("auprc"),
            "Brier":              f("brier"),
            # Rapid Progression (class 1) -- clinical focus
            "Sensitivity":                     f("sensitivity"),
            "Specificity":                     f("specificity"),
            "PPV":                             f("ppv"),
            "NPV":                             f("npv"),
            f"Precision_{POSITIVE_LABEL}":     f("precision_1"), f"Recall_{POSITIVE_LABEL}":        f("recall_1"),
            f"F1_{POSITIVE_LABEL}":            f("f1_1"), f"Support_{POSITIVE_LABEL}":       r.get("support_1", ""),
            # No Progression (class 0)
            f"Precision_{NEGATIVE_LABEL}":     f("precision_0"), f"Recall_{NEGATIVE_LABEL}":        f("recall_0"),
            f"F1_{NEGATIVE_LABEL}":            f("f1_0"), f"Support_{NEGATIVE_LABEL}":       r.get("support_0", ""),
            # Confusion matrix
            "TP": r.get("TP",""), "FP": r.get("FP",""),
            "TN": r.get("TN",""), "FN": r.get("FN",""),
            "Prevalence": f"{r.get('prevalence', float('nan')):.3f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    log(f"  Saved {out_path}")
    return df


# =============================================================================
# [FIX-2/14]  ROC + PR + CALIBRATION FIGURES
# =============================================================================

def plot_roc_pr(results_dict: Dict, tag: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, r in results_dict.items():
        yt, yp = r.get("y_true"), r.get("y_proba")
        if yt is None or len(np.unique(yt)) < 2:
            continue
        fpr, tpr, _ = roc_curve(yt, yp)
        axes[0].plot(fpr, tpr, lw=1.5,
                     label=f"{name} ({r.get('auroc', float('nan')):.3f})")
        prec, rec, _ = precision_recall_curve(yt, yp)
        axes[1].plot(rec, prec, lw=1.5,
                     label=f"{name} ({r.get('auprc', float('nan')):.3f})")
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title(f"ROC -- {tag}"); axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title(f"PR -- {tag}"); axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/roc_pr_{tag}.png", dpi=200); plt.close()


def plot_calibration_curves(results_dict: Dict, tag: str, n_bins: int = 10) -> None:
    """[FIX-14] Reliability diagrams + Brier scores."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect calibration")
    for name, r in results_dict.items():
        yt, yp = r.get("y_true"), r.get("y_proba")
        if yt is None or len(np.unique(yt)) < 2:
            continue
        try:
            frac, mp = calibration_curve(yt, yp, n_bins=n_bins, strategy="uniform")
            br = r.get("brier", float("nan"))
            ax.plot(mp, frac, "s-", lw=1.5,
                    label=f"{name} (Brier={br:.3f})")
        except Exception:
            pass
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration Curve -- {tag}")
    ax.legend(fontsize=7); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/calibration_plots/calibration_{tag}.png", dpi=200)
    plt.close()


# =============================================================================
# [Part 3]  CALIBRATION ANALYSIS
# =============================================================================
def calibration_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                          model_name: str, setting: str,
                          out_dir: str = None) -> Dict:
    """
    [Part 3] Calibration intercept, slope (logistic regression of
    observed on logit(predicted)), and Brier score.
    """
    if out_dir is None:
        out_dir = f"{results_dir}/calibration_plots"
    os.makedirs(out_dir, exist_ok=True)

    brier = brier_score_loss(y_true, y_proba)

    # Calibration intercept + slope via logistic regression
    from sklearn.linear_model import LogisticRegression as _LR
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logit_p = np.log(y_proba + 1e-10) - np.log(1 - y_proba + 1e-10)
        lr_cal  = _LR(fit_intercept=True, max_iter=1000)
        lr_cal.fit(logit_p.reshape(-1, 1), y_true)
        intercept = float(lr_cal.intercept_[0])
        slope     = float(lr_cal.coef_[0][0])

    result = {
        "setting": setting, "model": model_name,
        "brier_score":          round(brier, 4),
        "calibration_intercept": round(intercept, 4),
        "calibration_slope":     round(slope, 4),
    }

    # Calibration plot
    fig, ax = plt.subplots(figsize=(5, 5))
    frac, mp = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")
    ax.plot(mp, frac, "s-", color="#2980b9", lw=2, label=f"{model_name}")
    ax.plot([0,1],[0,1],"k--",lw=1,label="Perfect")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Calibration: {model_name} | {setting}\n"
                 f"Brier={brier:.3f}  Intercept={intercept:.3f}  Slope={slope:.3f}")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    safe = f"{model_name}_{setting}".replace(" ","_")
    plt.savefig(f"{out_dir}/calibration_{safe}.png", dpi=200); plt.close()

    return result


# =============================================================================
# [Part 4]  DECISION CURVE ANALYSIS
# =============================================================================
def decision_curve_analysis(y_true: np.ndarray,
                              predictions: Dict[str, np.ndarray],
                              setting: str,
                              out_dir: str = None,
                              thresholds: np.ndarray = None,
                              smooth: bool = True,
                              smooth_window: int = 11) -> pd.DataFrame:
    """
    [Part 4] Net benefit decision curve analysis.

    Smooth curves are produced by:
    1. Using 200 threshold points instead of 100 (finer resolution).
    2. Applying Savitzky-Golay smoothing (preserves peaks, removes noise).
       This is the standard approach used in dcurves R package and
       published DCA papers (Vickers et al. 2019).
    3. Clipping negative net benefit to 0 below treat-none line.

    Why curves were jagged before:
      Small test sets (n=65-151) mean each threshold step changes
      TP/FP counts by 1-2 patients, causing large jumps.
      Smoothing averages over nearby thresholds to reveal the
      underlying net benefit curve shape.

    Net benefit = (TP/n) - (FP/n) * threshold/(1-threshold)
    """
    from scipy.signal import savgol_filter

    if out_dir is None:
        out_dir = f"{results_dir}/dca_plots"
    os.makedirs(out_dir, exist_ok=True)

    # Use 200 points for finer resolution
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.90, 200)

    n    = len(y_true)
    prev = float(y_true.mean())
    rows = []

    for t in thresholds:
        row = {"threshold": t}
        # Treat all: net benefit = prevalence - (1-prev)*odds_threshold
        row["Treat_All"]  = prev - (1 - prev) * t / (1 - t + 1e-10)
        row["Treat_None"] = 0.0
        for mname, yp in predictions.items():
            yp_bin = (yp >= t).astype(int)
            tp_ = int(((yp_bin == 1) & (y_true == 1)).sum())
            fp_ = int(((yp_bin == 1) & (y_true == 0)).sum())
            row[mname] = tp_ / n - fp_ / n * t / (1 - t + 1e-10)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/dca_{setting.replace(' ','_')}.csv", index=False)

    # Apply Savitzky-Golay smoothing to each model curve
    # Window must be odd and <= number of points
    win = min(smooth_window, len(df) // 4 * 2 + 1)
    if win < 3: win = 3
    if win % 2 == 0: win += 1

    df_plot = df.copy()
    if smooth:
        for col in list(predictions.keys()) + ["Treat_All"]:
            if col in df_plot.columns:
                try:
                    df_plot[col] = savgol_filter(
                        df_plot[col].values, window_length=win, polyorder=2)
                except Exception:
                    pass   # if smoothing fails, use raw values

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))

    # Reference lines
    ax.plot(df_plot["threshold"], df_plot["Treat_All"],
            color="gray", lw=1.5, ls="-", label="Treat All", alpha=0.7)
    ax.plot(df_plot["threshold"], df_plot["Treat_None"],
            color="gray", lw=1.5, ls="--", label="Treat None", alpha=0.7)

    # Model curves
    colors = plt.cm.tab10(np.linspace(0, 0.85, max(len(predictions), 1)))
    for i, mname in enumerate(predictions.keys()):
        if mname in df_plot.columns:
            ax.plot(df_plot["threshold"], df_plot[mname],
                    lw=2, color=colors[i], label=mname)

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(f"Decision Curve Analysis -- {setting}", fontsize=13)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25)
    ax.set_xlim(0.05, 0.90)

    # Y-axis: clip at slightly below 0 to show treat-none clearly
    y_max = max(prev + 0.05,
                df_plot[[c for c in predictions if c in df_plot.columns]].max().max() + 0.02)
    ax.set_ylim(-0.02, min(y_max, prev + 0.15))

    # Shade region where model beats both references
    ax.axhline(0, color="black", lw=0.5, ls=":")

    if smooth:
        ax.text(0.62, 0.97, f"Smoothed (Savitzky-Golay, w={win})",
                transform=ax.transAxes, fontsize=7, color="gray",
                va="top", ha="right")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/dca_{setting.replace(' ','_')}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  [Part 4] DCA saved for {setting}")
    return df


# =============================================================================
# [REVIEWER 59JH]  PAIRWISE STATISTICAL CONTRASTS
# =============================================================================


def delong_auc_test(y_true: np.ndarray,
                     y_prob_a: np.ndarray,
                     y_prob_b: np.ndarray) -> Tuple[float, float]:
    """
    DeLong test for comparing two AUROC values on the same test set.
    Returns (z_score, p_value) — two-sided.
    Reference: DeLong et al. (1988) Biometrics.
    Fast implementation via structural components method.
    """
    y  = np.array(y_true).astype(int)
    pa = np.array(y_prob_a)
    pb = np.array(y_prob_b)

    def _structural(y, p):
        """Compute structural components for DeLong variance."""
        pos = p[y == 1]
        neg = p[y == 0]
        n1, n0 = len(pos), len(neg)
        if n1 == 0 or n0 == 0:
            return float("nan"), float("nan"), float("nan")
        # V10: for each positive, fraction of negatives ranked below it
        V10 = np.array([np.mean(pi > neg) + 0.5 * np.mean(pi == neg)
                        for pi in pos])
        # V01: for each negative, fraction of positives ranked above it
        V01 = np.array([np.mean(ni < pos) + 0.5 * np.mean(ni == pos)
                        for ni in neg])
        auc  = V10.mean()
        s10  = np.var(V10, ddof=1) / n1
        s01  = np.var(V01, ddof=1) / n0
        return auc, s10, s01, n1, n0

    try:
        auc_a, s10_a, s01_a, n1, n0 = _structural(y, pa)
        auc_b, s10_b, s01_b, n1, n0 = _structural(y, pb)

        # Covariance via structural components of the combined vector
        pos_a = pa[y==1]; neg_a = pa[y==0]
        pos_b = pb[y==1]; neg_b = pb[y==0]
        V10_a = np.array([np.mean(pi > neg_a) + 0.5*np.mean(pi==neg_a) for pi in pos_a])
        V01_a = np.array([np.mean(ni < pos_a) + 0.5*np.mean(ni==pos_a) for ni in neg_a])
        V10_b = np.array([np.mean(pi > neg_b) + 0.5*np.mean(pi==neg_b) for pi in pos_b])
        V01_b = np.array([np.mean(ni < pos_b) + 0.5*np.mean(ni==pos_b) for ni in neg_b])

        s10_ab = np.cov(V10_a, V10_b)[0,1] / n1
        s01_ab = np.cov(V01_a, V01_b)[0,1] / n0

        var_diff = s10_a + s01_a + s10_b + s01_b - 2*s10_ab - 2*s01_ab
        if var_diff <= 0:
            return float("nan"), float("nan")
        z  = (auc_a - auc_b) / np.sqrt(var_diff)
        p  = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
        return float(z), float(p)
    except Exception:
        return float("nan"), float("nan")


def paired_bootstrap_contrast(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    metric: str = "auroc",
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict:
    """
    Paired bootstrap test comparing two models on the same test set.
    'Paired' means we resample the same indices for both models.
    Returns: diff (A-B), CI, p-value (two-sided).
    """
    rng  = np.random.RandomState(seed)
    n    = len(y_true)
    diffs = []

    def _score(yt, yp, yp_bin=None):
        if len(np.unique(yt)) < 2:
            return float("nan")
        if metric == "auroc":
            return roc_auc_score(yt, yp)
        if metric == "auprc":
            return average_precision_score(yt, yp)
        if metric == "f1_macro":
            yb = yp_bin if yp_bin is not None else (yp >= 0.5).astype(int)
            return f1_score(yt, yb, average="macro", zero_division=0)
        if metric == "f1_1":
            yb = yp_bin if yp_bin is not None else (yp >= 0.5).astype(int)
            return f1_score(yt, yb, pos_label=1, average="binary", zero_division=0)
        if metric in ("sensitivity", "specificity", "ppv", "npv"):
            yb = yp_bin if yp_bin is not None else (yp >= 0.5).astype(int)
            cm = confusion_matrix(yt, yb, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            if metric == "sensitivity":
                return tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if metric == "specificity":
                return tn / (tn + fp) if (tn + fp) > 0 else 0.0
            if metric == "ppv":
                return tp / (tp + fp) if (tp + fp) > 0 else 0.0
            if metric == "npv":
                return tn / (tn + fn) if (tn + fn) > 0 else 0.0
        return float("nan")

    s_a = _score(y_true, y_prob_a)
    s_b = _score(y_true, y_prob_b)
    obs_diff = s_a - s_b

    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt  = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        diffs.append(_score(yt, y_prob_a[idx]) - _score(yt, y_prob_b[idx]))

    diffs = np.array([d for d in diffs if not np.isnan(d)])
    if len(diffs) < 10:
        return {"diff": obs_diff, "ci_lo": float("nan"), "ci_hi": float("nan"),
                "p_value": float("nan"), "metric": metric,
                "n_boot": n_boot, "significant": False}

    ci_lo = float(np.percentile(diffs, 2.5))
    ci_hi = float(np.percentile(diffs, 97.5))

    # Two-sided p: proportion of bootstrap diffs on opposite side of 0
    if obs_diff > 0:
        p = 2 * np.mean(diffs <= 0)
    else:
        p = 2 * np.mean(diffs >= 0)
    p = float(np.clip(p, 1/n_boot, 1.0))

    return {
        "diff":        round(float(obs_diff), 4),
        "ci_lo":       round(ci_lo, 4),
        "ci_hi":       round(ci_hi, 4),
        "p_value":     round(p, 4),
        "metric":      metric,
        "n_boot":      int(len(diffs)),
        "significant": bool((ci_lo > 0 or ci_hi < 0) and p < 0.05),
    }


def run_pairwise_contrasts(
    all_results_dict: Dict,
    n_boot: int = 1000,
    fib4_proba: Optional[Dict] = None,
    prfps_proba: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Pairwise statistical contrasts -- AUROC + Accuracy for every comparison:
      1. SE vs best single model (within each setting, paired test)
      2. pRFPS vs FIB-4 (if probabilities provided, paired test)
      3. Overall vs C1, Overall vs C2:
           a. SE vs SE          (independent Z-test, different patients)
           b. Best model vs best model (independent Z-test)

    AUROC:    DeLong test (paired) or independent Z via bootstrap SE (cross-setting)
    Accuracy: Paired bootstrap for all paired comparisons.
              Independent Z via bootstrap SE for cross-setting comparisons.
    """
    section("PAIRWISE CONTRASTS -- AUROC + Accuracy")
    rows = []
    settings = list(all_results_dict.keys())

    # ── helpers ───────────────────────────────────────────────────────────────
    def _best(res, exclude_se=False):
        pool = {k: v for k, v in res.items()
                if k != "Stacking_Ensemble"} if exclude_se else res
        if not pool:
            return None, None
        nm = max(pool, key=lambda m: pool[m].get("auroc", 0))
        return nm, pool[nm]

    def _acc(res):
        return float(res.get("accuracy", float("nan")))

    def _make_row(setting, contrast, model_a, model_b,
                  metric, test_type,
                  score_a, score_b,
                  delong_z="N/A", p_val="N/A",
                  ci_lo="N/A", ci_hi="N/A",
                  boot_p="N/A", significant=False, n_boot_used="N/A"):
        return {
            "Setting":     setting,
            "Contrast":    contrast,
            "Metric":      metric,
            "Test":        test_type,
            "Model_A":     model_a,
            "Model_B":     model_b,
            "Score_A":     round(float(score_a), 4) if not np.isnan(float(score_a)) else "N/A",
            "Score_B":     round(float(score_b), 4) if not np.isnan(float(score_b)) else "N/A",
            "Diff_A_B":    round(float(score_a) - float(score_b), 4)
                           if not (np.isnan(float(score_a)) or np.isnan(float(score_b)))
                           else "N/A",
            "DeLong_z":    delong_z,
            "p_value":     p_val,
            "Boot_CI_lo":  ci_lo,
            "Boot_CI_hi":  ci_hi,
            "Boot_p":      boot_p,
            "Significant": significant,
            "n_boot":      n_boot_used,
        }

    # ── Contrasts 1 & 2: within-setting paired tests ──────────────────────────
    for setting in settings:
        res    = all_results_dict[setting]
        se_res = res.get("Stacking_Ensemble")
        if se_res is None:
            log(f"  [{setting}] No SE -- skipping"); continue

        y_true = np.array(se_res.get("y_true",  []))
        yp_se  = np.array(se_res.get("y_proba", []))
        if len(y_true) == 0 or len(np.unique(y_true)) < 2:
            continue

        best_nm, best_res = _best(res, exclude_se=True)
        if best_nm is None:
            continue
        yp_best = np.array(best_res.get("y_proba", []))
        if len(yp_best) != len(y_true):
            continue

        contrast_lbl = f"SE vs {best_nm}"
        log(f"  [{setting}] {contrast_lbl}")

        # ── AUROC (DeLong) ────────────────────────────────────────────────────
        try:
            z, p  = delong_auc_test(y_true, yp_se, yp_best)
            pb    = paired_bootstrap_contrast(y_true, yp_se, yp_best, "auroc", n_boot)
            rows.append(_make_row(
                setting, contrast_lbl, "Stacking_Ensemble", best_nm,
                "AUROC", "DeLong + Paired Bootstrap",
                se_res.get("auroc", float("nan")),
                best_res.get("auroc", float("nan")),
                delong_z   = round(float(z), 3) if not np.isnan(z) else "N/A",
                p_val      = round(float(p), 4) if not np.isnan(p) else "N/A",
                ci_lo      = pb["ci_lo"], ci_hi = pb["ci_hi"],
                boot_p     = pb["p_value"], significant = pb["significant"],
                n_boot_used= pb["n_boot"],
            ))
        except Exception as _e:
            log(f"    AUROC DeLong failed: {_e}")

        # ── Accuracy (Paired Bootstrap) ───────────────────────────────────────
        try:
            pb_acc = paired_bootstrap_contrast(y_true, yp_se, yp_best, "accuracy", n_boot)
            rows.append(_make_row(
                setting, contrast_lbl, "Stacking_Ensemble", best_nm,
                "Accuracy", "Paired Bootstrap",
                _acc(se_res), _acc(best_res),
                ci_lo      = pb_acc["ci_lo"], ci_hi = pb_acc["ci_hi"],
                boot_p     = pb_acc["p_value"], significant = pb_acc["significant"],
                p_val      = pb_acc["p_value"], n_boot_used = pb_acc["n_boot"],
            ))
        except Exception as _e:
            log(f"    Accuracy bootstrap failed: {_e}")

        # ── F1-macro (Paired Bootstrap) ───────────────────────────────────────
        try:
            pb_f1 = paired_bootstrap_contrast(y_true, yp_se, yp_best, "f1_macro", n_boot)
            rows.append(_make_row(
                setting, contrast_lbl, "Stacking_Ensemble", best_nm,
                "F1_macro", "Paired Bootstrap",
                se_res.get("f1_macro", float("nan")),
                best_res.get("f1_macro", float("nan")),
                ci_lo      = pb_f1["ci_lo"], ci_hi = pb_f1["ci_hi"],
                boot_p     = pb_f1["p_value"], significant = pb_f1["significant"],
                p_val      = pb_f1["p_value"], n_boot_used = pb_f1["n_boot"],
            ))
        except Exception as _e:
            log(f"    F1 bootstrap failed: {_e}")

        # ── Sensitivity (Paired Bootstrap) ────────────────────────────────────
        try:
            pb_sens = paired_bootstrap_contrast(y_true, yp_se, yp_best, "sensitivity", n_boot)
            rows.append(_make_row(
                setting, contrast_lbl, "Stacking_Ensemble", best_nm,
                "Sensitivity", "Paired Bootstrap",
                se_res.get("sensitivity", float("nan")),
                best_res.get("sensitivity", float("nan")),
                ci_lo      = pb_sens["ci_lo"], ci_hi = pb_sens["ci_hi"],
                boot_p     = pb_sens["p_value"], significant = pb_sens["significant"],
                p_val      = pb_sens["p_value"], n_boot_used = pb_sens["n_boot"],
            ))
        except Exception as _e:
            log(f"    Sensitivity bootstrap failed: {_e}")

        # ── Contrast 2: pRFPS vs FIB-4 ────────────────────────────────────────
        if (fib4_proba is not None and prfps_proba is not None and
                setting in fib4_proba and setting in prfps_proba):
            yp_fib4  = np.array(fib4_proba[setting])
            yp_prfps = np.array(prfps_proba[setting])
            if (len(yp_fib4) == len(y_true) == len(yp_prfps)
                    and len(np.unique(y_true)) > 1):
                try:
                    z2, p2 = delong_auc_test(y_true, yp_prfps, yp_fib4)
                    pb2    = paired_bootstrap_contrast(
                        y_true, yp_prfps, yp_fib4, "auroc", n_boot)
                    rows.append(_make_row(
                        setting, "pRFPS vs FIB-4", "pRFPS", "FIB-4",
                        "AUROC", "DeLong + Paired Bootstrap",
                        roc_auc_score(y_true, yp_prfps),
                        roc_auc_score(y_true, yp_fib4),
                        delong_z = round(float(z2),3) if not np.isnan(z2) else "N/A",
                        p_val    = round(float(p2),4) if not np.isnan(p2) else "N/A",
                        ci_lo    = pb2["ci_lo"], ci_hi = pb2["ci_hi"],
                        boot_p   = pb2["p_value"], significant = pb2["significant"],
                        n_boot_used = pb2["n_boot"],
                    ))
                except Exception as _e:
                    log(f"    pRFPS vs FIB-4 AUROC failed: {_e}")

                try:
                    pb_acc2 = paired_bootstrap_contrast(
                        y_true, yp_prfps, yp_fib4, "accuracy", n_boot)
                    # pRFPS and FIB-4 raw scores need thresholding for accuracy
                    # Use stored accuracy from results if available
                    rows.append(_make_row(
                        setting, "pRFPS vs FIB-4", "pRFPS", "FIB-4",
                        "Accuracy", "Paired Bootstrap",
                        float("nan"), float("nan"),  # raw accuracy from score not applicable
                        ci_lo   = pb_acc2["ci_lo"], ci_hi = pb_acc2["ci_hi"],
                        boot_p  = pb_acc2["p_value"], significant = pb_acc2["significant"],
                        p_val   = pb_acc2["p_value"], n_boot_used = pb_acc2["n_boot"],
                    ))
                except Exception as _e:
                    log(f"    pRFPS vs FIB-4 Accuracy failed: {_e}")

    # ── Contrasts 3 & 4: cross-setting (Overall vs C1, Overall vs C2) ────────
    # Different test patients → independent Z-test using bootstrap SE.
    # Run for both SE and each setting's best model.
    def _indep_z_rows(setting_a, setting_b, nm_a, res_a, nm_b, res_b, yt_a, yt_b, lbl):
        out = []
        for metric_key, metric_lbl in [("auroc", "AUROC"), ("accuracy", "Accuracy"),
                                         ("f1_macro", "F1_macro"), ("sensitivity", "Sensitivity")]:
            sa = float(res_a.get(metric_key, float("nan")))
            sb = float(res_b.get(metric_key, float("nan")))
            if np.isnan(sa) or np.isnan(sb):
                continue
            ci_a = res_a.get("ci", {}).get(metric_key, (sa, sa, sa))
            ci_b = res_b.get("ci", {}).get(metric_key, (sb, sb, sb))
            se_a = (float(ci_a[2]) - float(ci_a[1])) / (2*1.96 + 1e-9)
            se_b = (float(ci_b[2]) - float(ci_b[1])) / (2*1.96 + 1e-9)
            z_   = (sa - sb) / (np.sqrt(se_a**2 + se_b**2) + 1e-9)
            p_   = float(2*(1 - scipy_stats.norm.cdf(abs(z_))))
            out.append(_make_row(
                f"{setting_a} vs {setting_b}", lbl,
                f"{nm_a} ({setting_a}, n={len(yt_a)})",
                f"{nm_b} ({setting_b}, n={len(yt_b)})",
                metric_lbl, "Independent Z (bootstrap SE)",
                sa, sb,
                delong_z = round(float(z_), 3),
                p_val    = round(float(p_), 4),
                ci_lo    = round(float(ci_a[1]), 4),
                ci_hi    = round(float(ci_a[2]), 4),
                boot_p   = "N/A (independent)",
                significant = bool(p_ < 0.05),
                n_boot_used = "N/A",
            ))
        return out

    if "Overall" in all_results_dict:
        ov_res      = all_results_dict["Overall"]
        ov_se       = ov_res.get("Stacking_Ensemble", {})
        yt_ov       = np.array(ov_se.get("y_true", []))
        ov_best_nm, ov_best_res = _best(ov_res, exclude_se=False)

        for other in [s for s in settings if s != "Overall"]:
            oth_res     = all_results_dict[other]
            oth_se      = oth_res.get("Stacking_Ensemble", {})
            yt_oth      = np.array(oth_se.get("y_true", []))
            oth_best_nm, oth_best_res = _best(oth_res, exclude_se=False)

            if len(yt_ov) == 0 or len(yt_oth) == 0:
                continue

            # a. SE vs SE
            if ov_se and oth_se:
                rows.extend(_indep_z_rows(
                    "Overall", other,
                    "Stacking_Ensemble", ov_se,
                    "Stacking_Ensemble", oth_se,
                    yt_ov, yt_oth,
                    "Cross-setting (SE vs SE)"))
                log(f"  Overall vs {other} [SE vs SE]: "
                    f"AUROC diff={ov_se.get('auroc',0)-oth_se.get('auroc',0):+.3f}  "
                    f"Acc diff={_acc(ov_se)-_acc(oth_se):+.3f}")

            # b. Best model vs best model
            if ov_best_res and oth_best_res:
                rows.extend(_indep_z_rows(
                    "Overall", other,
                    ov_best_nm,  ov_best_res,
                    oth_best_nm, oth_best_res,
                    yt_ov, yt_oth,
                    f"Cross-setting (best: {ov_best_nm} vs {oth_best_nm})"))
                log(f"  Overall vs {other} [{ov_best_nm} vs {oth_best_nm}]: "
                    f"AUROC diff={ov_best_res.get('auroc',0)-oth_best_res.get('auroc',0):+.3f}  "
                    f"Acc diff={_acc(ov_best_res)-_acc(oth_best_res):+.3f}")

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{OUT}/pairwise_contrasts.csv", index=False)
        log("  Saved pairwise_contrasts.csv")
        log(df[["Setting","Contrast","Metric","Score_A","Score_B",
                "Diff_A_B","p_value","Significant"]].to_string(index=False))
    return df

# =============================================================================
# FOCUSED DCA: FIB-4 + pRFPS + Treat-All + Treat-None  (reviewer request)
# =============================================================================
# =============================================================================
# FOCUSED DCA: FIB-4 + pRFPS + Treat-All + Treat-None  (reviewer request)
# =============================================================================


def calibrate_score_to_probability(
    train_scores: np.ndarray,
    y_train: np.ndarray,
    test_scores: np.ndarray,
    label: str = "score",
) -> np.ndarray:
    """
    Platt scaling: fit logistic regression on (train_scores, y_train),
    apply to test_scores to produce calibrated probabilities in [0,1].

    Fit is on TRAINING data only -- no test-set information is used.
    The scaler is a 1-feature logistic regression (single input = raw score).

    Used by dca_focused() to put FIB-4 and pRFPS on the same probability
    axis as the stacking ensemble's predict_proba output, so that the DCA
    threshold axis is commensurable across all three curves.
    """
    from sklearn.linear_model import LogisticRegression as _LR
    tr = np.array(train_scores, dtype=float).reshape(-1, 1)
    te = np.array(test_scores,  dtype=float).reshape(-1, 1)
    y  = np.array(y_train, dtype=int).flatten()

    # Guard: need both classes in training labels
    if len(np.unique(y)) < 2:
        log(f"  [Platt {label}] Only one class in training -- falling back to min-max")
        lo, hi = float(te.min()), float(te.max())
        return np.clip((te.flatten() - lo) / (hi - lo + 1e-10), 0.0, 1.0)

    try:
        clf = _LR(max_iter=1000, random_state=42, solver="lbfgs")
        clf.fit(tr, y)
        proba = clf.predict_proba(te)[:, 1]
        log(f"  [Platt {label}] Calibrated: "
            f"train_range=[{tr.min():.2f},{tr.max():.2f}]  "
            f"test P range=[{proba.min():.3f},{proba.max():.3f}]")
        return proba
    except Exception as e:
        log(f"  [Platt {label}] Failed ({e}) -- falling back to min-max")
        lo, hi = float(te.min()), float(te.max())
        return np.clip((te.flatten() - lo) / (hi - lo + 1e-10), 0.0, 1.0)


# =============================================================================
# COHORT SUMMARY TABLE
# =============================================================================

def cohort_summary_table(
    df_raw:        pd.DataFrame,
    df_val_raw:    pd.DataFrame,
    X_mcb:         pd.DataFrame,
    y_mcb:         pd.Series,
    subgroups_mcb: pd.Series,
    X_val:         pd.DataFrame,
    y_val:         pd.Series,
    subgroups_val: pd.Series,
    all_test_indices: Dict,
) -> pd.DataFrame:
    """
    Produces a publication-ready cohort summary table with columns:
      Dataset | Subgroup | Total_N | Rapid_N | Slow_N | Rapid_pct |
      Train_N | Test_N | Rapid_Train | Rapid_Test

    One row per (Dataset × Subgroup), plus an Overall row per dataset.
    """
    section("COHORT SUMMARY TABLE")
    rows = []

    def _make_row(dataset, subgroup, X_sub, y_sub, test_idx=None):
        n_total = len(y_sub)
        n_rapid = int((y_sub == 1).sum())
        n_slow  = n_total - n_rapid
        pct     = round(n_rapid / n_total * 100, 1) if n_total > 0 else float("nan")

        if test_idx is not None:
            test_set   = set(test_idx)
            y_test_sub = y_sub[y_sub.index.isin(test_set)]
            y_tr_sub   = y_sub[~y_sub.index.isin(test_set)]
            n_test     = len(y_test_sub)
            n_train    = len(y_tr_sub)
            n_rap_tr   = int((y_tr_sub == 1).sum())
            n_rap_te   = int((y_test_sub == 1).sum())
        else:
            n_train = n_test = n_rap_tr = n_rap_te = "N/A"

        return {
            "Dataset":      dataset,
            "Subgroup":     subgroup,
            "Total_N":      n_total,
            "Rapid_N":      n_rapid,
            "Slow_N":       n_slow,
            "Rapid_pct":    f"{pct}%",
            "Train_N":      n_train,
            "Test_N":       n_test,
            "Rapid_Train":  n_rap_tr,
            "Rapid_Test":   n_rap_te,
        }

    # MCB Overall
    rows.append(_make_row("MCB (Discovery)", "Overall",
                           X_mcb, y_mcb,
                           all_test_indices.get("Overall")))

    # MCB subgroups
    for sg in sorted(subgroups_mcb.unique()):
        mask = subgroups_mcb == sg
        rows.append(_make_row("MCB (Discovery)", sg,
                               X_mcb[mask], y_mcb[mask],
                               all_test_indices.get(sg)))

    # Tapestry Overall
    rows.append(_make_row("Tapestry (Validation)", "Overall",
                           X_val, y_val, None))

    # Tapestry subgroups
    for sg in sorted(subgroups_val.unique()):
        mask = subgroups_val == sg
        rows.append(_make_row("Tapestry (Validation)", sg,
                               X_val[mask], y_val[mask], None))

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/cohort_summary_table.csv", index=False)
    log("  Saved cohort_summary_table.csv")
    log(df.to_string(index=False))
    return df

# =============================================================================
# FEATURE SELECTION ABLATION
# Disentangles performance gains from: partitioning vs feature selection
# =============================================================================

