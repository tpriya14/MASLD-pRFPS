# masld_phenotype_ml — MASLD Rapid Fibrosis Progression prediction pipeline
# Modules split from Rebuttal_experiments.py

from .utils import (
    log, section, OUT, results_dir,
    TARGET_COL, SUBGROUP_COL, POSITIVE_LABEL, NEGATIVE_LABEL,
    CLASS_NAMES, MODELS_CONFIG, FEATURE_LABELS,
    compute_fib4_score, fib4_binary_label, prepare_fib4_comparator,
    load_and_prepare_data, drop_nan_rows, two_way_split, fit_scale,
)
from .train import train_and_evaluate_setting
from .prfps import build_shap_clinical_risk_score
from .evaluate import compute_best_model_shap, save_patient_level_dataset
from .ablation import ablation_feature_selection
from .validate import validate_external
from .main import main
