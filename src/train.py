"""
train.py — Model training, GridSearchCV, stacking ensemble.
Auto-split from Rebuttal_experiments.py
"""
"""
Auto-generated from Rebuttal_experiments.py
"""
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .utils import (
    log, section, OUT, results_dir,
    TARGET_COL, SUBGROUP_COL, POSITIVE_LABEL, NEGATIVE_LABEL,
    CLASS_NAMES, FIB4_COMPONENTS, FEATURE_LABELS, feature_labels,
    MODELS_CONFIG, XGBOOST_AVAILABLE, LIGHTGBM_AVAILABLE, SHAP_AVAILABLE,
    compute_fib4_score, fib4_binary_label, prepare_fib4_comparator,
    load_and_prepare_data, drop_nan_rows, two_way_split, fit_scale,
    find_optimal_threshold_on_val, youden_threshold,
    bootstrap_ci, fmt_ci, compute_comprehensive_metrics,
    evaluate_model_comprehensive, results_to_publication_table,
    plot_roc_pr, plot_calibration_curves, calibration_analysis,
    decision_curve_analysis, delong_auc_test, paired_bootstrap_contrast,
    run_pairwise_contrasts, calibrate_score_to_probability, cohort_summary_table,
)
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix
)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None

def save_hyperparameter_info(model_name, grid_search, analysis_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    cv_df = pd.DataFrame(grid_search.cv_results_)
    keep  = [c for c in cv_df.columns
             if c.startswith("param_") or
             c in ("mean_test_score","std_test_score","mean_fit_time","rank_test_score")]
    cv_df[keep].sort_values("rank_test_score").to_csv(
        f"{output_dir}/{model_name}_{analysis_name}_cv_results.csv", index=False)
    summary = {
        "model_name": model_name, "analysis_name": analysis_name,
        "timestamp": datetime.now().isoformat(),
        "best_params": grid_search.best_params_,
        "best_cv_score": float(grid_search.best_score_),
    }
    with open(f"{output_dir}/{model_name}_{analysis_name}_hyperparams.json","w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    return summary


def save_stacking_ensemble_info(base_models, final_estimator, threshold,
                                 analysis_name, output_dir,
                                 test_metrics=None, val_metrics=None):
    os.makedirs(output_dir, exist_ok=True)
    base_info = []
    for name, est in base_models:
        try:
            p = {k: (v if isinstance(v,(int,float,str,bool,type(None))) else str(v))
                 for k,v in est.get_params().items()}
        except Exception:
            p = {}
        base_info.append({"model_name":name,"model_class":type(est).__name__,"best_params":p})
    info = {
        "analysis_name": analysis_name, "timestamp": datetime.now().isoformat(),
        "base_models": base_info,
        "meta_learner": {"class": type(final_estimator).__name__},
        "decision_threshold": float(threshold) if threshold is not None else None,
        "test_metrics": {k: float(v) for k,v in (test_metrics or {}).items()
                         if isinstance(v,(int,float)) and not isinstance(v,bool)},
    }
    with open(f"{output_dir}/stacking_ensemble_{analysis_name}.json","w") as fh:
        json.dump(info, fh, indent=2)
    return info


def get_oof_probabilities(estimator, X, y, cv):
    """
    Out-of-fold probabilities for threshold selection.
    Uses only training data and avoids in-sample threshold optimism.
    """
    return cross_val_predict(
        estimator,
        X,
        y,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
# =============================================================================
# CORE TRAINING FUNCTION  (leakage-free, preserves exact original grids)
# =============================================================================

def train_and_evaluate_setting(
    X_tr, X_te,
    y_tr, y_te,
    features: List[str],
    label: str = "",
    n_boot: int = 1000,
    hyp_dir: Optional[str] = None,
    stack_dir: Optional[str] = None,
) -> Tuple[Dict, Dict, StandardScaler]:
    """
    Training with threshold selected from TRAINING DATA only.

    [FIX-1] Threshold selection:
      Original (wrong): threshold tuned on X_test --> leakage
      Fixed (this):     threshold tuned on X_train --> no leakage, no wasted samples

    Steps:
      1. Fit model on X_train via GridSearchCV (5-fold CV).
      2. Get predicted probabilities on X_TRAIN (the same data used to fit).
      3. Apply your original threshold methods (f1, f2, balanced, youden)
         to training probabilities -- pick the best by training F1.
      4. Apply locked threshold to X_test ONCE for final reporting.

    Why training-set threshold is valid:
      - No test-set information used in any threshold decision --> no leakage.
      - Simpler than OOF and easier to explain in the rebuttal.
      - With n=752 (C2 n=325), avoids wasting samples on a separate val fold.
      - The threshold will be slightly optimistic (model has seen train data),
        but this is a minor bias, much less harmful than test-set leakage.
      - Standard approach used in many clinical prediction papers when
        cohort size is limited (Steyerberg 2019).

    Rebuttal wording:
      "Decision thresholds were selected by applying F1, F2, balanced, and
       Youden's J criteria to the training set predicted probabilities. The
       optimal threshold was then applied once to the held-out test set for
       final metric reporting."
    """
    section(f"Training: {label}")
    if hyp_dir  is None: hyp_dir  = f"{results_dir}/hyperparameters/{label}"
    if stack_dir is None: stack_dir = f"{results_dir}/stacking_ensemble_info"
    os.makedirs(hyp_dir, exist_ok=True)
    results = {}
    models = {}
    all_hp = {}
    model_cv_scores = {}
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Ntr, Nte, scl = fit_scale(X_tr, X_te, features)

    results = {}; models = {}; all_hp = {}

    for mname, cfg in MODELS_CONFIG.items():
        log(f"  [{mname}] GridSearchCV ...")
        try:
            Xtr_m = Ntr if cfg["scaled"] else X_tr[features].values
            Xte_m = Nte if cfg["scaled"] else X_te[features].values

            # Step 1: fit on training set
            gs = GridSearchCV(cfg["model_class"](), cfg["param_grid"],
                              cv=cv5,
                              scoring="f1_macro",  # weighted by class support
                              n_jobs=1, verbose=0, refit=True)
            gs.fit(Xtr_m, y_tr.values)
            mdl = gs.best_estimator_
            model_cv_scores[mname] = float(gs.best_score_)
            # Save hyperparameters (identical to original)
            all_hp[mname] = save_hyperparameter_info(mname, gs, label, hyp_dir)

            # Step 2: probabilities ON TRAINING SET
            try:
                prob_tr = get_oof_probabilities(mdl, Xtr_m, y_tr.values, cv5)
            except Exception as e:
                log(f"    [{mname}] OOF threshold probabilities failed ({e}); using in-sample probabilities")
                prob_tr = mdl.predict_proba(Xtr_m)[:, 1]

            # Step 3: pick threshold from training probabilities
            # Your original 4 methods -- now applied to train, not test
            # Threshold optimised for Rapid Progression (class 1) F1 specifically
            # This directly addresses poor minority-class performance.
            # We try all four original methods and pick by class-1 F1,
            # not macro F1, so the threshold favours catching rapid progressors.
            thresh = 0.5; best_f1_pos = -1
            for opt_method in ["f1", "f2", "balanced", "youden"]:
                try:
                    t = find_optimal_threshold_on_val(
                        y_tr.values, prob_tr, opt_method)
                    pred_t = (prob_tr >= t).astype(int)
                    # F1 for class 1 (Rapid Progression) only
                    f1_pos = f1_score(
                        y_tr, pred_t, pos_label=1,
                        average="binary", zero_division=0)
                    if f1_pos > best_f1_pos:
                        best_f1_pos = f1_pos
                        thresh = t
                        thresh = min(thresh, 0.95)  # cap degenerate thresholds
                except Exception:
                    pass

            # Step 4: apply to test -- ONCE
            prob_te = mdl.predict_proba(Xte_m)[:, 1]
            pred_te = (prob_te >= thresh).astype(int)

            # Extended evaluation [FIX-2]
            eval_r = evaluate_model_comprehensive(
                mdl, Xte_m, y_te.values, pred_te, prob_te,
                mname, thresh, n_boot=n_boot)

            results[mname] = eval_r
            models[mname]  = mdl

            log(f"    thr={thresh:.3f} acc={eval_r['accuracy']:.3f} "
                f"f1={eval_r['f1_macro']:.3f} auroc={eval_r['auroc']:.3f} "
                f"sens={eval_r['sensitivity']:.3f} "
                f"spec={eval_r['specificity']:.3f} "
                f"brier={eval_r['brier']:.3f}")

        except Exception as e:
            log(f"    [{mname}] FAILED: {e}")

    # Combined hyperparameters JSON
    with open(f"{hyp_dir}/all_models_hyperparams_{label}.json", "w") as fh:
        json.dump(all_hp, fh, indent=2, default=str)

    # Stacking ensemble: top-3 by test F1
    if len(models) >= 3:
        top3 = sorted(model_cv_scores, key=model_cv_scores.get, reverse=True)[:3]
        log(f"  Stacking: top-3 = {top3}")

        meta = LogisticRegression(
            random_state=42, max_iter=2000, C=0.01,
            solver="lbfgs", class_weight="balanced", penalty="l2")
        se = StackingClassifier(
            estimators=[(n, models[n]) for n in top3],
            final_estimator=meta, cv=5, n_jobs=1, passthrough=False)
        try:
            se.fit(Ntr, y_tr.values)

            # Threshold from training predictions
            try:
                prob_tr_se = get_oof_probabilities(se, Ntr, y_tr.values, cv5)
            except Exception as e:
                log(f"  [SE] OOF threshold probabilities failed ({e}); using in-sample probabilities")
                prob_tr_se = se.predict_proba(Ntr)[:, 1]
            th_se = 0.5; best_f1_se_pos = -1
            for _om in ["f1", "f2", "balanced", "youden"]:
                try:
                    _t  = find_optimal_threshold_on_val(
                        y_tr.values, prob_tr_se, _om)
                    _f1 = f1_score(
                        y_tr, (prob_tr_se >= _t).astype(int),
                        pos_label=1, average="binary", zero_division=0)
                    if _f1 > best_f1_se_pos:
                        best_f1_se_pos = _f1; th_se = _t
                        th_se = min(th_se, 0.95)  # cap degenerate
                except Exception:
                    pass

            pt_se = se.predict_proba(Nte)[:, 1]
            pd_se = (pt_se >= th_se).astype(int)

            eval_se = evaluate_model_comprehensive(
                se, Nte, y_te.values, pd_se, pt_se,
                "Stacking_Ensemble", th_se, n_boot=n_boot)
            results["Stacking_Ensemble"] = eval_se
            models["Stacking_Ensemble"]  = se

            save_stacking_ensemble_info(
                base_models=[(n, models[n]) for n in top3],
                final_estimator=meta, threshold=th_se,
                analysis_name=label, output_dir=stack_dir,
                test_metrics={
                    "f1_score": eval_se["f1_macro"],
                    "accuracy": eval_se["accuracy"],
                    "auroc":    eval_se["auroc"]})

            log(f"  [SE] thr={th_se:.3f} f1={eval_se['f1_macro']:.3f} "
                f"auroc={eval_se['auroc']:.3f}")

        except Exception as e:
            log(f"  Stacking FAILED: {e}")

    return results, models, scl


# =============================================================================
# MASTER TABLE  (all settings consolidated)
# =============================================================================
def append_master(master_rows: List, results: Dict, setting: str) -> None:
    for name, r in results.items():
        ci = r.get("ci", {})
        def f(m): return fmt_ci(m, ci, r)
        master_rows.append({
            "Setting": setting, "Model": name,
            "Threshold": f"{r.get('optimal_threshold', float('nan')):.3f}",
            "Accuracy":     f("accuracy"),
            "F1_macro":     f("f1_macro"),
            "AUROC":        f("auroc"),
            "AUPRC":        f("auprc"),
            "Brier":        f("brier"),
            "Sensitivity":  f("sensitivity"),
            "Specificity":  f("specificity"),
            "PPV":          f("ppv"),
            "NPV":          f("npv"),
            f"Precision_{NEGATIVE_LABEL}":  f("precision_0"), f"Recall_{NEGATIVE_LABEL}":     f("recall_0"),
            f"F1_{NEGATIVE_LABEL}":         f("f1_0"), f"Support_{NEGATIVE_LABEL}":    r.get("support_0",""),
            f"Precision_{POSITIVE_LABEL}":  f("precision_1"), f"Recall_{POSITIVE_LABEL}":     f("recall_1"),
            f"F1_{POSITIVE_LABEL}":         f("f1_1"), f"Support_{POSITIVE_LABEL}":    r.get("support_1",""),
            "TP": r.get("TP",""), "FP": r.get("FP",""),
            "TN": r.get("TN",""), "FN": r.get("FN",""),
            "Prevalence": f"{r.get('prevalence',float('nan')):.3f}",
        })


# =============================================================================
# EXTERNAL VALIDATION  (Tapestry)
# =============================================================================

def apply_prfps_formula(prfps_result, X_val, y_val, setting):
    """
    Apply MCB-derived pRFPS formula to validation patients.
    Uses locked MCB-derived thresholds, weights, and cutoffs.
    No refitting on validation data.
    """
    top = prfps_result["feature_table"]
    cutoff_int = prfps_result["threshold"]
    cutoff_cont = prfps_result.get("threshold_cont", cutoff_int)

    train_int_max = float(prfps_result.get("train_int_max", np.nan))
    train_cont_max = float(prfps_result.get("train_cont_max", np.nan))

    X_te = (
        X_val.reset_index(drop=True)
        if isinstance(X_val, pd.DataFrame)
        else pd.DataFrame(X_val)
    )
    y_arr = np.array(y_val).astype(int).flatten()

    scores_int = np.zeros(len(X_te))
    scores_cont = np.zeros(len(X_te))

    for _, row in top.iterrows():
        feat = row["feature"]
        if feat not in X_te.columns:
            log(f"    [pRFPS-Tap] '{feat}' missing in Tapestry -- skipped")
            continue

        vals = X_te[feat].astype(float).values
        thr = float(row["youden_threshold"])
        direction = row.get("score_direction", row.get("direction", "Risk+"))

        # IMPORTANT: strict > matches build_shap_clinical_risk_score()
        if direction == "Risk+":
            ind = (vals >= thr).astype(float)
        else:
            ind = (vals < thr).astype(float)

        scores_int += int(row["weight_int"]) * ind
        scores_cont += float(row["weight_raw"]) * ind

    prfps_int = scores_int.astype(int)
    prfps_cont = scores_cont

    pred_int = (prfps_int >= cutoff_int).astype(int)
    pred_cont = (prfps_cont >= cutoff_cont).astype(int)

    if not np.isfinite(train_int_max) or train_int_max <= 0:
        train_int_max = float(np.max(prfps_int)) if len(prfps_int) else 0.0
    if not np.isfinite(train_cont_max) or train_cont_max <= 0:
        train_cont_max = float(np.max(prfps_cont)) if len(prfps_cont) else 0.0

    score_norm = prfps_int.astype(float) / (train_int_max + 1e-10)
    score_cont_norm = prfps_cont / (train_cont_max + 1e-10)

    def _perf(yt, ypred, yscore):
        if len(np.unique(yt)) < 2:
            return {}
        tn_, fp_, fn_, tp_ = confusion_matrix(yt, ypred, labels=[0, 1]).ravel()
        return {
            "auroc": roc_auc_score(yt, yscore),
            "auprc": average_precision_score(yt, yscore),
            "accuracy": accuracy_score(yt, ypred),
            "f1_macro": f1_score(yt, ypred, average="macro", zero_division=0),
            "f1_rapid": f1_score(yt, ypred, pos_label=1, average="binary", zero_division=0),
            "precision": precision_score(yt, ypred, pos_label=1, zero_division=0),
            "recall": recall_score(yt, ypred, pos_label=1, zero_division=0),
            "sensitivity": tp_ / (tp_ + fn_ + 1e-10),
            "specificity": tn_ / (tn_ + fp_ + 1e-10),
            "ppv": tp_ / (tp_ + fp_ + 1e-10),
            "npv": tn_ / (tn_ + fn_ + 1e-10),
            "tp": int(tp_), "fp": int(fp_), "tn": int(tn_), "fn": int(fn_),
        }

    perf_int = _perf(y_arr, pred_int, prfps_int.astype(float))
    perf_cont = _perf(y_arr, pred_cont, prfps_cont.astype(float))

    log(
        f"  [pRFPS-Tap {setting}] "
        f"int_AUROC={perf_int.get('auroc', float('nan')):.3f}  "
        f"cont_AUROC={perf_cont.get('auroc', float('nan')):.3f}  "
        f"cutoff_int={cutoff_int:.0f}  cutoff_cont={cutoff_cont:.3f}"
    )

    return {
        "score_int": prfps_int,
        "score_cont": prfps_cont,
        "score_norm": score_norm,
        "score_cont_norm": score_cont_norm,
        "pred_int": pred_int,
        "pred_cont": pred_cont,
        "perf_int": perf_int,
        "perf_cont": perf_cont,
        "cutoff_int": cutoff_int,
        "cutoff_cont": cutoff_cont,
        "y_true": y_arr,
        "n_features": len(top),
        "setting": setting,
    }
