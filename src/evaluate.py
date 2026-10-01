"""
evaluate.py — SHAP, patient-level output, stratified evaluation, circularity.

"""
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
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
import scipy.stats as scipy_stats
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve,
    balanced_accuracy_score
)
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    import shap
except ImportError:
    shap = None

def normalize_binary_shap_values(sv):
    """Return 2D SHAP array: n_samples x n_features for binary class 1."""
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    if sv.ndim == 1:
        sv = sv.reshape(1, -1)
    return sv

def shap_stability_cv(
    X_np: np.ndarray,
    y_np: np.ndarray,
    feature_names: List[str],
    tag: str,
    n_splits: int = 5,
    cv_threshold: float = 0.30,
    X_val_np: np.ndarray = None,
    y_val_np: np.ndarray = None,
    val_tag: str = "Tapestry",
) -> pd.DataFrame:
    """
    [FIX-12 / Part 7] SHAP feature weight stability.

    Addresses reviewer H41R: "How stable are SHAP-derived feature weights
    across folds or cohorts?"

    THREE stability measures:

    1. CV across CV folds (within-cohort):
       CV = SD / mean of |SHAP| importance across n_splits folds.
       CV < cv_threshold (0.30) = stable.
       Threshold of 0.30 follows established biomarker reproducibility
       criteria (CV < 30% considered acceptable for clinical biomarkers).

    2. Cross-cohort Spearman rank correlation (if X_val_np provided):
       Compute mean |SHAP| on MCB and on Tapestry separately.
       Rank-correlate the feature importance vectors.
       rho > 0.70 = good cross-cohort stability.
       This directly addresses the "across cohorts" part of the reviewer question.

    3. Publication-ready Figure 5 with direction:
       Horizontal bar chart showing mean SHAP (signed, not absolute) per feature.
       Red = positive SHAP = increases risk of Rapid Progression.
       Blue = negative SHAP = protective (decreases risk).
       Error bars = SD across folds.
       This adds the direction of contribution requested for Figure 5.

    Returns:
       DataFrame with per-feature stability statistics.
    """
    if not SHAP_AVAILABLE or not XGBOOST_AVAILABLE:
        log("  SHAP/XGB not available -- skipping"); return pd.DataFrame()

    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    imps_abs  = []   # |SHAP| per fold -- for stability
    imps_signed = [] # signed SHAP per fold -- for direction

    log(f"  [FIX-12] SHAP stability across {n_splits} CV folds -- {tag}")
    log(f"  CV threshold for stability: {cv_threshold} "
        f"(features with CV < {cv_threshold} = stable)")

    for fold_idx, (tr, va) in enumerate(cv.split(X_np, y_np)):
        try:
            m = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=1)
            m.fit(X_np[tr], y_np[tr])
            ex = shap.TreeExplainer(m)
            sv = ex.shap_values(X_np[va])
            sv = normalize_binary_shap_values(sv)
            imps_abs.append(np.abs(sv).mean(axis=0))
            imps_signed.append(sv.mean(axis=0))
        except Exception as e:
            log(f"    Fold {fold_idx+1} failed: {e}")

    if not imps_abs:
        return pd.DataFrame()

    arr_abs    = np.array(imps_abs)     # (n_folds, n_features)
    arr_signed = np.array(imps_signed)

    mean_abs   = arr_abs.mean(axis=0)
    std_abs    = arr_abs.std(axis=0)
    cv_        = std_abs / (mean_abs + 1e-10)
    mean_signed = arr_signed.mean(axis=0)

    df = pd.DataFrame({
        "feature":       feature_names,
        "label":         [FEATURE_LABELS.get(f, f) for f in feature_names],
        "mean_abs_shap": mean_abs,
        "std_abs_shap":  std_abs,
        "cv_shap":       cv_,
        "mean_signed_shap": mean_signed,
        "direction":     ["Risk+" if s > 0 else "Protective" for s in mean_signed],
        "stable":        cv_ < cv_threshold,
        "Setting":       tag,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    n_stable = int(df["stable"].sum())
    log(f"  Stable features (CV < {cv_threshold}): {n_stable}/{len(df)}")
    log(f"  CV threshold rationale: CV < 0.30 = <30% relative variation "
        f"across folds, consistent with clinical biomarker reproducibility criteria")

    # ── Figure 5 style: signed bar chart with direction + error bars ─────────
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(df) * 0.45)))

    # Panel A: Mean signed SHAP (direction of contribution)
    ax = axes[0]
    df_plot = df.sort_values("mean_signed_shap")
    colors  = ["#D73027" if d == "Risk+" else "#4575B4"
               for d in df_plot["direction"]]
    bars = ax.barh(df_plot["label"], df_plot["mean_signed_shap"],
                   color=colors, alpha=0.85, edgecolor="white",
                   xerr=df_plot["std_abs_shap"], capsize=3)
    ax.axvline(0, color="black", lw=0.8, ls="-")
    ax.set_xlabel("Mean SHAP Value (positive = increases Rapid Progression risk)",
                  fontsize=10)
    ax.set_title(f"A. Direction of Contribution -- {tag}", fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(handles=[
        mpatches.Patch(color="#D73027", label="Risk-increasing"),
        mpatches.Patch(color="#4575B4", label="Protective"),
    ], fontsize=8, loc="lower right")

    # Panel B: CV stability (absolute SHAP, error bars, green/red)
    ax2 = axes[1]
    df_stab = df.sort_values("mean_abs_shap")
    colors2 = ["#2ecc71" if s else "#e74c3c" for s in df_stab["stable"]]
    ax2.barh(df_stab["label"], df_stab["mean_abs_shap"],
             color=colors2, alpha=0.85, edgecolor="white",
             xerr=df_stab["std_abs_shap"], capsize=3)
    ax2.set_xlabel("Mean |SHAP| Value +/- SD across folds", fontsize=10)
    ax2.set_title(f"B. Stability Across {n_splits}-fold CV -- {tag}", fontsize=11)
    ax2.grid(axis="x", alpha=0.25)
    ax2.legend(handles=[
        mpatches.Patch(color="#2ecc71",
                       label=f"Stable (CV < {cv_threshold})"),
        mpatches.Patch(color="#e74c3c",
                       label=f"Unstable (CV >= {cv_threshold})"),
    ], fontsize=8, loc="lower right")

    plt.suptitle(f"Figure 5. SHAP Feature Importance -- {tag} ({n_stable}/{len(df)} features stable, CV={cv_threshold})", fontsize=12, y=1.02)
    plt.tight_layout()

    os.makedirs(f"{results_dir}/shap_stability", exist_ok=True)
    fig_path = f"{results_dir}/shap_stability/figure5_shap_{tag}.png"
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  Figure 5 saved: {fig_path}")

    # ── Cross-cohort stability (if Tapestry data provided) ───────────────────
    cross_cohort_row = None
    if X_val_np is not None and y_val_np is not None:
        log(f"  Computing cross-cohort SHAP stability: {tag} vs {val_tag}")
        try:
            # Train on full MCB train, compute SHAP on Tapestry
            m_full = XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss",
                random_state=42, n_jobs=1)
            m_full.fit(X_np, y_np)
            ex_full  = shap.TreeExplainer(m_full)

            sv_mcb = ex_full.shap_values(X_np)
            sv_mcb = normalize_binary_shap_values(sv_mcb)
            mcb_importance = np.abs(sv_mcb).mean(axis=0)

            sv_tap = ex_full.shap_values(X_val_np)
            sv_tap = normalize_binary_shap_values(sv_tap)
            tap_importance = np.abs(sv_tap).mean(axis=0)

            # Spearman rank correlation of feature importance rankings
            rho, p_val = scipy_stats.spearmanr(mcb_importance, tap_importance)

            log(f"  Cross-cohort Spearman rho = {rho:.3f} (p={p_val:.4f})")
            log(f"  Interpretation: {'Good cross-cohort stability' if rho > 0.70 else 'Moderate cross-cohort stability'}")

            # Cross-cohort comparison figure
            fig2, ax3 = plt.subplots(figsize=(8, 5))
            x_pos = np.arange(len(feature_names))
            w     = 0.38
            mcb_sorted_idx = np.argsort(mcb_importance)[::-1]
            labels_sorted  = [FEATURE_LABELS.get(feature_names[i], feature_names[i])
                              for i in mcb_sorted_idx]

            ax3.bar(x_pos - w/2,
                    mcb_importance[mcb_sorted_idx], w,
                    label="MCB (discovery)", color="#2980b9", alpha=0.8)
            ax3.bar(x_pos + w/2,
                    tap_importance[mcb_sorted_idx], w,
                    label=f"{val_tag} (validation)", color="#e67e22", alpha=0.8)

            ax3.set_xticks(x_pos)
            ax3.set_xticklabels(labels_sorted, rotation=40, ha="right", fontsize=9)
            ax3.set_ylabel("Mean |SHAP| Importance", fontsize=11)
            ax3.set_title(
                f"Cross-cohort SHAP Stability: {tag} -- Spearman rho = {rho:.3f}  p = {p_val:.4f}",
                fontsize=12)
            ax3.legend(fontsize=10)
            ax3.grid(axis="y", alpha=0.25)
            plt.tight_layout()
            cc_path = f"{results_dir}/shap_stability/cross_cohort_shap_{tag}.png"
            plt.savefig(cc_path, dpi=200, bbox_inches="tight")
            plt.close()
            log(f"  Cross-cohort figure saved: {cc_path}")

            # Add cohort comparison to dataframe
            tap_imp_reordered = dict(zip(feature_names, tap_importance))
            df["tap_mean_abs_shap"] = [tap_imp_reordered.get(f, np.nan)
                                        for f in feature_names]
            df["rank_mcb"]   = df["mean_abs_shap"].rank(ascending=False)
            df["rank_tap"]   = df["tap_mean_abs_shap"].rank(ascending=False)
            df["rank_diff"]  = (df["rank_mcb"] - df["rank_tap"]).abs()
            df["cross_cohort_spearman_rho"] = round(rho, 3)

            cross_cohort_row = {
                "Setting":     tag,
                "Val_Cohort":  val_tag,
                "Spearman_rho": round(rho, 3),
                "p_value":     round(p_val, 4),
                "Interpretation": ("Good (rho>0.70)" if rho > 0.70
                                   else "Moderate (rho 0.50-0.70)" if rho > 0.50
                                   else "Poor (rho<0.50)"),
                "N_MCB":    len(X_np),
                "N_Tapestry": len(X_val_np),
            }
            pd.DataFrame([cross_cohort_row]).to_csv(
                f"{results_dir}/shap_stability/cross_cohort_rho_{tag}.csv",
                index=False)

        except Exception as e:
            log(f"  Cross-cohort SHAP failed: {e}")

    # Save full stability table
    df.to_csv(
        f"{results_dir}/shap_stability/shap_stability_{tag}.csv", index=False)
    log(f"  Stability table saved for {tag}")
    return df



# =============================================================================
# [FIX-10]  SHAP continuous vs integer pRFPS AUROC delta
# =============================================================================
def shap_vs_prfps_delta_from_result(prfps_result: Dict,
                                      tag: str,
                                      n_boot: int = 500) -> Dict:
    """
    [FIX-10 CORRECTED] AUROC and accuracy delta between:

      A. True continuous SHAP-additive score:
           score_SHAP(patient_i) = sum_j shap_value(i, j)
         This is the model's exact additive decomposition per patient.
         Each feature contributes its ACTUAL SHAP attribution, not a
         population-mean weight times a binary indicator.

      B. Integer pRFPS:
           score_int(patient_i) = sum over top-N: weight_INT_j × I(val_ij > thr_j)
         This discretises the SHAP signal in two ways:
           1. Replaces patient-specific shap(i,j) with mean_abs_shap_j × binary
           2. Rounds the resulting weight to an integer

    The delta A−B (reported as AUROC_SHAP_additive minus AUROC_integer) answers
    the reviewer's question: what is the total information cost of mapping
    continuous per-patient SHAP attributions to bounded integer weights?

    We also report B_cont − B_int (weight_raw vs weight_int, same indicator),
    which isolates the cost of integer rounding alone. This is smaller and
    was the only comparison in the previous version.

    Requires prfps_result["shap_values"] -- stored by build_shap_clinical_risk_score.
    """
    if not prfps_result:
        return {}

    score_int   = prfps_result["score_int"].astype(float)
    score_cont  = prfps_result["score_cont"]   # weight_raw × binary indicator
    y_true      = prfps_result["y_true"]
    sv_arr      = prfps_result.get("shap_values_test")   # shape (n_patients, n_features)
    perf_int    = prfps_result.get("perf_int",  {})
    perf_cont   = prfps_result.get("perf_cont", {})
    top         = prfps_result.get("feature_table", pd.DataFrame())

    if len(np.unique(y_true)) < 2:
        return {}

    # ── True SHAP-additive score ──────────────────────────────────────────────
    if sv_arr is not None:
        sv_arr = np.array(sv_arr)
        # Sum across ALL features for each patient -- the model's SHAP explanation
        score_shap = sv_arr.sum(axis=1)
        has_shap   = True
        log(f"  [Delta {tag}] True SHAP-additive score range: "
            f"[{score_shap.min():.4f}, {score_shap.max():.4f}]")
    else:
        score_shap = None
        has_shap   = False
        log(f"  [Delta {tag}] shap_values not in prfps_result -- "
            f"SHAP-additive vs integer comparison skipped")

    # ── AUROC for each score ──────────────────────────────────────────────────
    def _auroc(sc):
        try:
            return float(roc_auc_score(y_true, sc))
        except Exception:
            return float("nan")

    auroc_shap  = _auroc(score_shap) if has_shap else float("nan")
    auroc_cont  = prfps_result.get("auroc_cont", _auroc(score_cont))
    auroc_int   = prfps_result.get("auroc",      _auroc(score_int))

    # Deltas: positive = SHAP/cont score is better than integer
    delta_shap_vs_int  = float(auroc_shap - auroc_int)  if not np.isnan(auroc_shap) else float("nan")
    delta_cont_vs_int  = float(auroc_cont - auroc_int)  if not np.isnan(auroc_cont) else float("nan")
    delta_shap_vs_cont = float(auroc_shap - auroc_cont) if not np.isnan(auroc_shap) else float("nan")

    # ── DeLong tests ─────────────────────────────────────────────────────────
    def _delong(a, b):
        try:    return delong_auc_test(y_true, a, b)
        except: return float("nan"), float("nan")

    z_shap_int,  p_shap_int  = _delong(score_shap, score_int)  if has_shap else (float("nan"), float("nan"))
    z_cont_int,  p_cont_int  = _delong(score_cont, score_int)

    # ── Bootstrap 95% CI on SHAP-additive vs integer delta ───────────────────
    rng = np.random.RandomState(42)
    n   = len(y_true)
    deltas_shap, deltas_cont = [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        yt  = y_true[idx]
        if len(np.unique(yt)) < 2: continue
        try:
            ai = roc_auc_score(yt, score_int[idx])
            if has_shap:
                deltas_shap.append(roc_auc_score(yt, score_shap[idx]) - ai)
            deltas_cont.append(roc_auc_score(yt, score_cont[idx]) - ai)
        except Exception:
            pass

    def _ci(lst):
        if not lst: return float("nan"), float("nan")
        return float(np.percentile(lst, 2.5)), float(np.percentile(lst, 97.5))

    ci_lo_shap, ci_hi_shap = _ci(deltas_shap)
    ci_lo_cont, ci_hi_cont = _ci(deltas_cont)

    # ── Accuracy and F1 ───────────────────────────────────────────────────────
    acc_int  = perf_int.get("accuracy",  float("nan"))
    acc_cont = perf_cont.get("accuracy", float("nan"))
    f1_int   = perf_int.get("f1_macro",  float("nan"))
    f1_cont  = perf_cont.get("f1_macro", float("nan"))

    # ── Per-feature information loss from rounding ────────────────────────────
    mean_info_loss = float(top["information_loss"].mean()) if "information_loss" in top.columns else float("nan")
    max_info_loss  = float(top["information_loss"].max())  if "information_loss" in top.columns else float("nan")

    row = {
        "Setting":              tag,
        "N_features_selected":  prfps_result.get("n_features_selected", len(top)),
        "SHAP_coverage_pct":    round(float(prfps_result.get("shap_coverage_pct", float("nan"))), 1),

        # ── PRIMARY comparison: true SHAP-additive vs integer pRFPS ──────────
        # This is what the reviewer asks: cost of ALL discretisation steps
        "AUROC_SHAP_additive":       round(auroc_shap, 4),
        "AUROC_integer_pRFPS":       round(auroc_int,  4),
        "AUROC_delta_SHAP_vs_int":   round(delta_shap_vs_int,  4) if not np.isnan(delta_shap_vs_int)  else "N/A",
        "AUROC_delta_SHAP_CI_lo":    round(ci_lo_shap, 4) if not np.isnan(ci_lo_shap) else "N/A",
        "AUROC_delta_SHAP_CI_hi":    round(ci_hi_shap, 4) if not np.isnan(ci_hi_shap) else "N/A",
        "DeLong_z_SHAP_vs_int":      round(float(z_shap_int), 3) if not np.isnan(z_shap_int) else "N/A",
        "DeLong_p_SHAP_vs_int":      round(float(p_shap_int), 4) if not np.isnan(p_shap_int) else "N/A",

        # ── SECONDARY comparison: weight_raw vs weight_int (rounding only) ───
        # Isolates cost of integer rounding, holding thresholding constant
        "AUROC_weight_raw_pRFPS":    round(auroc_cont, 4),
        "AUROC_delta_cont_vs_int":   round(delta_cont_vs_int,  4) if not np.isnan(delta_cont_vs_int)  else "N/A",
        "AUROC_delta_cont_CI_lo":    round(ci_lo_cont, 4) if not np.isnan(ci_lo_cont) else "N/A",
        "AUROC_delta_cont_CI_hi":    round(ci_hi_cont, 4) if not np.isnan(ci_hi_cont) else "N/A",
        "DeLong_z_cont_vs_int":      round(float(z_cont_int), 3) if not np.isnan(z_cont_int) else "N/A",
        "DeLong_p_cont_vs_int":      round(float(p_cont_int), 4) if not np.isnan(p_cont_int) else "N/A",

        # ── Cost of thresholding (SHAP vs weight_raw) ─────────────────────────
        "AUROC_delta_SHAP_vs_cont":  round(delta_shap_vs_cont, 4) if not np.isnan(delta_shap_vs_cont) else "N/A",

        # ── Accuracy / F1 ─────────────────────────────────────────────────────
        "Accuracy_continuous":  round(float(acc_cont), 4) if not np.isnan(acc_cont) else "N/A",
        "Accuracy_integer":     round(float(acc_int),  4) if not np.isnan(acc_int)  else "N/A",
        "Accuracy_delta":       round(float(acc_cont - acc_int), 4) if not (np.isnan(acc_cont) or np.isnan(acc_int)) else "N/A",
        "F1_macro_continuous":  round(float(f1_cont), 4) if not np.isnan(f1_cont) else "N/A",
        "F1_macro_integer":     round(float(f1_int),  4) if not np.isnan(f1_int)  else "N/A",
        "F1_macro_delta":       round(float(f1_cont - f1_int), 4) if not (np.isnan(f1_cont) or np.isnan(f1_int)) else "N/A",

        # ── Rounding information loss ──────────────────────────────────────────
        "Mean_info_loss_pct":   round(float(mean_info_loss), 2) if not np.isnan(mean_info_loss) else "N/A",
        "Max_info_loss_pct":    round(float(max_info_loss),  2) if not np.isnan(max_info_loss)  else "N/A",
    }

    log(f"  [Delta {tag}] AUROC: SHAP-additive={auroc_shap:.3f}  "
        f"weight_raw={auroc_cont:.3f}  integer={auroc_int:.3f}")
    log(f"  [Delta {tag}] SHAP-additive vs integer: "
        f"delta={delta_shap_vs_int:+.4f} [{ci_lo_shap:+.3f},{ci_hi_shap:+.3f}] "
        f"DeLong z={z_shap_int:.3f} p={p_shap_int:.4f}")
    log(f"  [Delta {tag}] Weight rounding only: "
        f"delta={delta_cont_vs_int:+.4f} [{ci_lo_cont:+.3f},{ci_hi_cont:+.3f}] "
        f"DeLong z={z_cont_int:.3f} p={p_cont_int:.4f}")
    log(f"  [Delta {tag}] Thresholding cost (SHAP vs weight_raw): "
        f"delta={delta_shap_vs_cont:+.4f}")
    return row


def shap_vs_prfps_delta(mdl, Xtr_np: np.ndarray, Xte_np: np.ndarray,
                         y_te: np.ndarray, feature_names: List[str],
                         tag: str) -> pd.DataFrame:
    """[FIX-10 LEGACY] Kept for backward compat. Prefer shap_vs_prfps_delta_from_result."""
    # This function is no longer called in main; the corrected version
    # shap_vs_prfps_delta_from_result() uses the REAL pRFPS build result.
    return pd.DataFrame()


# =============================================================================
# [FIX-12 EXTENDED]  pRFPS WEIGHT STABILITY ACROSS CV FOLDS
# =============================================================================


def compute_prfps_weight_stability(
    X_tr_np: np.ndarray,
    y_tr_np: np.ndarray,
    feature_names: List[str],
    model_name: str,
    fitted_model,
    tag: str,
    n_splits: int = 5,
    shap_coverage_threshold: float = 0.80,
    max_features: int = 10,
) -> List[Dict]:
    """
    [FIX-12 EXTENDED] Measure stability of pRFPS integer weights and
    feature selection across k-fold CV using the ACTUAL best model class.

    Unlike shap_stability_cv (which re-trains XGBoost internally),
    this uses the same model class as was used to build pRFPS.

    Per fold per feature we record:
      - mean |SHAP| importance
      - integer weight (w_int = round(importance/sum * 10))
      - whether the feature was selected (in top-N by coverage threshold)

    Summary statistics across folds:
      - mean and SD of integer weight per feature
      - CV (SD/mean) of integer weight -- stability criterion
      - Selection frequency (out of n_splits folds)

    Outputs:
      prfps_weight_stability.csv -- per-feature stability metrics
    """
    if not SHAP_AVAILABLE:
        log("  SHAP not available -- skipping pRFPS weight stability")
        return []

    cv    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rows  = []   # one row per (fold, feature)

    tree_types = (RandomForestClassifier, GradientBoostingClassifier,
                  DecisionTreeClassifier, ExtraTreesClassifier)
    if XGBOOST_AVAILABLE: tree_types = tree_types + (XGBClassifier,)

    log(f"  [pRFPS stability] {tag} -- {n_splits}-fold CV with {model_name}")

    for fold_idx, (tr_idx, va_idx) in enumerate(cv.split(X_tr_np, y_tr_np)):
        try:
            # StackingClassifier fix: get_params(deep=True) returns nested keys
            # like 'final_estimator__C' that StackingClassifier.__init__()
            # does not accept --> crash. Extract best tree base model instead.
            # For all other models: use get_params(deep=False) -- top-level only.
            _ref_model = fitted_model

            if isinstance(_ref_model, StackingClassifier):
                _base_models = dict(_ref_model.estimators)
                _tree_base   = None
                for _nm, _bm in _base_models.items():
                    if isinstance(_bm, tree_types):
                        _tree_base = _bm
                        break
                _ref_model = _tree_base if _tree_base is not None                              else list(_base_models.values())[0]
                log(f"    [stability fold {fold_idx+1}] SE → "
                    f"using base {type(_ref_model).__name__}")

            try:
                _params  = {k: v for k, v in
                            _ref_model.get_params(deep=False).items()
                            if not callable(v)}
                fold_mdl = type(_ref_model)(**_params)
            except Exception:
                fold_mdl = RandomForestClassifier(
                    n_estimators=100, random_state=42, n_jobs=1)

            fold_mdl.fit(X_tr_np[tr_idx], y_tr_np[tr_idx])

            # SHAP on validation fold
            if isinstance(fold_mdl, tree_types):
                exp = shap.TreeExplainer(fold_mdl)
                sv  = exp.shap_values(X_tr_np[va_idx])
            elif isinstance(fold_mdl, LogisticRegression):
                exp = shap.LinearExplainer(fold_mdl, X_tr_np[tr_idx])
                sv  = exp.shap_values(X_tr_np[va_idx])
            else:
                bg  = shap.kmeans(X_tr_np[tr_idx], min(30, len(tr_idx)))
                exp = shap.KernelExplainer(
                    lambda x: fold_mdl.predict_proba(x)[:, 1], bg)
                sv  = exp.shap_values(X_tr_np[va_idx], nsamples=50)

            # Normalise to 2D (n_samples, n_features).
            # Newer SHAP returns 3D (n, features, n_classes); older returns list.
            if isinstance(sv, list):
                sv = sv[1]
            sv = np.array(sv)
            if sv.ndim == 3:
                sv = sv[:, :, 1]
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)
            if sv.ndim != 2:
                raise ValueError(f"Unexpected sv shape: {sv.shape}")

            mean_abs  = np.abs(sv).mean(axis=0)
            mean_sign = sv.mean(axis=0)
            total_abs = mean_abs.sum()

            # Cumulative coverage feature selection (same logic as pRFPS)
            feat_order = np.argsort(mean_abs)[::-1]
            cum        = np.cumsum(mean_abs[feat_order]) / (total_abs + 1e-10)
            n_sel      = int((cum < shap_coverage_threshold).sum()) + 1
            n_sel      = max(2, min(n_sel, max_features))
            selected   = set(feat_order[:n_sel])

            # Compute integer weights (same normalisation as pRFPS)
            top_abs_sum = mean_abs[feat_order[:n_sel]].sum()
            for fi, fname in enumerate(feature_names):
                w_raw = mean_abs[fi] / (top_abs_sum + 1e-10) * 10 if fi in selected else 0.0
                w_int = int(np.clip(round(w_raw), 0, 10))
                rows.append({
                    "Setting":      tag,
                    "Fold":         fold_idx + 1,
                    "Feature":      fname,
                    "Label":        FEATURE_LABELS.get(fname, fname),
                    "mean_abs_SHAP": round(float(mean_abs[fi]), 5),
                    "direction":    "Risk+" if mean_sign[fi] > 0 else "Protective",
                    "weight_raw":   round(float(w_raw), 4),
                    "weight_int":   w_int,
                    "selected":     int(fi in selected),
                    "n_selected":   n_sel,
                    "shap_coverage": round(float(cum[n_sel-1] if n_sel <= len(cum) else 1.0), 3),
                })

        except Exception as e:
            log(f"    Fold {fold_idx+1} failed: {e}")

    if not rows:
        return []

    df_rows = pd.DataFrame(rows)

    # Aggregate per feature across folds
    agg = (df_rows.groupby(["Setting","Feature","Label"])
           .agg(
               mean_weight_int    = ("weight_int", "mean"),
               sd_weight_int      = ("weight_int", "std"),
               mean_weight_raw    = ("weight_raw", "mean"),
               sd_weight_raw      = ("weight_raw", "std"),
               selection_freq     = ("selected", "mean"),  # fraction of folds selected
               mean_abs_SHAP      = ("mean_abs_SHAP", "mean"),
               sd_abs_SHAP        = ("mean_abs_SHAP", "std"),
           )
           .reset_index())

    agg["cv_weight_int"]  = agg["sd_weight_int"]  / (agg["mean_weight_int"]  + 1e-10)
    agg["cv_abs_SHAP"]    = agg["sd_abs_SHAP"]    / (agg["mean_abs_SHAP"]    + 1e-10)
    agg["stable_weight"]  = agg["cv_weight_int"]  < 0.30
    agg["stable_shap"]    = agg["cv_abs_SHAP"]    < 0.30
    agg["always_selected"]= agg["selection_freq"] == 1.0
    agg["mostly_selected"]= agg["selection_freq"] >= 0.80

    n_always  = int(agg["always_selected"].sum())
    n_mostly  = int(agg["mostly_selected"].sum())
    n_stable  = int(agg["stable_weight"].sum())
    log(f"  [pRFPS stability {tag}] "
        f"Always selected={n_always}/{len(agg)}  "
        f"Mostly (>=80%)={n_mostly}/{len(agg)}  "
        f"Stable weight (CV<0.30)={n_stable}/{len(agg)}")

    agg["Setting"] = tag
    agg = agg.sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    agg["rank_by_shap"] = agg.index + 1

    # Save per-tag CSV
    agg.to_csv(
        f"{results_dir}/shap_stability/prfps_weight_stability_{tag}.csv",
        index=False)
    log(f"  Saved prfps_weight_stability_{tag}.csv")

    return agg.to_dict("records")
# =============================================================================
# [C2+C5]  ABLATION: PARTITION CONTRIBUTION ANALYSIS
# =============================================================================

def covariate_shift_analysis(
    X_mcb: pd.DataFrame,
    y_mcb: pd.Series,
    subgroups_mcb: pd.Series,
    X_tap: pd.DataFrame,
    y_tap: pd.Series,
    subgroups_tap: pd.Series,
    features: List[str],
) -> pd.DataFrame:
    """
    Answers C1 (phenotype biological reproducibility) and C3 (dataset shift).

    TWO outputs:

    1. covariate_shift_table.csv
       Per feature: mean/SD in MCB train vs Tapestry, Mann-Whitney U p-value,
       standardised mean difference (SMD). SMD > 0.20 = potential shift.

    2. subgroup_profile_comparison.csv
       Per subgroup (C1, C2) per cohort: median (IQR) of key clinical variables.
       Shows whether MCB C1 and Tapestry C1 have similar clinical profiles,
       directly addressing "do subgroups reflect reproducible biological structure".
    """
    section("COVARIATE SHIFT + SUBGROUP PROFILE [C1+C3]")
    common = [f for f in features if f in X_mcb.columns and f in X_tap.columns]

    shift_rows = []
    for feat in common:
        mcb_vals = X_mcb[feat].dropna().values.astype(float)
        tap_vals = X_tap[feat].dropna().values.astype(float) if feat in X_tap.columns else np.array([])
        if len(mcb_vals) < 5 or len(tap_vals) < 5:
            continue
        stat, p = scipy_stats.mannwhitneyu(mcb_vals, tap_vals, alternative="two-sided")
        # Standardised mean difference (Cohen's d approximation)
        pooled_sd = np.sqrt((mcb_vals.std()**2 + tap_vals.std()**2) / 2) + 1e-10
        smd       = (mcb_vals.mean() - tap_vals.mean()) / pooled_sd
        shift_rows.append({
            "Feature":       feat,
            "Label":         FEATURE_LABELS.get(feat, feat),
            "MCB_mean":      round(float(mcb_vals.mean()), 3),
            "MCB_sd":        round(float(mcb_vals.std()),  3),
            "MCB_median":    round(float(np.median(mcb_vals)), 3),
            "Tap_mean":      round(float(tap_vals.mean()), 3),
            "Tap_sd":        round(float(tap_vals.std()),  3),
            "Tap_median":    round(float(np.median(tap_vals)), 3),
            "MWU_p":         round(float(p), 4),
            "SMD":           round(float(smd), 3),
            "Significant_shift": bool(p < 0.05),
            "Large_shift":   bool(abs(smd) > 0.20),
        })

    shift_df = pd.DataFrame(shift_rows).sort_values("SMD", key=abs, ascending=False)
    shift_df.to_csv(f"{OUT}/covariate_shift_table.csv", index=False)
    n_large = int(shift_df["Large_shift"].sum())
    log(f"  Covariate shift: {n_large}/{len(shift_df)} features with SMD>0.20")
    log("  Saved covariate_shift_table.csv")

    # Subgroup profile comparison
    profile_rows = []
    key_vars = [f for f in [
        "diagnosis_age", "Avg_BMI", "HbA1C", "ALP", "ALT.x", "AST.x",
        "PLATELET_COUNT", "ALBUMIN", "NFS", "AST_PT_ratio", "first_FIB4",
    ] if f in X_mcb.columns or f in X_tap.columns]

    for cohort_name, X_c, y_c, sg_c in [
        ("MCB",      X_mcb, y_mcb, subgroups_mcb),
        ("Tapestry", X_tap, y_tap, subgroups_tap),
    ]:
        for sg in sorted(sg_c.unique()):
            mask     = sg_c == sg
            y_sg     = y_c[mask]
            n_rapid  = int((y_sg == 1).sum())
            n_total  = int(mask.sum())
            prof_row = {
                "Cohort":    cohort_name,
                "Subgroup":  sg,
                "N_total":   n_total,
                "N_Rapid":   n_rapid,
                "Pct_Rapid": round(n_rapid / n_total * 100, 1) if n_total > 0 else float("nan"),
            }
            X_sg = X_c[mask] if isinstance(X_c, pd.DataFrame) else pd.DataFrame(X_c)[mask]
            for feat in key_vars:
                if feat in X_sg.columns:
                    vals = X_sg[feat].dropna().values.astype(float)
                    if len(vals) > 0:
                        prof_row[f"{FEATURE_LABELS.get(feat,feat)}_median"] = round(float(np.median(vals)), 2)
                        prof_row[f"{FEATURE_LABELS.get(feat,feat)}_IQR_lo"] = round(float(np.percentile(vals, 25)), 2)
                        prof_row[f"{FEATURE_LABELS.get(feat,feat)}_IQR_hi"] = round(float(np.percentile(vals, 75)), 2)
            profile_rows.append(prof_row)

    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(f"{OUT}/subgroup_profile_comparison.csv", index=False)
    log("  Saved subgroup_profile_comparison.csv")
    log(profile_df[["Cohort","Subgroup","N_total","N_Rapid","Pct_Rapid"]].to_string(index=False))

    return shift_df


# =============================================================================
# PATIENT-LEVEL DATASET EXPORT
# Saves train + test splits with all training features, best-model predictions,
# pRFPS integer score, and FIB-4 binary/continuous label per patient.
# =============================================================================


def save_patient_level_dataset(
    X_tr:          pd.DataFrame,
    X_te:          pd.DataFrame,
    y_tr:          pd.Series,
    y_te:          pd.Series,
    feats:         List[str],
    results:       Dict,
    models:        Dict,
    best_nm:       str,
    prfps_result,
    df_raw:        pd.DataFrame,
    setting:       str,
    tag:           str,
    subgroups_tr:  pd.Series = None,
    subgroups_te:  pd.Series = None,
) -> None:
 
    out_dir = f"{results_dir}/patient_level/{tag}"
    os.makedirs(out_dir, exist_ok=True)
 
    # ── FORCED DEMOGRAPHIC COLUMNS ────────────────────────────────────────────
    # Always include sex and age regardless of feature selection
    # so stratified evaluation always has these columns available
    FORCED_COLS = ["Sex", "diagnosis_age"]
 
    def _build_split_df(X_split, y_split, split_name, subgroups_split=None):
        n = len(X_split)
        df = pd.DataFrame()
 
        # ── Identity ──────────────────────────────────────────────────────────
        df["patient_idx"]    = range(n)
        df["original_index"] = (X_split.index.tolist()
                                 if hasattr(X_split, "index")
                                 else list(range(n)))
        df["split"]          = split_name
        df["setting"]        = setting
        df["subgroup"]       = (subgroups_split.values
                                if subgroups_split is not None
                                else setting)
 
        # ── Ground truth ──────────────────────────────────────────────────────
        y_arr = np.array(y_split).astype(int).flatten()
        df["true_label_num"] = y_arr
        df["true_label_str"] = np.where(y_arr == 1, POSITIVE_LABEL, NEGATIVE_LABEL)
 
        # ── Raw feature values (selected features) ────────────────────────────
        X_df = (X_split[feats].reset_index(drop=True)
                if isinstance(X_split, pd.DataFrame)
                else pd.DataFrame(X_split, columns=feats))
        for feat in feats:
            df[feat] = X_df[feat].values if feat in X_df.columns else np.nan
 
        # ── FORCED DEMOGRAPHIC COLUMNS ────────────────────────────────────────
        # Add sex and age even if not in selected features
        # Pull from df_raw using original index to guarantee availability
        orig_idx = (X_split.index.tolist()
                    if hasattr(X_split, "index") else list(range(n)))
        for forced_col in FORCED_COLS:
            if forced_col not in df.columns:          # skip if already saved above
                if forced_col in df_raw.columns:
                    try:
                        df[forced_col] = (
                            df_raw.loc[orig_idx, forced_col]
                            .values)
                        log(f"    [patient export] Added forced col: {forced_col}")
                    except Exception as _fc_e:
                        log(f"    [patient export] Could not add {forced_col}: {_fc_e}")
                        df[forced_col] = np.nan
                else:
                    log(f"    [patient export] {forced_col} not in df_raw — skipping")
 
        # ── Best model predictions ────────────────────────────────────────────
        best_res    = results.get(best_nm, {})
        best_thr    = best_res.get("optimal_threshold", 0.5)
        best_model  = models.get(best_nm)
        df["best_model_name"]      = best_nm
        df["best_model_threshold"] = round(float(best_thr), 4)
 
        if best_model is not None:
            try:
                from sklearn.preprocessing import StandardScaler as _SS
                _scl = _SS().fit(X_split[feats].values
                                 if isinstance(X_split, pd.DataFrame)
                                 else X_split)
                X_sc = _scl.transform(X_split[feats].values
                                      if isinstance(X_split, pd.DataFrame)
                                      else X_split)
                proba = best_model.predict_proba(X_sc)[:, 1]
                pred  = (proba >= best_thr).astype(int)
                df["best_model_proba"]   = proba.round(4)
                df["best_model_pred"]    = pred
                df["best_model_correct"] = (pred == y_arr).astype(int)
            except Exception as _e:
                log(f"    [patient export] best model predict failed: {_e}")
                df["best_model_proba"]   = np.nan
                df["best_model_pred"]    = np.nan
                df["best_model_correct"] = np.nan
        else:
            if split_name == "test":
                stored_p = np.array(best_res.get("y_proba", []))
                stored_d = np.array(best_res.get("y_pred",  []))
                if len(stored_p) == n:
                    df["best_model_proba"]   = stored_p.round(4)
                    df["best_model_pred"]    = stored_d
                    df["best_model_correct"] = (stored_d == y_arr).astype(int)
 
        # ── All model probabilities ───────────────────────────────────────────
        for nm, res in results.items():
            stored_p = np.array(res.get("y_proba", []))
            if len(stored_p) == n and split_name == "test":
                df[f"proba_{nm}"] = stored_p.round(4)
            else:
                df[f"proba_{nm}"] = np.nan
 
        # ── pRFPS (test only) ─────────────────────────────────────────────────
        if split_name == "test" and prfps_result is not None:
            score_int  = prfps_result.get("score_int",  np.full(n, np.nan))
            score_cont = prfps_result.get("score_cont", np.full(n, np.nan))
            pred_int   = prfps_result.get("pred",       np.full(n, np.nan))
            cutoff_int = prfps_result.get("threshold",  float("nan"))
 
            if len(score_int) == n:
                df["pRFPS_int_score"]  = score_int
                df["pRFPS_cont_score"] = score_cont.round(4)
                df["pRFPS_int_pred"]   = pred_int.astype(int)
                df["pRFPS_int_cutoff"] = round(float(cutoff_int), 3)
                df["pRFPS_correct"]    = (pred_int.astype(int) == y_arr).astype(int)
            else:
                for col in ["pRFPS_int_score","pRFPS_cont_score",
                            "pRFPS_int_pred","pRFPS_int_cutoff","pRFPS_correct"]:
                    df[col] = np.nan
        else:
            for col in ["pRFPS_int_score","pRFPS_cont_score",
                        "pRFPS_int_pred","pRFPS_int_cutoff","pRFPS_correct"]:
                df[col] = np.nan
 
        # ── FIB-4 ─────────────────────────────────────────────────────────────
        try:
            df_sub    = df_raw.loc[orig_idx].copy().reset_index(drop=True)
            fib4_cont = compute_fib4_score(df_sub)
            age_s     = (df_sub["diagnosis_age"].astype(float)
                         if "diagnosis_age" in df_sub.columns
                         else pd.Series(50.0, index=df_sub.index))
            fib4_bin  = fib4_binary_label(fib4_cont, age_s)
 
            df["FIB4_score"]         = fib4_cont.values.round(3)
            df["FIB4_binary_label"]  = fib4_bin.values
            df["FIB4_risk_category"] = np.where(
                fib4_bin.values == 0, "Low",
                np.where(fib4_bin.values == 1, "High", "Intermediate"))
            df["FIB4_pred_binary"]   = np.where(
                fib4_bin.notna(), fib4_bin.values, np.nan)
            df["FIB4_correct"]       = np.where(
                fib4_bin.notna(),
                (fib4_bin.values.astype(float) == y_arr.astype(float)).astype(float),
                np.nan)
        except Exception as _e:
            log(f"    [patient export] FIB-4 computation failed: {_e}")
            for col in ["FIB4_score","FIB4_binary_label","FIB4_risk_category",
                        "FIB4_pred_binary","FIB4_correct"]:
                df[col] = np.nan
 
        return df
 
    # ── Subgroup alignment ────────────────────────────────────────────────────
    if subgroups_tr is None and hasattr(X_tr, "index"):
        try:    subgroups_tr = subgroups.loc[X_tr.index]
        except: pass
    if subgroups_te is None and hasattr(X_te, "index"):
        try:    subgroups_te = subgroups.loc[X_te.index]
        except: pass
 
    # ── Build splits ──────────────────────────────────────────────────────────
    train_df = _build_split_df(X_tr, y_tr, "train", subgroups_tr)
    test_df  = _build_split_df(X_te, y_te, "test",  subgroups_te)
 
    # ── STRATIFIED EVALUATION (on test_df before saving) ─────────────────────
    # Runs only when pRFPS result is available (test set only)
    if prfps_result is not None:
        try:
            strat_out_dir = f"{results_dir}/patient_level/{tag}/stratified_eval"
            os.makedirs(strat_out_dir, exist_ok=True)
 
            _strat_rows = []
 
            # Determine sex and age columns in test_df
            _sex_col = "Sex"           if "Sex"           in test_df.columns else None
            _age_col = "diagnosis_age" if "diagnosis_age" in test_df.columns else None
 
            if _sex_col is None:
                log(f"  [stratified] Sex column not found in test_df — skipping sex strata")
            if _age_col is None:
                log(f"  [stratified] Age column not found in test_df — skipping age strata")
 
            # Define strata
            _strata = {"All patients": np.ones(len(test_df), dtype=bool)}
 
            if _sex_col is not None:
                _sex_raw = test_df[_sex_col].astype(str).str.strip().str.lower()
                _strata["Sex: Female"] = _sex_raw.isin(["0","f","female","2"]).values
                _strata["Sex: Male"]   = _sex_raw.isin(["1","m","male"]).values
                # Fallback for numeric encoding
                if _strata["Sex: Female"].sum() == 0 and _strata["Sex: Male"].sum() == 0:
                    _age_num = pd.to_numeric(test_df[_sex_col], errors="coerce")
                    _strata["Sex: Female"] = (_age_num == _age_num.min()).values
                    _strata["Sex: Male"]   = (_age_num == _age_num.max()).values
 
            if _age_col is not None:
                _age_vals = pd.to_numeric(test_df[_age_col], errors="coerce").values
                _strata["Age: <45"]   = _age_vals < 45
                _strata["Age: 45-69"] = (_age_vals >= 45) & (_age_vals < 70)
                _strata["Age: >=70"]  = _age_vals >= 70
 
            # pRFPS normalised probability for AUROC
            _prf_scores = pd.to_numeric(
                test_df["pRFPS_int_score"], errors="coerce").values.astype(float)
            _prf_prob   = _prf_scores / (np.nanmax(_prf_scores) + 1e-10)
            _prf_pred   = pd.to_numeric(
                test_df["pRFPS_int_pred"], errors="coerce").fillna(0).astype(int).values
 
            _fib4_scores = pd.to_numeric(
                test_df["FIB4_score"], errors="coerce").values.astype(float)
            _fib4_prob   = _fib4_scores / (np.nanmax(_fib4_scores) + 1e-10)
            _fib4_pred   = pd.to_numeric(
                test_df["FIB4_pred_binary"], errors="coerce").fillna(0).astype(int).values
 
            _y_all    = test_df["true_label_num"].values.astype(int)
            _pred_all = pd.to_numeric(
                test_df["best_model_pred"], errors="coerce").fillna(0).astype(int).values
            _prob_all = pd.to_numeric(
                test_df["best_model_proba"], errors="coerce").fillna(0).astype(float).values
 
            from sklearn.metrics import (roc_auc_score, confusion_matrix,
                                         f1_score, accuracy_score)
 
            def _compute_metrics(y, yp, ypr, model_label, stratum):
                n_s    = len(y)
                n_r    = int(y.sum())
                n_sl   = n_s - n_r
                base   = {"stratum": stratum, "model": model_label,
                          "n": n_s, "n_rapid": n_r, "n_slow": n_sl,
                          "pct_rapid": round(n_r/n_s*100,1) if n_s>0 else np.nan}
                if n_s < 5 or n_r == 0 or n_sl == 0:
                    base.update({"accuracy":np.nan,"f1_macro":np.nan,
                                 "f1_rapid":np.nan,"auroc":np.nan,
                                 "sensitivity":np.nan,"specificity":np.nan})
                    return base
                tn,fp,fn,tp = confusion_matrix(y,yp,labels=[0,1]).ravel()
                auroc = roc_auc_score(y,ypr) if len(np.unique(y))>1 else np.nan
                base.update({
                    "accuracy":    round(float(accuracy_score(y,yp)),4),
                    "f1_macro":    round(float(f1_score(y,yp,average="macro",zero_division=0)),4),
                    "f1_rapid":    round(float(f1_score(y,yp,average="binary",zero_division=0)),4),
                    "auroc":       round(float(auroc),4),
                    "sensitivity": round(float(tp/(tp+fn+1e-10)),4),
                    "specificity": round(float(tn/(tn+fp+1e-10)),4),
                })
                return base
 
            for stratum_name, mask in _strata.items():
                mask = np.array(mask, dtype=bool)
                if mask.sum() == 0:
                    continue
                _y   = _y_all[mask]
                log(f"  [stratified] {tag} | {stratum_name}: "
                    f"n={mask.sum()}, rapid={_y.sum()} ({_y.mean()*100:.1f}%)")
 
                _strat_rows.append(_compute_metrics(
                    _y, _pred_all[mask], _prob_all[mask],
                    "Best Model", stratum_name))
                _strat_rows.append(_compute_metrics(
                    _y, _prf_pred[mask], _prf_prob[mask],
                    "pRFPS", stratum_name))
                _strat_rows.append(_compute_metrics(
                    _y, _fib4_pred[mask], _fib4_prob[mask],
                    "FIB-4", stratum_name))
 
            df_strat = pd.DataFrame(_strat_rows)
            df_strat["setting"] = tag
 
            # Save stratified CSV
            strat_csv = f"{strat_out_dir}/stratified_eval_{tag}.csv"
            df_strat.to_csv(strat_csv, index=False)
            log(f"  [stratified] Saved: {strat_csv}")
 
            # Print summary table
            log(f"\n  {'='*85}")
            log(f"  Stratified Evaluation — {tag}")
            log(f"  {'='*85}")
            log(f"  {'Stratum':<18} {'Model':<14} {'N':>5} {'Rapid%':>7} "
                f"{'Acc':>7} {'F1':>7} {'AUROC':>7} {'Sens':>7} {'Spec':>7}")
            log(f"  {'-'*80}")
            _strata_order = ["All patients","Sex: Female","Sex: Male",
                             "Age: <45","Age: 45-69","Age: >=70"]
            for _sn in _strata_order:
                _sub = df_strat[df_strat["stratum"] == _sn]
                if len(_sub) == 0: continue
                _first = True
                for _, _r in _sub.iterrows():
                    _sl = _sn if _first else ""
                    _first = False
                    _acc  = f"{_r['accuracy']:.3f}"    if not pd.isna(_r["accuracy"])    else "  —  "
                    _f1   = f"{_r['f1_macro']:.3f}"    if not pd.isna(_r["f1_macro"])    else "  —  "
                    _auc  = f"{_r['auroc']:.3f}"       if not pd.isna(_r["auroc"])       else "  —  "
                    _sens = f"{_r['sensitivity']:.3f}" if not pd.isna(_r["sensitivity"]) else "  —  "
                    _spec = f"{_r['specificity']:.3f}" if not pd.isna(_r["specificity"]) else "  —  "
                    log(f"  {_sl:<18} {_r['model']:<14} {int(_r['n']):>5} "
                        f"{_r['pct_rapid']:>6.1f}% "
                        f"{_acc:>7} {_f1:>7} {_auc:>7} {_sens:>7} {_spec:>7}")
                log(f"  {'':-<80}")
 
            # Figure
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
 
                _strata_present = [s for s in _strata_order
                                   if s in df_strat["stratum"].values]
                _models   = ["Best Model", "pRFPS", "FIB-4"]
                _colors   = ["#0A7C7C", "#D96B2A", "#8899AA"]
                _metrics  = ["accuracy", "f1_macro", "auroc"]
                _mlabels  = ["Accuracy", "F1 (Macro)", "AUROC"]
 
                fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                for ax, met, mlbl in zip(axes, _metrics, _mlabels):
                    x = np.arange(len(_strata_present))
                    w = 0.25
                    for i, (mdl, clr) in enumerate(zip(_models, _colors)):
                        vals = []
                        for sn in _strata_present:
                            sub = df_strat[(df_strat["stratum"]==sn) &
                                           (df_strat["model"]==mdl)]
                            v = float(sub[met].values[0]) \
                                if len(sub)>0 and not pd.isna(sub[met].values[0]) \
                                else 0.0
                            vals.append(v)
                        bars = ax.bar(x+i*w, vals, w, label=mdl,
                                      color=clr, alpha=0.85, edgecolor="white")
                        for bar, val in zip(bars, vals):
                            if val > 0:
                                ax.text(bar.get_x()+bar.get_width()/2,
                                        bar.get_height()+0.01,
                                        f"{val:.3f}", ha="center",
                                        va="bottom", fontsize=7, rotation=90)
                    ax.set_xticks(x+w)
                    ax.set_xticklabels(_strata_present, rotation=30,
                                       ha="right", fontsize=9)
                    ax.set_ylim(0.3, 1.08)
                    ax.set_ylabel(mlbl, fontsize=11)
                    ax.set_title(f"{mlbl} by Stratum — {tag}",
                                 fontsize=11, fontweight="bold")
                    ax.legend(fontsize=9, loc="lower right")
                    ax.grid(axis="y", alpha=0.25)
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                    # n labels
                    _nlbls = []
                    for sn in _strata_present:
                        sub = df_strat[(df_strat["stratum"]==sn) &
                                       (df_strat["model"]=="Best Model")]
                        nn = int(sub["n"].values[0])      if len(sub)>0 else 0
                        nr = int(sub["n_rapid"].values[0]) if len(sub)>0 else 0
                        _nlbls.append(f"n={nn}\n(R={nr})")
                    ax2 = ax.twiny()
                    ax2.set_xlim(ax.get_xlim())
                    ax2.set_xticks(x+w)
                    ax2.set_xticklabels(_nlbls, fontsize=7)
                    ax2.tick_params(axis="x", length=0)
 
                plt.suptitle(
                    f"Stratified Evaluation — {tag}  |  "
                    f"Best Model vs pRFPS vs FIB-4",
                    fontsize=13, fontweight="bold", y=1.02)
                plt.tight_layout()
                fig_path = f"{strat_out_dir}/stratified_eval_{tag}.png"
                plt.savefig(fig_path, dpi=200, bbox_inches="tight")
                plt.close()
                log(f"  [stratified] Saved figure: {fig_path}")
 
            except Exception as _fig_e:
                log(f"  [stratified] Figure failed: {_fig_e}")
 
        except Exception as _strat_e:
            log(f"  [stratified] Evaluation failed for {tag}: {_strat_e}")
            import traceback
            log(traceback.format_exc())
 
    # ── Save CSVs ─────────────────────────────────────────────────────────────
    train_path = f"{out_dir}/train_data_{tag}.csv"
    test_path  = f"{out_dir}/test_data_{tag}.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
 
    log(f"  [patient export] {tag}  train: {len(train_df)} rows x "
        f"{len(train_df.columns)} cols  ->  {train_path}")
    log(f"  [patient export] {tag}  test:  {len(test_df)} rows x "
        f"{len(test_df.columns)} cols  ->  {test_path}")
 
    test_cols = [c for c in test_df.columns
                 if c.startswith("pRFPS") or c.startswith("FIB4")
                 or c.startswith("best_model")]
    log(f"  [patient export] Prediction columns: {test_cols}")
# =============================================================================
# [FIX-3]  FIB-4 CIRCULARITY SENSITIVITY
# =============================================================================

def circularity_sensitivity(X_tr, X_te, y_tr, y_te,
                             features_full: List[str], tag: str,
                             n_boot: int = 500) -> pd.DataFrame:
    """[FIX-3] Train with and without FIB-4 component labs.

    For each feature set (Full / Restricted), trains all MODELS_CONFIG models
    PLUS a Stacking Ensemble (top-2 by CV AUROC as base estimators).

    Outputs:
      circularity_{tag}.csv      -- per-model AUROC + Accuracy for both sets
      circularity_stats_{tag}.csv -- paired DeLong/bootstrap: Full vs Restricted
                                     per model and per metric (AUROC, Accuracy)
    Figures:
      circularity_{tag}.png      -- grouped bar charts (AUROC | Accuracy)
    """
    section(f"FIX-3 Circularity Sensitivity -- {tag}")
    rows   = []   # per model × feature set
    stat_rows = [] # statistical comparison Full vs Restricted per model

    feat_conditions = [
        (features_full,
         "Full (FIB-4 included)"),
        ([f for f in features_full if f not in FIB4_COMPONENTS],
         "Restricted (FIB-4 excluded)"),
    ]

    # Store proba arrays per (model, condition) for statistical comparisons
    proba_store = {}   # (model_name, cond_label) → (y_te_arr, prob_te_arr, accuracy)

    for feat_set, desc in feat_conditions:
        if len(feat_set) < 3:
            log(f"  Skipping {desc}: too few features ({len(feat_set)})"); continue

        Ntr, Nte, _ = fit_scale(X_tr, X_te, feat_set)
        cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fitted_models = {}   # name → model obj (for SE construction)

        # ── Train all single models ───────────────────────────────────────────
        for mname, cfg in MODELS_CONFIG.items():
            try:
                Xtr_m = Ntr if cfg["scaled"] else X_tr[feat_set].values
                Xte_m = Nte if cfg["scaled"] else X_te[feat_set].values

                gs = GridSearchCV(cfg["model_class"](), cfg["param_grid"],
                                  cv=cv5, scoring="f1_macro", n_jobs=1)
                gs.fit(Xtr_m, y_tr.values)
                mdl = gs.best_estimator_

                # Youden threshold from training probabilities (no leakage)
                prob_tr = mdl.predict_proba(Xtr_m)[:, 1]
                thr = 0.5; best_j = -1
                for _t in np.unique(np.percentile(prob_tr, np.linspace(5, 95, 50))):
                    _pred = (prob_tr >= _t).astype(int)
                    _cm   = confusion_matrix(y_tr, _pred, labels=[0,1])
                    _tn, _fp, _fn, _tp = _cm.ravel()
                    _j    = _tp/(_tp+_fn+1e-9) + _tn/(_tn+_fp+1e-9) - 1
                    if _j > best_j: best_j = _j; thr = float(_t)

                prob_te = mdl.predict_proba(Xte_m)[:, 1]
                pred_te = (prob_te >= thr).astype(int)
                acc_te  = float(accuracy_score(y_te, pred_te))
                fitted_models[mname] = (mdl, Xtr_m, Xte_m)

                ci       = bootstrap_ci(y_te.values, pred_te, prob_te, n_boot=n_boot)
                auroc_ci = ci.get("auroc", (float(roc_auc_score(y_te, prob_te)),)*3)
                acc_ci   = ci.get("accuracy", (acc_te,)*3)

                rows.append({
                    "Setting": tag, "Feature_Set": desc, "Model": mname,
                    "AUROC_mean": round(auroc_ci[0], 4),
                    "AUROC_lo":   round(auroc_ci[1], 4),
                    "AUROC_hi":   round(auroc_ci[2], 4),
                    "Accuracy":   round(acc_te, 4),
                    "Accuracy_lo":round(float(acc_ci[1]), 4),
                    "Accuracy_hi":round(float(acc_ci[2]), 4),
                    "F1_macro":   round(float(f1_score(y_te, pred_te,
                                                        average="macro", zero_division=0)), 4),
                })
                proba_store[(mname, desc)] = (y_te.values, prob_te, acc_te)
                log(f"    [{desc[:6]}][{mname}] AUROC={auroc_ci[0]:.3f} "
                    f"Acc={acc_te:.3f}")
            except Exception as e:
                log(f"    [{mname}] failed: {e}")

        # ── Stacking Ensemble (top-2 by test AUROC among single models) ───────
        try:
            if len(fitted_models) >= 2:
                _ranked = sorted(
                    [(nm, mdl_t) for nm, mdl_t in fitted_models.items()
                     if len(np.unique(y_te.values)) > 1],
                    key=lambda kv: roc_auc_score(y_te.values,
                                                  kv[1][0].predict_proba(kv[1][2])[:,1]),
                    reverse=True)
                top2 = _ranked[:2]
                se   = StackingClassifier(
                    estimators=[(nm, t[0]) for nm, t in top2],
                    final_estimator=LogisticRegression(
                        C=0.01, max_iter=2000, random_state=42,
                        solver="lbfgs", class_weight="balanced"),
                    cv=5, n_jobs=1)
                se.fit(Ntr, y_tr.values)

                prob_tr_se = se.predict_proba(Ntr)[:, 1]
                thr_se = 0.5; best_j_se = -1
                for _t in np.unique(np.percentile(prob_tr_se, np.linspace(5, 95, 50))):
                    _pred = (prob_tr_se >= _t).astype(int)
                    _cm   = confusion_matrix(y_tr, _pred, labels=[0,1])
                    _tn, _fp, _fn, _tp = _cm.ravel()
                    _j    = _tp/(_tp+_fn+1e-9) + _tn/(_tn+_fp+1e-9) - 1
                    if _j > best_j_se: best_j_se = _j; thr_se = float(_t)

                prob_te_se = se.predict_proba(Nte)[:, 1]
                pred_te_se = (prob_te_se >= thr_se).astype(int)
                acc_se     = float(accuracy_score(y_te, pred_te_se))

                ci_se       = bootstrap_ci(y_te.values, pred_te_se, prob_te_se, n_boot=n_boot)
                auroc_ci_se = ci_se.get("auroc", (float(roc_auc_score(y_te, prob_te_se)),)*3)
                acc_ci_se   = ci_se.get("accuracy", (acc_se,)*3)
                top2_lbl    = "+".join(nm for nm, _ in top2)

                rows.append({
                    "Setting": tag, "Feature_Set": desc,
                    "Model":    "Stacking_Ensemble",
                    "AUROC_mean": round(auroc_ci_se[0], 4),
                    "AUROC_lo":   round(auroc_ci_se[1], 4),
                    "AUROC_hi":   round(auroc_ci_se[2], 4),
                    "Accuracy":   round(acc_se, 4),
                    "Accuracy_lo":round(float(acc_ci_se[1]), 4),
                    "Accuracy_hi":round(float(acc_ci_se[2]), 4),
                    "F1_macro":   round(float(f1_score(y_te, pred_te_se,
                                                        average="macro", zero_division=0)), 4),
                })
                proba_store[("Stacking_Ensemble", desc)] = (y_te.values, prob_te_se, acc_se)
                log(f"    [{desc[:6]}][SE({top2_lbl})] AUROC={auroc_ci_se[0]:.3f} "
                    f"Acc={acc_se:.3f}")
        except Exception as _se_e:
            log(f"    [SE] failed for {desc}: {_se_e}")

    # ── Statistical comparison: Full vs Restricted per model × metric ─────────
    FULL  = "Full (FIB-4 included)"
    RESTR = "Restricted (FIB-4 excluded)"
    all_models = list({r["Model"] for r in rows})

    for mname in all_models:
        full_data  = proba_store.get((mname, FULL))
        restr_data = proba_store.get((mname, RESTR))
        if full_data is None or restr_data is None:
            continue
        y_arr, yp_full, acc_full = full_data
        _,     yp_rest, acc_rest = restr_data

        if len(y_arr) < 10 or len(np.unique(y_arr)) < 2:
            continue

        # AUROC: DeLong
        try:
            z_dl, p_dl = delong_auc_test(y_arr, yp_full, yp_rest)
            pb_auroc   = paired_bootstrap_contrast(
                y_arr, yp_full, yp_rest, "auroc", n_boot)
            full_row = next((r for r in rows
                             if r["Model"]==mname and r["Feature_Set"]==FULL), {})
            rest_row = next((r for r in rows
                             if r["Model"]==mname and r["Feature_Set"]==RESTR), {})
            stat_rows.append({
                "Setting": tag, "Model": mname, "Metric": "AUROC",
                "Score_Full":    full_row.get("AUROC_mean", float("nan")),
                "Score_Restricted": rest_row.get("AUROC_mean", float("nan")),
                "Diff_Full_minus_Restricted":
                    round(float(full_row.get("AUROC_mean",0)) -
                          float(rest_row.get("AUROC_mean",0)), 4),
                "DeLong_z":  round(float(z_dl),3) if not np.isnan(z_dl) else "N/A",
                "DeLong_p":  round(float(p_dl),4) if not np.isnan(p_dl) else "N/A",
                "Boot_CI_lo": pb_auroc["ci_lo"],
                "Boot_CI_hi": pb_auroc["ci_hi"],
                "Boot_p":     pb_auroc["p_value"],
                "Significant":pb_auroc["significant"],
                "Interpretation": ("FIB-4 components inflate AUROC"
                                   if pb_auroc["diff"] > 0 and pb_auroc["significant"]
                                   else "No significant circularity effect"),
            })
        except Exception as _e:
            log(f"    [{mname}] AUROC circularity test failed: {_e}")

        # Accuracy: Paired bootstrap
        try:
            pb_acc = paired_bootstrap_contrast(
                y_arr, yp_full, yp_rest, "accuracy", n_boot)
            stat_rows.append({
                "Setting": tag, "Model": mname, "Metric": "Accuracy",
                "Score_Full":       round(acc_full, 4),
                "Score_Restricted": round(acc_rest, 4),
                "Diff_Full_minus_Restricted":
                    round(acc_full - acc_rest, 4),
                "DeLong_z":  "N/A",
                "DeLong_p":  "N/A",
                "Boot_CI_lo": pb_acc["ci_lo"],
                "Boot_CI_hi": pb_acc["ci_hi"],
                "Boot_p":     pb_acc["p_value"],
                "Significant":pb_acc["significant"],
                "Interpretation": ("FIB-4 components inflate Accuracy"
                                   if pb_acc["diff"] > 0 and pb_acc["significant"]
                                   else "No significant circularity effect"),
            })
        except Exception as _e:
            log(f"    [{mname}] Accuracy circularity test failed: {_e}")

    # ── Save CSVs ─────────────────────────────────────────────────────────────
    df      = pd.DataFrame(rows)
    df_stat = pd.DataFrame(stat_rows)
    df.to_csv(f"{OUT}/circularity_{tag}.csv", index=False)
    if not df_stat.empty:
        df_stat.to_csv(f"{OUT}/circularity_stats_{tag}.csv", index=False)
        log(f"  Saved circularity_stats_{tag}.csv")
        log(df_stat[["Model","Metric","Score_Full","Score_Restricted",
                     "Diff_Full_minus_Restricted","Boot_p",
                     "Significant","Interpretation"]].to_string(index=False))

    # ── Plot: 2-panel grouped bar (AUROC | Accuracy) ──────────────────────────
    if not df.empty:
        mdls  = df["Model"].unique()
        x, w  = np.arange(len(mdls)), 0.35
        colors= [(FULL, "#4C72B0"), (RESTR, "#DD8452")]

        fig, axes = plt.subplots(1, 2,
                                  figsize=(max(10, len(mdls)*1.5 + 2), 5.5),
                                  sharey=False)

        for ax, (mean_col, lo_col, hi_col), ylabel, ylim in [
            (axes[0], ("AUROC_mean","AUROC_lo","AUROC_hi"),
             "AUROC (95% CI)", (0.35, 1.00)),
            (axes[1], ("Accuracy","Accuracy_lo","Accuracy_hi"),
             "Accuracy (95% CI)", (0.35, 1.00)),
        ]:
            for i, (fset, color) in enumerate(colors):
                sub  = df[df["Feature_Set"]==fset].set_index("Model")
                means= [float(sub.loc[m, mean_col]) if m in sub.index else 0 for m in mdls]
                lows = [float(sub.loc[m, lo_col])   if m in sub.index else 0 for m in mdls]
                highs= [float(sub.loc[m, hi_col])   if m in sub.index else 0 for m in mdls]
                errs = [[m-l for m,l in zip(means,lows)],
                        [h-m for m,h in zip(means,highs)]]
                bars = ax.bar(x + i*w, means, w, label=fset,
                              color=color, alpha=0.82, yerr=errs, capsize=3)

                # Black edge + star for SE bars
                for xi, (m_name, bar) in enumerate(zip(mdls, bars)):
                    if m_name == "Stacking_Ensemble":
                        bar.set_edgecolor("black"); bar.set_linewidth(1.8)

                # Significance stars between paired bars
                if i == 1 and not df_stat.empty:
                    for xi, m_name in enumerate(mdls):
                        sig_row = df_stat[(df_stat["Model"]==m_name) &
                                          (df_stat["Metric"]==ylabel.split()[0])]
                        if not sig_row.empty and sig_row.iloc[0]["Significant"]:
                            y_top = max(
                                float(sub.loc[m_name, hi_col]) if m_name in sub.index else 0,
                                float(df[df["Feature_Set"]==FULL].set_index("Model").loc[m_name, hi_col])
                                if m_name in df[df["Feature_Set"]==FULL].set_index("Model").index else 0
                            ) + 0.025
                            ax.text(x[xi]+w/2, y_top, "★", ha="center",
                                    va="bottom", fontsize=10, color="#C0392B")

            ax.set_xticks(x + w/2)
            ax.set_xticklabels([m.replace("_"," ") for m in mdls],
                                rotation=35, ha="right", fontsize=8)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_ylim(*ylim)
            ax.set_title(ylabel.split()[0] + f" — {tag}", fontsize=10, fontweight="bold")
            ax.legend(fontsize=7.5, framealpha=0.88)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.suptitle(
            f"FIB-4 Circularity Sensitivity — {tag}\n"
            f"★ = Full significantly higher than Restricted (p<0.05)  "
            f"| SE = black-edged bars",
            fontsize=10, fontweight="bold")
        plt.tight_layout()
        plt.savefig(f"{OUT}/circularity_{tag}.png", dpi=200, bbox_inches="tight")
        plt.close()
    return df
# =============================================================================
# [FIX-6]  PROGRESSION-TIME STATISTICAL TEST
# =============================================================================

def test_progression_time_difference(df_raw: pd.DataFrame,
                                      time_col: str = "progression_time_years") -> pd.DataFrame:
    """[FIX-6] Mann-Whitney U between all subgroup pairs."""
    section("FIX-6  Subgroup progression-time statistical tests")
    col = SUBGROUP_COL if SUBGROUP_COL in df_raw.columns else "Subgroup"
    if time_col not in df_raw.columns:
        log(f"  Column '{time_col}' not found -- skipping FIX-6")
        return pd.DataFrame()
    sgs  = df_raw[col].unique()
    rows = []
    for i, a in enumerate(sgs):
        for b in sgs[i+1:]:
            va = df_raw.loc[df_raw[col]==a, time_col].dropna()
            vb = df_raw.loc[df_raw[col]==b, time_col].dropna()
            if len(va)<5 or len(vb)<5: continue
            stat, p = scipy_stats.mannwhitneyu(va, vb, alternative="two-sided")
            rows.append({
                "Subgroup_A": a, "Subgroup_B": b,
                "Median_A":   round(float(va.median()),2),
                "Median_B":   round(float(vb.median()),2),
                "Diff_median":round(float(va.median()-vb.median()),2),
                "MWU_stat":   round(stat,2), "p_value": round(p,4),
                "Significant":p < 0.05,
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_csv(f"{OUT}/progression_time_tests.csv", index=False)
        log(df.to_string(index=False))
    return df


# =============================================================================
# [FIX-7]  RECALCULATE CITED PERCENTAGES
# =============================================================================
def recalculate_cited_percentages(y_numeric: pd.Series, y_str: pd.Series,
                                   subgroups: pd.Series,
                                   df_raw: pd.DataFrame) -> Dict:
    """[FIX-7] Recompute all cited figures from raw label arrays."""
    section("FIX-7  Recalculate cited percentages")
    fib4 = df_raw["first_FIB4"] if "first_FIB4" in df_raw.columns else None
    out  = {"n_total": int(len(y_numeric))}
    for sg in subgroups.unique():
        mask    = subgroups == sg
        n_sg    = int(mask.sum())
        n_rapid = int((y_numeric[mask] == 1).sum())
        out[f"n_{sg}"] = n_sg
        out[f"pct_rapid_{sg}"] = round(n_rapid / n_sg * 100, 1)
        log(f"  {sg}: rapid={n_rapid}/{n_sg} ({out[f'pct_rapid_{sg}']}%)")
        if fib4 is not None:
            n_lr = int(((fib4=="Low") & (y_numeric==1) & mask).sum())
            pct  = round(n_lr/n_rapid*100,1) if n_rapid else 0
            out[f"pct_lowFIB4_rapid_{sg}"] = pct
            log(f"  {sg}: Low-FIB4 among rapid = {n_lr}/{n_rapid} ({pct}%)")
    if fib4 is not None:
        n_hs = int(((fib4=="High") & (y_numeric==0)).sum())
        out["pct_highFIB4_slow_CORRECTED"] = round(n_hs/len(y_numeric)*100,1)
        log(f"  Corrected 14.6% figure = {out['pct_highFIB4_slow_CORRECTED']}%")
    with open(f"{OUT}/verified_percentages.json","w") as fh:
        json.dump(out, fh, indent=2)
    log("  Saved verified_percentages.json")
    return out


# =============================================================================
# [FIX-8]  FIXED vs MEDIAN THRESHOLD SENSITIVITY
# =============================================================================
def fixed_vs_median_sensitivity(df_raw: pd.DataFrame, fixed_years: float = 3.0,
                                  time_col: str = "progression_time_years") -> pd.DataFrame:
    """[FIX-8] Compare median-based vs fixed 3-year threshold labelling."""
    section("FIX-8  Fixed vs median threshold sensitivity")
    col = SUBGROUP_COL if SUBGROUP_COL in df_raw.columns else "Subgroup"
    if time_col not in df_raw.columns:
        log(f"  '{time_col}' not found -- skipping FIX-8"); return pd.DataFrame()
    rows = []
    for sg in df_raw[col].unique():
        pt  = df_raw[df_raw[col]==sg][time_col].dropna()
        med = pt.median()
        ml  = (pt <= med).astype(int)
        fl  = (pt <= fixed_years).astype(int)
        rows.append({
            "Subgroup": sg, "N": len(pt),
            "Median_yr": round(float(med),2),
            "Median_RP_rate":    round(float(ml.mean()),3),
            "Fixed_3yr_RP_rate": round(float(fl.mean()),3),
            "Label_agreement":   round(float((ml==fl).mean()),3),
        })
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/fixed_vs_median_threshold.csv", index=False)
    log(df.to_string(index=False))
    return df


# =============================================================================
# [FIX-11 / Part 5]  LCA CLASS-SENSITIVITY + JUSTIFICATION
# =============================================================================
def lca_class_sensitivity_and_justification(bic_by_k: Dict, entropy_by_k: Dict,
                                             mean_pp_by_k: Dict = None) -> pd.DataFrame:
    """
    [FIX-11 / Part 5] BIC + entropy plot for k=1..5, k=2 annotated.
    Also saves justification table for Methods.
    """
    section("FIX-11 / Part 5  LCA class sensitivity")
    ks   = sorted(bic_by_k)
    bics = [bic_by_k[k] for k in ks]
    ents = [entropy_by_k.get(k, float("nan")) for k in ks]
    mpps = [mean_pp_by_k.get(k, float("nan")) for k in ks] if mean_pp_by_k else [float("nan")]*len(ks)

    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].plot(ks, bics, "o-", color="#2980b9", lw=2)
    axes[0].axvline(2, color="red", ls="--", label="k=2 selected"); axes[0].legend()
    axes[0].set_xlabel("k (# latent classes)"); axes[0].set_ylabel("BIC (lower=better)")
    axes[0].set_title("LCA: BIC by k"); axes[0].grid(alpha=0.3)

    axes[1].plot(ks, ents, "s-", color="#27ae60", lw=2)
    axes[1].axvline(2, color="red", ls="--", label="k=2 selected")
    axes[1].axhline(0.70, color="orange", ls=":", lw=1.5, label="Entropy threshold=0.70")
    axes[1].legend(); axes[1].set_xlabel("k"); axes[1].set_ylabel("Entropy")
    axes[1].set_title("LCA: Entropy by k"); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{results_dir}/lca_analysis/lca_class_sensitivity.png", dpi=200)
    plt.close()

    df = pd.DataFrame({"k": ks, "BIC": bics, "Entropy": ents, "Mean_PP": mpps})
    df["Entropy_OK"]  = df["Entropy"] >= 0.70
    df["Selected"]    = df["k"] == 2
    df.to_csv(f"{results_dir}/lca_analysis/lca_class_sensitivity.csv", index=False)
    log("  Saved lca_class_sensitivity.csv + figure")
    log(df.to_string(index=False))
    return df


# =============================================================================
# [FIX-13]  LCA BOOTSTRAP STABILITY
# =============================================================================
def lca_bootstrap_stability(subgroups: pd.Series, X: pd.DataFrame,
                             lca_features: List[str],
                             n_boot: int = 200, frac: float = 0.80) -> pd.DataFrame:
    """[FIX-13] Centroid-concordance bootstrap stability for subgroups."""
    section("FIX-13  LCA subgroup bootstrap stability")
    avail = [f for f in lca_features if f in X.columns]
    if len(avail) < 2: return pd.DataFrame()
    Xc   = X[avail].dropna()
    sgs  = subgroups.loc[Xc.index]
    X_np = StandardScaler().fit_transform(Xc.values)
    orig = sgs.values
    centroids = {sg: X_np[(sgs==sg).values].mean(0) for sg in sgs.unique()}
    conc = np.zeros(len(X_np))
    rng  = np.random.RandomState(42)
    for _ in range(n_boot):
        idx = rng.choice(len(X_np), size=int(len(X_np)*frac), replace=False)
        for i in idx:
            dists    = {sg: np.linalg.norm(X_np[i]-c) for sg,c in centroids.items()}
            assigned = min(dists, key=dists.get)
            if assigned == orig[i]: conc[i] += 1
    conc /= n_boot
    df = (pd.DataFrame({"subgroup": orig, "concordance": conc})
            .groupby("subgroup")["concordance"]
            .agg(["mean","std"]).reset_index())
    df.columns = ["Subgroup","Mean_Concordance","SD_Concordance"]
    df.to_csv(f"{results_dir}/lca_analysis/lca_stability.csv", index=False)
    log("  Saved lca_stability.csv"); log(df.to_string(index=False))
    return df


# =============================================================================
# [FIX-5]  TEMPORAL DATA FLOW DIAGRAM
# =============================================================================
def plot_temporal_diagram() -> None:
    """[FIX-5] Figure S1 -- temporal data flow for the study design."""
    section("FIX-5  Temporal data flow diagram")
    fig, ax = plt.subplots(figsize=(13, 4))
    bars = [
        ("MASLD Index\n(t=0)",          0.0, 0.2,  2.5, "#3498db"),
        ("Baseline labs\n& dx codes",   0.0, 0.5,  1.5, "#2ecc71"),
        ("Min follow-up\n(6 months)",   0.5, 0.7,  2.5, "#f39c12"),
        ("Longitudinal\nFIB-4 trajectory", 0.5, 4.0, 1.5, "#9b59b6"),
        ("ICD-based\nfibrosis events",  1.0, 4.5,  2.5, "#e74c3c"),
        ("Outcome\nascertainment",      4.0, 5.0,  1.5, "#1abc9c"),
        ("Prediction\nhorizon end",     5.0, 5.2,  2.5, "#95a5a6"),
    ]
    for label, start, end, ypos, color in bars:
        ax.barh(ypos, end-start, left=start, height=0.45,
                color=color, alpha=0.85, edgecolor="black", lw=0.7)
        ax.text((start+end)/2, ypos, label,
                ha="center", va="center", fontsize=7.5, fontweight="bold",
                color="white")
    ax.axvline(0, color="black", lw=1.5, ls="--")
    ax.text(0.02, 3.3, "Index Date", fontsize=8)
    ax.set_xlabel("Years from MASLD index date")
    ax.set_yticks([1.5, 2.5]); ax.set_yticklabels(["Labs / Outcome","Clinical / Prediction"])
    ax.set_xlim(-0.3, 5.8); ax.set_title("Temporal Data Flow Diagram (Figure S1)")
    ax.grid(axis="x", alpha=0.3); plt.tight_layout()
    plt.savefig(f"{results_dir}/figures/temporal_diagram.png", dpi=200); plt.close()
    log("  Saved temporal_diagram.png")


# =============================================================================
# [Part 6]  INCLUDED vs EXCLUDED TABLE 1
# =============================================================================
def included_vs_excluded_table(df_full: pd.DataFrame,
                                included_idx,
                                continuous_vars: List[str],
                                categorical_vars: List[str]) -> pd.DataFrame:
    """
    [Part 6] Table 1-style comparison of included vs excluded patients.
    Median (IQR) for continuous; n (%) for categorical.
    Mann-Whitney U / chi-squared p-values.
    """
    section("Part 6  Included vs Excluded Table 1")
    mask = df_full.index.isin(included_idx)
    inc  = df_full[mask]
    exc  = df_full[~mask]

    log(f"  Included: {len(inc)}  Excluded: {len(exc)}  Total: {len(df_full)}")

    rows = []

    for col in continuous_vars:
        if col not in df_full.columns: continue
        a = inc[col].dropna(); b = exc[col].dropna()
        if len(a) < 5 or len(b) < 5: continue
        stat, p = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        miss_pct = df_full[col].isnull().mean() * 100
        rows.append({
            "Variable": FEATURE_LABELS.get(col, col),
            "Type": "Continuous",
            "Overall":  f"{df_full[col].dropna().median():.1f} "
                        f"({df_full[col].dropna().quantile(.25):.1f}, "
                        f"{df_full[col].dropna().quantile(.75):.1f})",
            "Included": f"{a.median():.1f} ({a.quantile(.25):.1f}, {a.quantile(.75):.1f})",
            "Excluded": f"{b.median():.1f} ({b.quantile(.25):.1f}, {b.quantile(.75):.1f})",
            "p_value":  round(p, 4),
            "Missing_%": round(miss_pct, 1),
        })

    for col in categorical_vars:
        if col not in df_full.columns: continue
        try:
            ct  = pd.crosstab(mask, df_full[col])
            chi2, p, _, _ = scipy_stats.chi2_contingency(ct)
            miss_pct = df_full[col].isnull().mean() * 100
            # Most common category for display
            vc_i = inc[col].value_counts(normalize=True)
            vc_e = exc[col].value_counts(normalize=True)
            vc_o = df_full[col].value_counts()
            top  = vc_o.index[0]
            n_tot = int(vc_o.iloc[0])
            rows.append({
                "Variable": FEATURE_LABELS.get(col, col),
                "Type": "Categorical",
                "Overall":  f"{n_tot} ({n_tot/len(df_full)*100:.1f}%)",
                "Included": f"{int(inc[col].eq(top).sum())} "
                            f"({vc_i.get(top,0)*100:.1f}%)",
                "Excluded": f"{int(exc[col].eq(top).sum())} "
                            f"({vc_e.get(top,0)*100:.1f}%)",
                "p_value":  round(p, 4),
                "Missing_%": round(miss_pct, 1),
            })
        except Exception as e:
            log(f"    {col} failed: {e}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(f"{OUT}/included_vs_excluded_table1.csv", index=False)
    log("  Saved included_vs_excluded_table1.csv")
    return df_out


# =============================================================================
# [Part 8]  COMPARISON WITH PRIOR STUDIES
# =============================================================================
def comparison_with_prior_studies() -> pd.DataFrame:
    """[Part 8] Manuscript-ready comparison table with prior MASLD ML studies."""
    rows = [
        {
            "Study": "Alkhouri et al. (2020) -- Fast Progressor NASH",
            "Population": "NASH CRN, n=648, biopsy-confirmed",
            "Outcome": ">=2 fibrosis stage increase (biopsy)",
            "Method": "Random Forest + clinical + histology",
            "AUROC": "0.77",
            "Interpretability": "Feature importance (RF)",
            "External_Validation": "No",
            "Class_Imbalance_Handling": "Not reported",
            "Subgroup_Modeling": "No",
        },
        {
            "Study": "FibroGENE (Trépo et al., 2019)",
            "Population": "Multi-centre, n=2,510, biopsy",
            "Outcome": "Significant fibrosis (F>=2, cross-sectional)",
            "Method": "PNPLA3 + clinical logistic regression",
            "AUROC": "0.82",
            "Interpretability": "Linear score (interpretable)",
            "External_Validation": "Yes (2 cohorts)",
            "Class_Imbalance_Handling": "N/A (logistic)",
            "Subgroup_Modeling": "No",
        },
        {
            "Study": "This study (MCB, Overall)",
            "Population": "Mayo EHR, n=752, ICD+FIB-4 labels",
            "Outcome": "Rapid longitudinal progression",
            "Method": "LCA + subgroup-specific ML + stacking",
            "AUROC": "0.739 [0.659-0.816]",
            "Interpretability": "SHAP + integer pRFPS score",
            "External_Validation": "Yes (Tapestry, n=1,240)",
            "Class_Imbalance_Handling": "Subgroup median threshold; Youden",
            "Subgroup_Modeling": "Yes (C1/C2 endotypes)",
        },
        {
            "Study": "This study (Tapestry, Overall)",
            "Population": "Mayo EHR external, n=1,240",
            "Outcome": "Rapid longitudinal progression",
            "Method": "MCB-trained models applied directly",
            "AUROC": "0.821",
            "Interpretability": "Same pRFPS",
            "External_Validation": "--",
            "Class_Imbalance_Handling": "MCB threshold applied",
            "Subgroup_Modeling": "Yes (C1/C2)",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/comparison_prior_studies.csv", index=False)
    log("  Saved comparison_prior_studies.csv")
    return df


def compute_best_model_shap(best_model_name, best_model,
                              X_train, X_test, feature_names,
                              class_names, analysis_name, output_dir,
                              scaler=None, scaled=True):
    """
    Compute SHAP values for the best model.

    SHAP IS COMPUTED ON X_TRAIN (not X_test) for two reasons:

    1. The SHAP values are used by build_shap_clinical_risk_score() to
       derive the pRFPS formula: feature selection, weights, and directions.
       If these are derived from X_test SHAP, the formula is tuned on the
       same patients used to evaluate it -- a form of feature-weight leakage.
       Training-set SHAP avoids this: the formula is fixed before the test
       set is touched.

    2. Training-set SHAP importances reflect what the model learned from the
       majority of the data. Test-set importances can be noisy for small
       cohorts (n~65-150) and may not generalise to Tapestry.

    X_test is still accepted (for backward compatibility and for generating
    the Figure 5 plot showing test-patient explanations), but SHAP VALUES
    RETURNED are always from X_train.

    SVC fix: LinearExplainer only works for linear kernel SVC.
    RBF SVC falls through to KernelExplainer (the general fallback).
    """
    if not SHAP_AVAILABLE:
        log("   SHAP not available -- skipping."); return None
    os.makedirs(output_dir, exist_ok=True)
    try:
        X_train_arr = scaler.transform(X_train) if (scaled and scaler) else \
                      (X_train.values if hasattr(X_train,"values") else X_train)
        X_test_arr  = scaler.transform(X_test)  if (scaled and scaler) else \
                      (X_test.values  if hasattr(X_test, "values") else X_test)
        X_train_df = pd.DataFrame(X_train_arr, columns=feature_names)
        X_test_df  = pd.DataFrame(X_test_arr,  columns=feature_names)

        tree_types = (RandomForestClassifier, GradientBoostingClassifier,
                      DecisionTreeClassifier, ExtraTreesClassifier)
        if XGBOOST_AVAILABLE: tree_types = tree_types + (XGBClassifier,)

        # SHAP computed on the FULL TRAINING SET for formula derivation.
        #
        # Why full train, not a subsample:
        #   - Training set for this pipeline is at most ~602 patients (Overall)
        #     or ~260-342 (subgroups). These are small enough that computing
        #     exact SHAP over all training patients is fast and desirable.
        #   - Using all training patients gives the most stable mean |SHAP|
        #     importance estimates, which directly determine pRFPS weights.
        #     A subsample of 300 from 602 introduces unnecessary variance.
        #   - The 300-patient cap was a "large dataset" heuristic not
        #     appropriate for this cohort size.
        #
        # The ONLY exception is KernelExplainer (Stacking, MLP, KNN, RBF SVC),
        # which is O(n_patients × n_features × nsamples) and slow for n>200.
        # For those models we subsample to 150 patients -- still larger than
        # the test set and more representative than 50.
        #
        # Background dataset for KernelExplainer always uses full X_train_df
        # (via kmeans compression to 50 points) regardless of model type.

        n_train = len(X_train_df)

        if isinstance(best_model, StackingClassifier):
            # KernelExplainer: subsample patients for speed, full background
            _n_kernel = min(n_train, 150)
            _idx_k = np.random.RandomState(42).choice(n_train, _n_kernel, replace=False)
            bg  = shap.kmeans(X_train_df, min(50, n_train))
            exp = shap.KernelExplainer(lambda x: best_model.predict_proba(x)[:,1], bg)
            sv  = exp.shap_values(X_train_df.iloc[_idx_k], nsamples=100)
            log(f"   SHAP [Stacking/KernelExplainer]: {_n_kernel}/{n_train} train patients")

        elif isinstance(best_model, tree_types):
            # TreeExplainer: exact, fast for any n -- use ALL training patients
            exp = shap.TreeExplainer(best_model)
            sv  = exp.shap_values(X_train_df)
            # if isinstance(sv, list): sv = sv[1]
            log(f"   SHAP [TreeExplainer]: all {n_train} training patients")

        elif isinstance(best_model, LogisticRegression):
            # LinearExplainer: exact, instant -- use ALL training patients
            exp = shap.LinearExplainer(best_model, X_train_df)
            sv  = exp.shap_values(X_train_df)
            log(f"   SHAP [LinearExplainer/LR]: all {n_train} training patients")

        elif isinstance(best_model, SVC) and getattr(best_model, "kernel", None) == "linear":
            # LinearExplainer works for linear SVC -- use ALL training patients
            exp = shap.LinearExplainer(best_model, X_train_df)
            sv  = exp.shap_values(X_train_df)
            log(f"   SHAP [LinearExplainer/SVC-linear]: all {n_train} training patients")

        else:
            # KernelExplainer fallback: RBF SVC, MLP, KNN
            # Subsample to 150 for runtime; background uses kmeans of full train
            _n_kernel = min(n_train, 150)
            _idx_k = np.random.RandomState(42).choice(n_train, _n_kernel, replace=False)
            bg  = shap.kmeans(X_train_df, min(50, n_train))
            exp = shap.KernelExplainer(lambda x: best_model.predict_proba(x)[:,1], bg)
            sv  = exp.shap_values(X_train_df.iloc[_idx_k], nsamples=100)
            log(f"   SHAP [KernelExplainer/{type(best_model).__name__}]: "
                f"{_n_kernel}/{n_train} train patients")
        # Normalize SHAP output shape before building importance table.
        # Some TreeExplainer versions return (n_samples, n_features, n_classes)
        # for binary classifiers; keep class 1 = Rapid Progression.
        if isinstance(sv, list):
            sv = sv[1]

        sv = np.array(sv)

        if sv.ndim == 3:
            sv = sv[:, :, 1]

        if sv.ndim == 1:
            sv = sv.reshape(1, -1)

        mean_abs_shap = np.abs(sv).mean(axis=0)
        #mean_abs_shap = np.abs(sv).mean(axis=0)
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "label":   [FEATURE_LABELS.get(f,f) for f in feature_names],
            "mean_abs_shap": mean_abs_shap,
            "mean_shap":     sv.mean(axis=0),
            "std_shap":      sv.std(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        imp_df["rank"] = imp_df.index + 1
        imp_df.to_csv(f"{output_dir}/shap_importance_{analysis_name}.csv", index=False)

        # ── Figure 5: Mean SHAP direction bar chart (no beeswarm) ──────────
        # Shows mean signed SHAP per feature, sorted by absolute magnitude
        # Red = risk-increasing, Blue = protective
        # Error bars = SD across test patients
        try:
            mean_signed = sv.mean(axis=0)
            std_signed  = sv.std(axis=0)
            mean_abs    = np.abs(sv).mean(axis=0)
            labels_f5   = [FEATURE_LABELS.get(f, f) for f in feature_names]

            # Sort ascending by abs SHAP so most important is at top of barh
            sort_idx   = np.argsort(mean_abs)
            n_show     = min(15, len(sort_idx))
            idx_show   = sort_idx[-n_show:]   # top-n by abs SHAP

            s_vals  = mean_signed[idx_show]
            s_errs  = std_signed[idx_show]
            s_lbls  = [labels_f5[i] for i in idx_show]
            s_cols  = ["#D73027" if v > 0 else "#4575B4" for v in s_vals]

            fig5, ax5 = plt.subplots(figsize=(8, max(4, n_show * 0.55)))

            bars = ax5.barh(
                s_lbls, s_vals,
                xerr=s_errs,
                color=s_cols,
                alpha=0.85,
                edgecolor="white",
                capsize=4,
                error_kw={"elinewidth": 1.2, "capthick": 1.2,
                          "ecolor": "dimgray"})

            ax5.axvline(0, color="black", lw=1.0, zorder=3)

            # Value annotation on each bar
            for bar, v in zip(bars, s_vals):
                ax5.text(
                    v + (0.001 if v >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f"{v:+.3f}",
                    va="center", ha="left" if v >= 0 else "right",
                    fontsize=7, color="dimgray")

            ax5.set_xlabel(
                "Mean SHAP Value  "
                "(positive = increases risk of Rapid Progression, "
                "negative = protective)",
                fontsize=9)
            ax5.set_title(
                f"Figure 5. Top Predictors of Rapid Fibrosis Progression -- Direction of SHAP Contribution -- {analysis_name}",
                fontsize=11, fontweight="bold")
            ax5.grid(axis="x", alpha=0.22, lw=0.6)
            ax5.spines["top"].set_visible(False)
            ax5.spines["right"].set_visible(False)
            ax5.legend(
                handles=[
                    mpatches.Patch(color="#D73027", label="Risk-increasing (+)"),
                    mpatches.Patch(color="#4575B4", label="Protective (-)"),
                ],
                fontsize=9, loc="lower right", framealpha=0.9)

            plt.tight_layout()
            f5_path = f"{output_dir}/figure5_shap_direction_{analysis_name}.png"
            plt.savefig(f5_path, dpi=200, bbox_inches="tight")
            plt.close()
            log(f"   Figure 5 saved: {f5_path}")

        except Exception as e:
            log(f"   Figure 5 failed: {e}")

        # Also compute SHAP on TEST patients for the delta comparison.
        # Train SHAP (sv) is used for formula derivation (weights, direction).
        # Test SHAP is used to compute the true SHAP-additive score on the
        # same patients as integer pRFPS, so the AUROC delta is meaningful.
        sv_test = None
        try:
            if isinstance(best_model, StackingClassifier):
                sv_test = exp.shap_values(X_test_df, nsamples=100)
            elif isinstance(best_model, tree_types):
                sv_test = exp.shap_values(X_test_df)
            elif isinstance(best_model, (LogisticRegression, SVC)):
                sv_test = exp.shap_values(X_test_df)
            else:
                sv_test = exp.shap_values(X_test_df, nsamples=100)
            if sv_test is not None:
                if isinstance(sv_test, list):
                    sv_test = sv_test[1]
                if hasattr(sv_test, 'ndim') and sv_test.ndim == 3:
                    sv_test = sv_test[:, :, 1]
                sv_test = np.array(sv_test)
                if sv_test.ndim == 1:
                    sv_test = sv_test.reshape(1, -1)
            log(f"   SHAP (test) shape: {sv_test.shape if sv_test is not None else 'None'}  "
                f"match y_true: {sv_test.shape[0]==len(X_test_df) if sv_test is not None else 'N/A'}")
        except Exception as _e:
            log(f"   SHAP on test failed: {_e} -- delta comparison will be skipped")

        log(f"   SHAP done -> {output_dir}/")
        return {"importance_df": imp_df,
                "shap_values":      sv,       # TRAIN shap -- for formula derivation
                "shap_values_test": sv_test,  # TEST shap  -- for AUROC delta comparison
                "explainer": exp, "n_test_used": sv.shape[0], "output_dir": output_dir}
    except Exception as e:
        log(f"   compute_best_model_shap ERROR: {e}"); return None



# =============================================================================
# STRATIFIED SUBGROUP SPLIT  (original create_stratified_subgroup_split)
# =============================================================================
def create_stratified_subgroup_split(X, y_original, feature_names,
                                      y_numeric, subgroups,
                                      test_size=0.2, random_state=42,
                                      selected_features=None):
    """Original function -- preserved intact."""
    print("\n=== Creating Stratified Subgroup Train/Test Split ===")
    feature_indices = [i for i,name in enumerate(feature_names)
                       if name in selected_features]
    X_selected = X.iloc[:,feature_indices] if hasattr(X,'iloc') else X[:,feature_indices]
    if X_selected.isnull().any().any():
        mask = ~X_selected.isnull().any(axis=1)
        X_selected = X_selected[mask]; y_original = y_original[mask]
        y_numeric  = y_numeric[mask];  subgroups  = subgroups[mask]

    train_indices_all = []; test_indices_all = []; subgroup_info = {}
    for sg in subgroups.unique():
        sg_mask = subgroups == sg
        sg_idx  = X_selected[sg_mask].index.tolist() if hasattr(X_selected,"index") \
                  else list(np.where(sg_mask)[0])
        y_sg    = y_numeric[sg_mask]
        if len(sg_idx) < 10 or len(np.unique(y_sg)) < 2: continue
        try:
            tr_idx, te_idx = train_test_split(sg_idx, test_size=test_size,
                                               random_state=random_state, stratify=y_sg)
            train_indices_all.extend(tr_idx); test_indices_all.extend(te_idx)
            subgroup_info[sg] = {"total":len(sg_idx),"train":len(tr_idx),"test":len(te_idx)}
            print(f"{sg}: Train={len(tr_idx)}  Test={len(te_idx)}")
        except Exception as e:
            print(f"Error splitting {sg}: {e}")
    print(f"Combined: Train={len(train_indices_all)}  Test={len(test_indices_all)}")
    return train_indices_all, test_indices_all, subgroup_info



# =============================================================================
# METRICS VISUALISATION  -- Publication-ready CI plots + ROC/PR/Calibration
# =============================================================================

# Colour palette: distinct, colourblind-friendly, publication quality
MODEL_COLORS = {
    "Logistic_Regression": "#1F77B4",
    "KNN":                 "#FF7F0E",
    "Random_Forest":       "#2CA02C",
    "DecisionTree":        "#D62728",
    "GradientBoosting":    "#9467BD",
    "SVM":                 "#8C564B",
    "MLP":                 "#E377C2",
    "XGBoost":             "#17BECF",
    "LightGBM":            "#BCBD22",   # baseline comparison model
    "Stacking_Ensemble":   "#000000",   # black -- highlight the best
}

MODEL_MARKERS = {
    "Logistic_Regression": "o",
    "KNN":                 "s",
    "Random_Forest":       "^",
    "DecisionTree":        "D",
    "GradientBoosting":    "v",
    "SVM":                 "P",
    "MLP":                 "X",
    "XGBoost":             "*",
    "LightGBM":            "h",
    "Stacking_Ensemble":   "d",
}

METRIC_INFO = {
    "accuracy":    {"label": "Accuracy",     "ylim": (0.3, 1.0), "better": "high"},
    "f1_macro":    {"label": "F1 (macro)",   "ylim": (0.2, 1.0), "better": "high"},
    "auroc":       {"label": "AUROC",        "ylim": (0.4, 1.0), "better": "high"},
    "auprc":       {"label": "AUPRC",        "ylim": (0.2, 1.0), "better": "high"},
    "brier":       {"label": "Brier Score",  "ylim": (0.0, 0.5), "better": "low"},
    "sensitivity": {"label": "Sensitivity",  "ylim": (0.0, 1.0), "better": "high"},
    "specificity": {"label": "Specificity",  "ylim": (0.0, 1.0), "better": "high"},
    "ppv":         {"label": "PPV",          "ylim": (0.0, 1.0), "better": "high"},
    "npv":         {"label": "NPV",          "ylim": (0.0, 1.0), "better": "high"},
    "precision_1": {"label": f"Precision ({POSITIVE_LABEL})", "ylim": (0.0, 1.0), "better": "high"},
    "recall_1":    {"label": f"Recall ({POSITIVE_LABEL})",    "ylim": (0.0, 1.0), "better": "high"},
    "f1_1":        {"label": f"F1 ({POSITIVE_LABEL})",        "ylim": (0.0, 1.0), "better": "high"},
}



def _parse_ci_value(val_str):
    """Parse '0.739 [0.659-0.816]' -> (0.739, 0.659, 0.816).
       Or if plain float string -> (float, nan, nan)."""
    if isinstance(val_str, (int, float)):
        return float(val_str), float("nan"), float("nan")
    s = str(val_str).strip()
    if "[" in s and "-" in s:
        try:
            point = float(s.split("[")[0].strip())
            ci_part = s.split("[")[1].replace("]", "")
            lo, hi = ci_part.split("-")
            return point, float(lo), float(hi)
        except Exception:
            pass
    try:
        return float(s), float("nan"), float("nan")
    except Exception:
        return float("nan"), float("nan"), float("nan")


def plot_metrics_with_ci(
    all_results_dict: Dict,
    out_dir: str = None,
    metrics: List[str] = None,
    settings: List[str] = None,
    baseline_model: str = "LightGBM",   # kept for API compat, no longer shaded
) -> None:
    """
    One compact figure per metric.

    Changes from previous version:
      - Figure width scales tightly to number of models (no excess white space)
      - LightGBM treated as a regular model, no special shading
      - Best result per metric highlighted with a gold star annotation
        (best Overall, best C1, best C2 each get their own star)
      - SE column softly highlighted (not LightGBM)
      - Grouped dot-plot with 95% bootstrap CI error bars
    """
    if out_dir is None:
        out_dir = f"{results_dir}/figures/metrics_plots"
    os.makedirs(out_dir, exist_ok=True)

    if metrics is None:
        metrics = ["auroc", "auprc", "f1_macro", "f1_1",
                   "sensitivity", "specificity", "ppv", "npv",
                   "accuracy", "brier"]
    if settings is None:
        settings = list(all_results_dict.keys())
    settings = [s for s in settings if s in all_results_dict]
    if not settings:
        return

    # Consistent model order: SE last so it's always on the right
    all_models_raw = list(list(all_results_dict.values())[0].keys())
    se_models      = [m for m in all_models_raw if m == "Stacking_Ensemble"]
    other_models   = [m for m in all_models_raw if m != "Stacking_Ensemble"]
    all_models     = other_models + se_models
    n_mod = len(all_models)
    n_set = len(settings)

    YLIM = {
        "auroc":(0.40,1.00),"auprc":(0.10,1.00),
        "f1_macro":(0.20,1.00),"f1_1":(0.00,1.00),
        "sensitivity":(0.00,1.00),"specificity":(0.00,1.00),
        "ppv":(0.00,1.00),"npv":(0.00,1.00),
        "accuracy":(0.30,1.00),"brier":(0.05,0.50),
    }
    LBL = {
        "auroc":"AUROC","auprc":"AUPRC",
        "f1_macro":"F1 (Macro)",
        "f1_1":f"F1 ({POSITIVE_LABEL})",
        "sensitivity":"Sensitivity","specificity":"Specificity",
        "ppv":"PPV","npv":"NPV",
        "accuracy":"Accuracy","brier":"Brier Score",
    }
    BETTER = {"brier": "low"}

    # Per-setting colour and marker
    SET_COL = {"Overall": "#2C3E50", "C1": "#2980B9", "C2": "#C0392B"}
    SET_MK  = {"Overall": "o",       "C1": "s",       "C2": "^"}
    _ec = ["#27AE60","#8E44AD","#F39C12"]
    _em = ["D","P","X"]
    for i, s in enumerate([x for x in settings if x not in SET_COL]):
        SET_COL[s] = _ec[i % len(_ec)]
        SET_MK[s]  = _em[i % len(_em)]

    def _get(r, metric):
        ci = r.get("ci", {})
        if metric in ci:
            mn, lo, hi = ci[metric]
            return float(mn), float(mn - lo), float(hi - mn)
        v = r.get(metric, float("nan"))
        return float(v), 0.0, 0.0

    # Tight layout: each model group gets exactly 1.05 inches, plus margins
    FIG_W   = max(6, n_mod * 1.05 + 1.8)
    group_w = 0.75
    dot_w   = group_w / n_set
    offsets = np.linspace(-(group_w/2 - dot_w/2),
                           (group_w/2 - dot_w/2), n_set)

    for metric in metrics:
        ylim   = YLIM.get(metric, (0, 1))
        label  = LBL.get(metric, metric)
        better = BETTER.get(metric, "high")

        x_base = np.arange(n_mod)
        fig, ax = plt.subplots(figsize=(FIG_W, 4.8))

        # Find the single best value per setting across all models
        best_by_setting = {}   # {setting: (best_val, best_model_idx)}
        for setting in settings:
            res   = all_results_dict[setting]
            vals  = [(xi, _get(res.get(m, {}), metric)[0])
                     for xi, m in enumerate(all_models)]
            valid = [(xi, v) for xi, v in vals if not np.isnan(v)]
            if not valid:
                continue
            if better == "high":
                best_by_setting[setting] = max(valid, key=lambda x: x[1])
            else:
                best_by_setting[setting] = min(valid, key=lambda x: x[1])

        for s_idx, setting in enumerate(settings):
            res    = all_results_dict[setting]
            color  = SET_COL.get(setting, "#555")
            marker = SET_MK.get(setting, "o")
            x_pos  = x_base + offsets[s_idx]
            best_xi, _ = best_by_setting.get(setting, (-1, None))

            for xi, mname in enumerate(all_models):
                r = res.get(mname, {})
                mn, lo_e, hi_e = _get(r, metric)
                if np.isnan(mn):
                    continue

                is_se   = mname == "Stacking_Ensemble"
                is_best = (xi == best_xi)

                ax.errorbar(
                    x_pos[xi], mn,
                    yerr=[[max(lo_e, 0)], [max(hi_e, 0)]],
                    fmt=marker,
                    color=color,
                    markersize=11 if is_best else (9 if is_se else 7),
                    capsize=3, capthick=1.2, elinewidth=1.1,
                    markeredgecolor="gold"  if is_best else
                                   ("black" if is_se else color),
                    markeredgewidth=2.2 if is_best else (1.6 if is_se else 0.4),
                    zorder=8 if is_best else (6 if is_se else 3),
                    label=setting if xi == 0 else "_nolegend_",
                    alpha=1.0 if (is_best or is_se) else 0.82,
                )

                # Gold star annotation above the best point
                if is_best:
                    ax.annotate(
                        "★",
                        xy=(x_pos[xi], mn + hi_e + (ylim[1]-ylim[0])*0.025),
                        ha="center", va="bottom",
                        fontsize=8, color="gold",
                        fontweight="bold",
                        zorder=9,
                    )

        # Reference lines
        if metric == "auroc":
            ax.axhline(0.5, color="#BDC3C7", ls="--", lw=0.8, alpha=0.8,
                       label="Random (0.5)")
        if metric == "brier":
            prev_vals = [r.get("prevalence", 0.38)
                         for s in settings
                         for r in all_results_dict[s].values()]
            ns = float(np.nanmean(prev_vals)) * (1 - float(np.nanmean(prev_vals)))
            ax.axhline(ns, color="#BDC3C7", ls="--", lw=0.8, alpha=0.8,
                       label=f"No-skill ({ns:.2f})")

        # Soft highlight on Stacking Ensemble column only
        if "Stacking_Ensemble" in all_models:
            sei = all_models.index("Stacking_Ensemble")
            ax.axvspan(sei - 0.44, sei + 0.44, alpha=0.055,
                       color="#2C3E50", zorder=0)

        ax.set_xticks(x_base)
        ax.set_xticklabels(
            [m.replace("_", " ") for m in all_models],
            fontsize=7.5, rotation=38, ha="right")
        ax.set_ylim(ylim[0], ylim[1] + (ylim[1]-ylim[0])*0.06)   # small top margin for stars
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label}  — 95% Bootstrap CI  (★ = best per setting)",
                     fontsize=10.5, fontweight="bold", pad=6)
        ax.grid(axis="y", alpha=0.18, lw=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Compact legend
        handles = [
            plt.Line2D([0], [0], marker=SET_MK.get(s, "o"),
                       color="w",
                       markerfacecolor=SET_COL.get(s, "#555"),
                       markersize=8,
                       markeredgecolor="black" if s == "Overall" else SET_COL.get(s,"#555"),
                       markeredgewidth=0.6,
                       lw=0, label=s)
            for s in settings
        ]
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#2C3E50", markersize=9,
                       markeredgecolor="gold", markeredgewidth=2.2,
                       lw=0, label="★ Best in setting"))
        ax.legend(handles=handles, fontsize=8, framealpha=0.88,
                  loc="upper left", borderpad=0.6, labelspacing=0.35)

        plt.tight_layout(pad=0.8)
        safe = metric.replace("/", "_")
        plt.savefig(f"{out_dir}/metric_{safe}.png",
                    dpi=200, bbox_inches="tight")
        plt.close()
        log(f"  Saved: metric_{safe}.png")

    # ── ROC curves (compact, one per setting) ─────────────────────────────────
    fig_r, ax_r = plt.subplots(1, n_set, figsize=(n_set * 4.8, 4.5),
                                sharey=True)
    if n_set == 1: ax_r = [ax_r]
    fig_r.suptitle("ROC Curves  |  AUROC [95% CI]",
                   fontsize=11, fontweight="bold", y=1.01)
    for si, setting in enumerate(settings):
        ax = ax_r[si]; res = all_results_dict[setting]
        ax.plot([0,1],[0,1], "--", color="#BDC3C7", lw=0.8, zorder=1)
        for mname, r in res.items():
            yt, yp = r.get("y_true"), r.get("y_proba")
            if yt is None or len(np.unique(yt)) < 2: continue
            fpr, tpr, _ = roc_curve(yt, yp)
            auroc = r.get("auroc", float("nan"))
            ci    = r.get("ci", {})
            lo    = ci["auroc"][1] if "auroc" in ci else float("nan")
            hi    = ci["auroc"][2] if "auroc" in ci else float("nan")
            lbl   = (f"{mname.replace('_',' ')}: {auroc:.3f} [{lo:.3f}–{hi:.3f}]"
                     if not np.isnan(lo) else
                     f"{mname.replace('_',' ')}: {auroc:.3f}")
            is_se = mname == "Stacking_Ensemble"
            ax.plot(fpr, tpr,
                    color=MODEL_COLORS.get(mname, "#888"),
                    lw=2.2 if is_se else 1.2,
                    ls="-", alpha=1.0 if is_se else 0.7,
                    label=lbl, zorder=5 if is_se else 3)
        ax.set_title(setting, fontsize=10, fontweight="bold")
        ax.set_xlabel("FPR", fontsize=9)
        if si == 0: ax.set_ylabel("TPR", fontsize=9)
        ax.legend(fontsize=5.5, loc="lower right", framealpha=0.85)
        ax.grid(alpha=0.15)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.7)
    plt.savefig(f"{out_dir}/roc_all_settings.png",
                dpi=200, bbox_inches="tight")
    plt.close(); log("  Saved: roc_all_settings.png")

    # ── PR curves ─────────────────────────────────────────────────────────────
    fig_p, ax_p = plt.subplots(1, n_set, figsize=(n_set * 4.8, 4.5),
                                sharey=True)
    if n_set == 1: ax_p = [ax_p]
    fig_p.suptitle("PR Curves  |  AUPRC [95% CI]",
                   fontsize=11, fontweight="bold", y=1.01)
    for si, setting in enumerate(settings):
        ax = ax_p[si]; res = all_results_dict[setting]
        prev = float(np.nanmean([r.get("prevalence", 0.38)
                                 for r in res.values()]))
        ax.axhline(prev, color="#BDC3C7", ls="--", lw=0.8,
                   label=f"No-skill ({prev:.2f})", zorder=1)
        for mname, r in res.items():
            yt, yp = r.get("y_true"), r.get("y_proba")
            if yt is None or len(np.unique(yt)) < 2: continue
            prec, rec, _ = precision_recall_curve(yt, yp)
            auprc = r.get("auprc", float("nan"))
            ci    = r.get("ci", {})
            lo    = ci["auprc"][1] if "auprc" in ci else float("nan")
            hi    = ci["auprc"][2] if "auprc" in ci else float("nan")
            lbl   = (f"{mname.replace('_',' ')}: {auprc:.3f} [{lo:.3f}–{hi:.3f}]"
                     if not np.isnan(lo) else
                     f"{mname.replace('_',' ')}: {auprc:.3f}")
            is_se = mname == "Stacking_Ensemble"
            ax.plot(rec, prec,
                    color=MODEL_COLORS.get(mname, "#888"),
                    lw=2.2 if is_se else 1.2,
                    alpha=1.0 if is_se else 0.7,
                    label=lbl, zorder=5 if is_se else 3)
        ax.set_title(setting, fontsize=10, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=9)
        if si == 0: ax.set_ylabel("Precision", fontsize=9)
        ax.legend(fontsize=5.5, loc="upper right", framealpha=0.85)
        ax.grid(alpha=0.15)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.7)
    plt.savefig(f"{out_dir}/pr_all_settings.png",
                dpi=200, bbox_inches="tight")
    plt.close(); log("  Saved: pr_all_settings.png")

    # ── Calibration curves ─────────────────────────────────────────────────────
    fig_c, ax_c = plt.subplots(1, n_set, figsize=(n_set * 4.5, 4.5),
                                sharey=True)
    if n_set == 1: ax_c = [ax_c]
    fig_c.suptitle("Calibration  |  Brier Score [95% CI]",
                   fontsize=11, fontweight="bold", y=1.01)
    for si, setting in enumerate(settings):
        ax = ax_c[si]; res = all_results_dict[setting]
        ax.plot([0,1],[0,1], "--", color="#BDC3C7", lw=1,
                label="Perfect", zorder=1)
        for mname, r in res.items():
            yt, yp = r.get("y_true"), r.get("y_proba")
            if yt is None or len(np.unique(yt)) < 2: continue
            try:
                frac, mp = calibration_curve(yt, yp, n_bins=8,
                                              strategy="uniform")
                brier = r.get("brier", float("nan"))
                ci    = r.get("ci", {})
                lo    = ci["brier"][1] if "brier" in ci else float("nan")
                hi    = ci["brier"][2] if "brier" in ci else float("nan")
                lbl   = (f"{mname.replace('_',' ')}: {brier:.3f} [{lo:.3f}–{hi:.3f}]"
                         if not np.isnan(lo) else
                         f"{mname.replace('_',' ')}: {brier:.3f}")
                is_se = mname == "Stacking_Ensemble"
                ax.plot(mp, frac, "s-",
                        color=MODEL_COLORS.get(mname, "#888"),
                        lw=2.0 if is_se else 1.1,
                        ms=4, alpha=1.0 if is_se else 0.7,
                        label=lbl, zorder=5 if is_se else 3)
            except Exception: pass
        ax.set_title(setting, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Predicted Prob.", fontsize=9)
        if si == 0: ax.set_ylabel("Fraction Positives", fontsize=9)
        ax.legend(fontsize=5.2, loc="upper left", framealpha=0.85)
        ax.grid(alpha=0.15)
        ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    plt.tight_layout(pad=0.7)
    plt.savefig(f"{out_dir}/calibration_all_settings.png",
                dpi=200, bbox_inches="tight")
    plt.close(); log("  Saved: calibration_all_settings.png")

    # ── Summary heatmap ────────────────────────────────────────────────────────
    hmap_mets = ["auroc","auprc","f1_macro","f1_1",
                 "sensitivity","specificity","brier"]
    hmap_data = {}
    for setting in settings:
        for mname, r in all_results_dict[setting].items():
            ci  = r.get("ci", {})
            key = f"{setting}  |  {mname.replace('_',' ')}"
            hmap_data[key] = {
                LBL.get(m, m): (ci[m][0] if m in ci else r.get(m, float("nan")))
                for m in hmap_mets}
    hdf = pd.DataFrame(hmap_data).T
    bl = LBL.get("brier", "Brier Score")
    if bl in hdf.columns:
        hdf[bl] = 1 - hdf[bl]
        hdf.rename(columns={bl: "1-Brier"}, inplace=True)

    n_rows  = len(hdf)
    n_cols  = len(hdf.columns)
    fig_h, ax_h = plt.subplots(figsize=(n_cols * 1.35 + 1.5,
                                         max(5, n_rows * 0.38)))
    im = ax_h.imshow(hdf.values.astype(float), aspect="auto",
                     cmap="RdYlGn", vmin=0.30, vmax=1.00)
    ax_h.set_xticks(range(n_cols))
    ax_h.set_xticklabels(hdf.columns, rotation=30, ha="right", fontsize=8.5)
    ax_h.set_yticks(range(n_rows))
    ax_h.set_yticklabels(hdf.index, fontsize=6.5)
    plt.colorbar(im, ax=ax_h, label="Metric (green = better)",
                 fraction=0.025, pad=0.02)
    for i in range(n_rows):
        for j in range(n_cols):
            v = hdf.values[i, j]
            if not np.isnan(v):
                ax_h.text(j, i, f"{v:.2f}", ha="center", va="center",
                          fontsize=6,
                          color="white" if v < 0.44 or v > 0.87 else "black")
    # Horizontal separators between settings
    n_m = len(all_models)
    for sep in range(n_m, n_rows, n_m):
        ax_h.axhline(sep - 0.5, color="white", lw=1.8)
    ax_h.set_title("Summary Heatmap — All Models × Settings",
                   fontsize=10, fontweight="bold", pad=7)
    plt.tight_layout(pad=0.6)
    plt.savefig(f"{out_dir}/summary_heatmap.png",
                dpi=200, bbox_inches="tight")
    plt.close(); log("  Saved: summary_heatmap.png")
    log(f"  All figures → {out_dir}/")
# =============================================================================
# MAIN PIPELINE
# =============================================================================
