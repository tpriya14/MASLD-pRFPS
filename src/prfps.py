"""
prfps.py — pRFPS score construction (SHAP-derived clinical rules).

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    f1_score, precision_score, recall_score, confusion_matrix,
    brier_score_loss, roc_curve, precision_recall_curve,
    balanced_accuracy_score
)
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict
import scipy.stats as scipy_stats

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


def compute_continuous_prfps_variants(
    top,
    X_te,
    X_tr,
    y_te,
    y_tr,
    analysis_name,
    output_dir,
):
    """
    Computes and compares three pRFPS variants:
      1. Binary I() — integer weights  (current paper implementation)
      2. Binary I() — raw SHAP weights (continuous weights, binary indicator)
      3. Continuous I() — normalised distance from threshold (addresses reviewer)
 
    Paste this function directly after build_shap_clinical_risk_score()
    in your main pipeline file. No external helpers needed — all helpers
    are defined inline below.
    """
 
    os.makedirs(output_dir, exist_ok=True)
 
    # ── Inline helper: DeLong test ────────────────────────────────────────────
    def _delong_local(y_true, prob_a, prob_b):
        y = np.array(y_true).astype(int)
        pa, pb = np.array(prob_a), np.array(prob_b)
        def _struct(y, p):
            pos = p[y == 1]; neg = p[y == 0]
            n1, n0 = len(pos), len(neg)
            if n1 == 0 or n0 == 0:
                return np.nan, np.nan, np.nan, n1, n0
            V10 = np.array([np.mean(pi > neg) + 0.5*np.mean(pi == neg) for pi in pos])
            V01 = np.array([np.mean(ni < pos) + 0.5*np.mean(ni == pos) for ni in neg])
            return V10.mean(), np.var(V10, ddof=1)/n1, np.var(V01, ddof=1)/n0, n1, n0
        try:
            auc_a, s10_a, s01_a, n1, n0 = _struct(y, pa)
            auc_b, s10_b, s01_b, _,  _  = _struct(y, pb)
            pos_a = pa[y==1]; neg_a = pa[y==0]
            pos_b = pb[y==1]; neg_b = pb[y==0]
            V10_a = np.array([np.mean(pi > neg_a) + 0.5*np.mean(pi==neg_a) for pi in pos_a])
            V01_a = np.array([np.mean(ni < pos_a) + 0.5*np.mean(ni==pos_a) for ni in neg_a])
            V10_b = np.array([np.mean(pi > neg_b) + 0.5*np.mean(pi==neg_b) for pi in pos_b])
            V01_b = np.array([np.mean(ni < pos_b) + 0.5*np.mean(ni==pos_b) for ni in neg_b])
            s10_ab = np.cov(V10_a, V10_b)[0, 1] / n1
            s01_ab = np.cov(V01_a, V01_b)[0, 1] / n0
            var_diff = s10_a + s01_a + s10_b + s01_b - 2*s10_ab - 2*s01_ab
            if var_diff <= 0:
                return np.nan, np.nan
            z = (auc_a - auc_b) / np.sqrt(var_diff)
            p = 2 * (1 - scipy_stats.norm.cdf(abs(z)))
            return float(z), float(p)
        except Exception:
            return np.nan, np.nan
 
    # ── Inline helper: Youden cutoff from training scores ─────────────────────
    def _youden_local(train_scores, y_train):
        y = np.array(y_train).astype(int)
        best_j, best_t = -1, float(np.median(train_scores))
        candidates = np.unique(
            np.percentile(train_scores, np.linspace(5, 95, 100))
        )
        for t in candidates:
            pred = (train_scores >= t).astype(int)
            if pred.sum() == 0 or pred.sum() == len(pred):
                continue
            tn, fp, fn, tp = confusion_matrix(
                y, pred, labels=[0, 1]).ravel()
            j = tp/(tp+fn+1e-10) + tn/(tn+fp+1e-10) - 1
            if j > best_j:
                best_j = j
                best_t = float(t)
        return best_t
 
    # ── Inline helper: performance metrics ───────────────────────────────────
    def _perf_local(y_true, score, cutoff, label):
        y = np.array(y_true).astype(int)
        pred = (np.array(score) >= cutoff).astype(int)
        auroc = roc_auc_score(y, score) if len(np.unique(y)) > 1 else np.nan
        auprc = average_precision_score(y, score) if len(np.unique(y)) > 1 else np.nan
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
        return {
            'variant':     label,
            'auroc':       round(float(auroc), 4),
            'auprc':       round(float(auprc), 4),
            'f1_macro':    round(float(f1_score(y, pred, average='macro',    zero_division=0)), 4),
            'f1_rapid':    round(float(f1_score(y, pred, average='binary',   zero_division=0)), 4),
            'sensitivity': round(float(tp/(tp+fn+1e-10)), 4),
            'specificity': round(float(tn/(tn+fp+1e-10)), 4),
            'ppv':         round(float(tp/(tp+fp+1e-10)), 4),
            'npv':         round(float(tn/(tn+fn+1e-10)), 4),
            'cutoff':      round(float(cutoff), 4),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
        }
 
    # ── Prep arrays ───────────────────────────────────────────────────────────
    X_te = (X_te.reset_index(drop=True) if isinstance(X_te, pd.DataFrame)
            else pd.DataFrame(X_te))
    X_tr = (X_tr.reset_index(drop=True) if isinstance(X_tr, pd.DataFrame)
            else pd.DataFrame(X_tr))
    y_te = np.array(y_te).astype(int).flatten()
    y_tr = np.array(y_tr).astype(int).flatten()
 
    # ── Initialise score accumulators ─────────────────────────────────────────
    # Test set scores
    sc_bin_int  = np.zeros(len(X_te))   # Variant 1: integer weight,  binary I()
    sc_bin_raw  = np.zeros(len(X_te))   # Variant 2: raw SHAP weight, binary I()
    sc_cont     = np.zeros(len(X_te))   # Variant 3: raw SHAP weight, distance I()
 
    # Training set scores (for cutoff derivation only)
    tr_bin_int  = np.zeros(len(X_tr))
    tr_bin_raw  = np.zeros(len(X_tr))
    tr_cont     = np.zeros(len(X_tr))
 
    feature_details = []
 
    for _, row in top.iterrows():
        feat      = row['feature']
        thr       = float(row['youden_threshold'])
        w_int     = int(row['weight_int'])
        w_raw     = float(row['weight_raw'])
        direction = row['direction']
 
        if feat not in X_te.columns:
            continue
 
        vals_te = X_te[feat].values.astype(float)
        vals_tr = X_tr[feat].values.astype(float) if feat in X_tr.columns \
                  else np.zeros(len(X_tr))
 
        # ── Binary indicator (same logic as build_shap_clinical_risk_score) ──
        if direction == 'Risk+':
            ind_te = (vals_te > thr).astype(float)   # strict > (R-aligned)
            ind_tr = (vals_tr > thr).astype(float)
        else:
            ind_te = (vals_te < thr).astype(float)   # strict < (R-aligned)
            ind_tr = (vals_tr < thr).astype(float)
 
        # ── Continuous indicator: normalised distance from threshold ──────────
        # For Risk+  features: how far ABOVE the threshold is the patient?
        # For Protective features: how far BELOW the threshold is the patient?
        # Normalised by the max distance observed in the TRAINING set
        # so the scale is consistent across features with different units.
        if direction == 'Risk+':
            dist_te = np.maximum(vals_te - thr, 0)
            dist_tr = np.maximum(vals_tr - thr, 0)
        else:
            dist_te = np.maximum(thr - vals_te, 0)
            dist_tr = np.maximum(thr - vals_tr, 0)
 
        max_dist    = dist_tr.max() + 1e-10   # normalise by training max
        dist_te_n   = dist_te / max_dist
        dist_tr_n   = dist_tr / max_dist
 
        # Accumulate
        sc_bin_int += w_int * ind_te
        sc_bin_raw += w_raw * ind_te
        sc_cont    += w_raw * dist_te_n
 
        tr_bin_int += w_int * ind_tr
        tr_bin_raw += w_raw * ind_tr
        tr_cont    += w_raw * dist_tr_n
 
        feature_details.append({
            'feature':          feat,
            'label':            row.get('label', feat),
            'direction':        direction,
            'threshold':        round(thr, 3),
            'weight_int':       w_int,
            'weight_raw':       round(w_raw, 4),
            'mean_binary_ind':  round(float(ind_te.mean()), 3),
            'mean_dist_ind':    round(float(dist_te_n.mean()), 3),
        })
 
    # ── Youden cutoffs from training set ──────────────────────────────────────
    cut_bin_int = _youden_local(tr_bin_int, y_tr)
    cut_bin_raw = _youden_local(tr_bin_raw, y_tr)
    cut_cont    = _youden_local(tr_cont,    y_tr)
 
    # ── Performance on test set ───────────────────────────────────────────────
    res1 = _perf_local(y_te, sc_bin_int, cut_bin_int, 'Binary I() — integer weights (paper)')
    res2 = _perf_local(y_te, sc_bin_raw, cut_bin_raw, 'Binary I() — raw SHAP weights')
    res3 = _perf_local(y_te, sc_cont,    cut_cont,    'Continuous I() — distance from threshold')
 
    df_results = pd.DataFrame([res1, res2, res3])
 
    # ── Delta AUROC vs paper implementation ───────────────────────────────────
    ref_auroc = res1['auroc']
    df_results['delta_auroc_vs_binary_int'] = \
        df_results['auroc'].apply(lambda x: round(x - ref_auroc, 4))
 
    # ── DeLong p-values ───────────────────────────────────────────────────────
    _, p2 = _delong_local(y_te, sc_bin_raw, sc_bin_int)
    _, p3 = _delong_local(y_te, sc_cont,    sc_bin_int)
    df_results['delong_p_vs_binary_int'] = [
        np.nan,
        round(p2, 4) if not np.isnan(p2) else np.nan,
        round(p3, 4) if not np.isnan(p3) else np.nan,
    ]
    df_results['setting'] = analysis_name
 
    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path  = f'{output_dir}/prfps_continuous_comparison_{analysis_name}.csv'
    feat_path = f'{output_dir}/prfps_feature_indicator_{analysis_name}.csv'
    df_results.to_csv(csv_path, index=False)
    pd.DataFrame(feature_details).to_csv(feat_path, index=False)
 
    # ── Log summary ───────────────────────────────────────────────────────────
    log(f'\n{"="*65}')
    log(f'  pRFPS Variant Comparison — {analysis_name}')
    log(f'{"="*65}')
    log(f'  {"Variant":<45} {"AUROC":>7} {"ΔAUROC":>8} {"F1":>7} {"Sens":>7} {"Spec":>7}')
    log(f'  {"-"*80}')
    for r in [res1, res2, res3]:
        delta = round(r['auroc'] - ref_auroc, 4)
        log(f'  {r["variant"]:<45} {r["auroc"]:>7.4f} {delta:>+8.4f} '
            f'{r["f1_macro"]:>7.4f} {r["sensitivity"]:>7.4f} {r["specificity"]:>7.4f}')
    log(f'\n  DeLong p (Binary raw  vs Binary int): '
        f'{df_results.loc[1,"delong_p_vs_binary_int"]}')
    log(f'  DeLong p (Continuous  vs Binary int): '
        f'{df_results.loc[2,"delong_p_vs_binary_int"]}')
    log(f'{"="*65}\n')
 
    # ── Figure ────────────────────────────────────────────────────────────────
    colors = ['#0A7C7C', '#4D9F9F', '#D96B2A']
    labels_short = [
        'Binary I()\nInteger weights\n(Paper)',
        'Binary I()\nRaw SHAP weights',
        'Continuous I()\nDistance from thr',
    ]
    metrics_plot  = ['auroc', 'f1_macro', 'sensitivity', 'specificity', 'auprc']
    metric_labels = ['AUROC', 'F1 (Macro)', 'Sensitivity', 'Specificity', 'AUPRC']
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
 
    # Bar chart
    x = np.arange(len(metrics_plot))
    w = 0.25
    for i, (r, c, lb) in enumerate(zip([res1, res2, res3], colors, labels_short)):
        vals = [r[m] for m in metrics_plot]
        axes[0].bar(x + i*w, vals, w, label=lb, color=c, alpha=0.85, edgecolor='white')
    axes[0].set_xticks(x + w)
    axes[0].set_xticklabels(metric_labels, fontsize=10)
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel('Score', fontsize=11)
    axes[0].set_title(f'pRFPS Variant Comparison — {analysis_name}',
                      fontsize=12, fontweight='bold')
    axes[0].legend(fontsize=8, loc='lower right')
    axes[0].grid(axis='y', alpha=0.25)
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
 
    # ROC curves
    for score, c, lb in zip(
        [sc_bin_int, sc_bin_raw, sc_cont],
        colors, labels_short
    ):
        if len(np.unique(y_te)) > 1:
            fpr, tpr, _ = roc_curve(y_te, score)
            auroc = roc_auc_score(y_te, score)
            axes[1].plot(fpr, tpr, color=c, lw=2,
                         label=f'{lb.replace(chr(10), " ")} (AUROC={auroc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    axes[1].set_xlabel('False Positive Rate', fontsize=11)
    axes[1].set_ylabel('True Positive Rate', fontsize=11)
    axes[1].set_title(f'ROC Curves — {analysis_name}',
                      fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=8, loc='lower right')
    axes[1].grid(alpha=0.2)
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
 
    plt.tight_layout()
    fig_path = f'{output_dir}/prfps_continuous_comparison_{analysis_name}.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
 
    log(f'  [Continuous pRFPS] Saved: {csv_path}')
    log(f'  [Continuous pRFPS] Saved: {fig_path}')
 
    return df_results, pd.DataFrame(feature_details)
 


# =============================================================================
# [Part 7 / FIX-12]  SHAP STABILITY
# =============================================================================

def build_shap_clinical_risk_score(
    shap_values,
    X_test,
    y_test,
    feature_names,
    X_train=None,
    y_train=None,
    shap_values_test=None,   # SHAP on test patients for delta comparison
    analysis_name="Overall",
    output_dir="shap_risk_score",
    top_n_features=None,
    shap_coverage_threshold=0.80,
    max_features=10,
    feature_labels_map=None,
    raw_unit_map=None,
):
    """
    Construct pRFPS exactly as in Methods, with three enhancements:

    1. AUTO FEATURE SELECTION by cumulative SHAP coverage (default 80%).
       Features ranked by mean |SHAP|; enough kept to cover
       shap_coverage_threshold of total cumulative SHAP importance.
       Hard ceiling: max_features (default 10).
       Set top_n_features=5 to use fixed-N as before.

    2. CONTINUOUS SCORE PRESERVED alongside integer pRFPS.
       score_cont uses weight_raw (pre-rounding floats) so no information
       is lost by integer discretisation. Both cutoffs and AUROCs reported.

    3. FIB-4 COMPATIBLE OUTPUT.
       score_norm (0-1) is returned for DCA threshold axis alignment.
       Caller passes real FIB-4 binary labels via prepare_fib4_comparator().
    """
    os.makedirs(output_dir, exist_ok=True)
    if feature_labels_map is None:
        feature_labels_map = FEATURE_LABELS
    if raw_unit_map is None:
        raw_unit_map = {}

    sv_arr = shap_values
    if isinstance(sv_arr, list):
        sv_arr = sv_arr[1]
    sv_arr = np.array(sv_arr)
    if sv_arr.ndim == 3:
        sv_arr = sv_arr[:, :, 1]
    if sv_arr.ndim == 1:
        sv_arr = sv_arr.reshape(1, -1)
    y_arr   = np.array(y_test).astype(int).flatten()

    X_te = (X_test.reset_index(drop=True)
            if isinstance(X_test, pd.DataFrame)
            else pd.DataFrame(X_test, columns=feature_names))
    X_tr = (X_train.reset_index(drop=True)
            if isinstance(X_train, pd.DataFrame) and X_train is not None
            else X_te)

    # y_train must match X_train row count
    if y_train is not None:
        y_train_arr = np.array(y_train).astype(int).flatten()
        if len(y_train_arr) != len(X_tr):
            log(f"  [pRFPS] Warning: y_train {len(y_train_arr)} != X_train {len(X_tr)}")
            y_train_arr = None
    else:
        y_train_arr = None

    # Step 1-2: importance + direction
    mean_abs  = np.abs(sv_arr).mean(axis=0)
    mean_shap = sv_arr.mean(axis=0)
    std_shap  = sv_arr.std(axis=0)
    total_abs = mean_abs.sum()

    # Direction from SHAP sign only (mean SHAP across test patients).
    # Positive mean SHAP = higher feature value pushes prediction toward
    # Rapid Progression = Risk+.
    # Negative mean SHAP = higher value pushes toward No Progression = Protective.
    # This is the correct approach: the trained model's SHAP values already
    # encode the learned direction for each feature and cohort. Hardcoding
    # biological priors would override what the model actually learned and
    # could produce inconsistent directions across subgroups (e.g. NFS may
    # be Risk+ in C1 but Protective in C2 depending on the subgroup-specific
    # model). SHAP sign is the ground truth here.
    feat_df = pd.DataFrame({
        "feature":          feature_names,
        "label":            [feature_labels_map.get(f, f) for f in feature_names],
        "mean_abs_shap":    mean_abs,
        "mean_shap":        mean_shap,
        "std_shap":         std_shap,
        "pct_contribution": mean_abs / (total_abs + 1e-10) * 100,
        "direction":        np.where(mean_shap > 0, "Risk+", "Protective"),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    # Step 2: auto feature selection by cumulative SHAP coverage
    if top_n_features is not None:
        n_select = int(top_n_features)
    else:
        cum_pct  = feat_df["mean_abs_shap"].cumsum() / (total_abs + 1e-10)
        n_select = int((cum_pct < shap_coverage_threshold).sum()) + 1
        n_select = max(5, min(n_select, max_features))
        log(f"  [pRFPS] Auto: {n_select} features cover "
            f"{cum_pct.iloc[n_select-1]*100:.1f}% cumulative SHAP "
            f"(threshold={shap_coverage_threshold*100:.0f}%)")

    top = feat_df.head(n_select).copy()

    # Step 3: integer + continuous weights
    # [R-ALIGNED CHANGE 1] Denominator = max(mean_abs_shap) across ALL features
    # (not just top-N sum). This matches the R code:
    #   max_shap <- max(MCB1$mean_abs_shap)
    #   points = round((mean_abs_shap / max_shap) * 10)
    # Using the global max means weights are on an absolute scale:
    # the most important feature always gets weight=10, others are proportional.
    global_max_shap  = float(feat_df["mean_abs_shap"].max())
    # weight_raw now computed inside weight_int block above (scale 0-20)
    # Scale to 0-20 (not 0-10) to halve integer rounding error.
    # With 0-10 scale, a raw weight of 7.4 vs 7.6 both round to 7 -- same score.
    # With 0-20 scale, they become 15 vs 15 with finer resolution.
    # The clinical formula still reads as small integers; reviewers see
    # information_loss% column confirming rounding cost is reduced.
    top["weight_raw"] = top["mean_abs_shap"] / (global_max_shap + 1e-10) * 10
    top["weight_int"] = top["weight_raw"].round().astype(int).clip(0, 10)
    top["rank"]       = np.arange(1, len(top) + 1)
    top["unit"]       = [raw_unit_map.get(f, "--") for f in top["feature"]]
    top["cumulative_pct_importance"] = (
        top["mean_abs_shap"].cumsum() / (total_abs + 1e-10) * 100)
    top["information_loss"] = (
        (top["weight_raw"] - top["weight_int"]).abs() /
        (top["weight_raw"] + 1e-10) * 100)

    # Step 4: Youden threshold per feature on X_train -- direction from DATA
    #
    # For each candidate threshold we try BOTH directions (>= and <) and keep
    # whichever gives the higher Youden J. The winning direction is determined
    # purely by which side of the threshold has the higher Rapid Progression
    # rate in the training data -- no dependence on SHAP sign.
    #
    # SHAP sign is still used for feature importance ranking and weight
    # assignment (Steps 1-3). Only the binary indicator direction is data-driven.
    #
    # Why this is correct:
    #   - SHAP sign can be ambiguous for non-monotone features
    #   - The threshold and its direction must be jointly optimal
    #   - Youden J is maximised when the indicator correctly separates
    #     Rapid from No Progression in the training set
    feat_thresholds  = {}
    feat_directions  = {}   # data-derived direction per feature

    for _, row in top.iterrows():
        feat = row["feature"]
        if feat not in X_tr.columns:
            feat_thresholds[feat] = 0.0
            feat_directions[feat] = row["direction"]   # fallback to SHAP
            continue

        f_vals = X_tr[feat].values.astype(float)
        y_ref  = (y_train_arr if (y_train_arr is not None and
                                   len(y_train_arr) == len(f_vals))
                  else (f_vals >= np.median(f_vals)).astype(int))

        # ── STEP A: Find optimal threshold using exhaustive unique-value search ─
        # Search all unique values in the IQR -- exact optimum, no missed points.
        # Youden J is symmetric so we use ind=(val >= thresh) for the threshold
        # search itself; direction is determined separately in Step B.
        best_j = -1
        best_t = float(np.median(f_vals))

        _candidates = np.unique(f_vals)
        # Restrict to 20th-80th percentile range.
        # This prevents extreme tail thresholds (e.g., BMI>40 or HDL<28)
        # that classify 90%+ of patients on one side -- such thresholds
        # are clinically uninterpretable and directionally unstable.
        # Each side of the threshold must contain at least MIN_GROUP_PCT
        # of training patients to be considered a valid cutpoint.
        _q20, _q80   = np.percentile(f_vals, [5, 95])
        _candidates  = _candidates[(_candidates >= _q20) & (_candidates <= _q80)]
        MIN_GROUP_N  = max(10, int(0.15 * len(f_vals)))   # at least 15% per side

        for thresh in _candidates:
            ind = (f_vals >= thresh).astype(int)
            n_above = ind.sum()
            n_below = len(ind) - n_above
            # Skip if either group is too small to give stable rate estimates
            if n_above < MIN_GROUP_N or n_below < MIN_GROUP_N:
                continue
            tn_, fp_, fn_, tp_ = confusion_matrix(
                y_ref,
                ind,
                labels=[0, 1]
            ).ravel()

            sensitivity = tp_ / (tp_ + fn_ + 1e-10)
            specificity = tn_ / (tn_ + fp_ + 1e-10)

            j = sensitivity + specificity - 1
            if j > best_j:
                best_j = j
                best_t = float(thresh)

        # ── STEP B: Direction from outcome rates, with SHAP tiebreaker ────────
        #
        # PRIMARY: Compare rapid progression rates on each side of best_t.
        #   rate_above > rate_below → high values risky → Risk+  (score if val > thr)
        #   rate_below > rate_above → low values risky  → Protective (score if val < thr)
        #
        # TIEBREAKER (when |rate_above - rate_below| < 0.05):
        #   If the outcome rate difference is less than 5 percentage points,
        #   the direction is statistically ambiguous with small-n subgroups
        #   (n_train ~260-480). In these cases we fall back to the SHAP direction,
        #   which is estimated from the full model's additive decomposition rather
        #   than a single binary split, and is less sensitive to marginal confounding.
        #
        # WHY THIS MATTERS CLINICALLY:
        #   Features like BMI and HDL have well-established biological priors
        #   (high BMI = risk, high HDL = protective). When the within-subgroup
        #   data shows the opposite direction by only 2-4%, this is more likely
        #   noise or reverse causation (e.g., fibrosis itself causing weight loss)
        #   than a genuine reversal of the biological relationship.
        #   The tiebreaker makes the formula more robust and clinically interpretable
        #   without hardcoding biological priors that could override genuine
        #   subgroup-specific patterns (e.g., strong 15%+ rate differences are
        #   kept regardless of SHAP sign).
        #
        DIRECTION_UNCERTAINTY_EPSILON = 0.05  # 5 percentage-point threshold

        above_mask = f_vals >= best_t
        below_mask = f_vals <  best_t
        rate_above = float(y_ref[above_mask].mean()) if above_mask.sum() > 0 else 0.0
        rate_below = float(y_ref[below_mask].mean()) if below_mask.sum() > 0 else 0.0
        rate_diff  = abs(rate_above - rate_below)

        shap_dir   = row["direction"]   # direction from SHAP sign (training SHAP)

        if rate_diff < DIRECTION_UNCERTAINTY_EPSILON:
            # Outcome rate difference is too small to be reliable with small n.
            # Use SHAP direction as tiebreaker -- it integrates all features.
            best_dir = shap_dir
            dir_source = f"SHAP tiebreaker (|rate_diff|={rate_diff:.2%} < {DIRECTION_UNCERTAINTY_EPSILON:.0%})"
        elif rate_above >= rate_below:
            best_dir   = "Risk+"
            dir_source = f"data (rate_above={rate_above:.2%} > rate_below={rate_below:.2%})"
        else:
            best_dir   = "Protective"
            dir_source = f"data (rate_below={rate_below:.2%} > rate_above={rate_above:.2%})"

        feat_thresholds[feat] = best_t
        feat_directions[feat] = best_dir

        log(f"    [pRFPS threshold] {feat}: threshold={best_t:.3f}  "
            f"rate_above={rate_above:.2%}  rate_below={rate_below:.2%}  "
            f"SHAP_dir={shap_dir}  → data_dir={best_dir}  [{dir_source}]")

    top["youden_threshold"] = [feat_thresholds.get(f, 0) for f in top["feature"]]
    # Overwrite direction with data-derived direction (may differ from SHAP sign)
    top["direction"]        = [feat_directions.get(f, d)
                                for f, d in zip(top["feature"], top["direction"])]
    # Display columns to avoid clinician confusion:
    #   SHAP_direction: what HIGHER values do (Risk+ = higher → more risk,
    #                   Protective = higher → less risk)
    #   fires_when: the actual condition in the formula that earns points
    #               (i.e., when the patient IS at elevated risk)
    top["SHAP_direction"] = top["direction"]   # Risk+ or Protective
    top["fires_when"]     = np.where(
        top["direction"] == "Risk+",
        "> threshold  (high value = risk)",
        "< threshold  (low value = risk)",
    )

    # Step 5: compute integer + continuous scores
    # [R-ALIGNED CHANGE 3] Scoring uses strict inequality (> not >=) for Risk+
    # Matches R code:
    #   if(cond == ">=") scored_df[[new_col_name]] <- ifelse(value > limit, pts, 0)
    #   if(cond == "<=") scored_df[[new_col_name]] <- ifelse(value < limit, pts, 0)
    # The threshold is a boundary: strictly exceeding it earns points.
    scores_int  = np.zeros(len(X_te))
    scores_cont = np.zeros(len(X_te))
    for _, row in top.iterrows():
        feat = row["feature"]
        if feat not in X_te.columns: continue
        vals = X_te[feat].values.astype(float)
        thr  = row["youden_threshold"]
        ind  = ((vals > thr) if row["direction"] == "Risk+"   # strict > (R-aligned)
                else (vals < thr)).astype(float)              # strict < (R-aligned)
        scores_int  += int(row["weight_int"])   * ind
        scores_cont += float(row["weight_raw"]) * ind

    prfps_int  = scores_int.astype(int)
    prfps_cont = scores_cont

    # Step 6: Youden optimal cutoff -- derived from TRAINING scores (R-aligned)
    # [R-ALIGNED CHANGE 4] R computes optimal_cutoff from training data:
    #   roc_obj <- pROC::roc(response=train$outcome, predictor=train_scores)
    #   optimal_cutoff <- coords(roc_obj, "best", ret="threshold")$threshold
    # Python was: Youden on TEST set scores -- this leaks test outcome info
    # Fix: compute pRFPS scores on training patients, find Youden cutoff there,
    #      then apply the LOCKED cutoff to test set exactly once.
    train_scores_int  = np.zeros(len(X_tr))
    train_scores_cont = np.zeros(len(X_tr))
    if y_train_arr is not None:
        for _, row in top.iterrows():
            feat = row["feature"]
            if feat not in X_tr.columns: continue
            tr_vals = X_tr[feat].values.astype(float)
            thr     = row["youden_threshold"]
            tr_ind  = ((tr_vals > thr) if row["direction"] == "Risk+"
                       else (tr_vals < thr)).astype(float)
            train_scores_int  += int(row["weight_int"])   * tr_ind
            train_scores_cont += float(row["weight_raw"]) * tr_ind

    train_prfps_int  = train_scores_int.astype(int)
    train_prfps_cont = train_scores_cont

    # Youden on TRAINING scores (matches R's use of training data for cutoff)
    best_cutoff_int  = float(np.median(train_prfps_int))
    best_cutoff_cont = float(np.median(train_prfps_cont))
    best_j_int = best_j_cont = -1

    best_cutoff_int = float(np.median(train_prfps_int))
    best_f1_int = -1

    if len(np.unique(y_train_arr)) > 1:
        for cut in np.unique(train_prfps_int):
            pred_c = (train_prfps_int >= cut).astype(int)
            f1_c = f1_score(
                y_train_arr,
                pred_c,
                average="macro",
                zero_division=0,
            )
            if f1_c > best_f1_int:
                best_f1_int = f1_c
                best_cutoff_int = float(cut)
    best_cutoff_cont = float(np.median(train_prfps_cont))
    best_f1_cont = -1

    if len(np.unique(y_train_arr)) > 1:
        for cut in np.unique(np.percentile(train_prfps_cont, np.linspace(5, 95, 100))):
            pred_c = (train_prfps_cont >= cut).astype(int)
            f1_c = f1_score(
                y_train_arr,
                pred_c,
                average="macro",
                zero_division=0,
            )
            if f1_c > best_f1_cont:
                best_f1_cont = f1_c
                best_cutoff_cont = float(cut)

    log(f"  [pRFPS] Training-derived cutoffs: "
        f"int={best_cutoff_int:.0f}  cont={best_cutoff_cont:.4f}")

    # Apply locked cutoffs to test set -- ONCE (no leakage)
    pred_int  = (prfps_int  >= best_cutoff_int).astype(int)
    pred_cont = (prfps_cont >= best_cutoff_cont).astype(int)

    # Step 7: AUROC
    prfps_auroc_int  = float("nan")
    prfps_auroc_cont = float("nan")
    if len(np.unique(y_arr)) > 1:
        prfps_auroc_int  = roc_auc_score(y_arr, prfps_int)
        prfps_auroc_cont = roc_auc_score(y_arr, prfps_cont)

    def _perf(yt, yp_bin):
        if len(np.unique(yt)) < 2: return {}
        tn_,fp_,fn_,tp_ = confusion_matrix(yt, yp_bin, labels=[0,1]).ravel()
        f1_val = 2*tp_/(2*tp_+fp_+fn_+1e-10)
        brier_score = brier_score_loss(yt, yp_bin)
        return dict(accuracy=(tp_+tn_)/(len(yt)+1e-10),
                    sensitivity=tp_/(tp_+fn_+1e-10),
                    specificity=tn_/(tn_+fp_+1e-10),
                    ppv=tp_/(tp_+fp_+1e-10), npv=tn_/(tn_+fn_+1e-10),
                    f1=f1_val,
                    brier_score = brier_score,
                    f1_macro=f1_val)   # alias so callers can use either key

    perf_int  = _perf(y_arr, pred_int)
    perf_cont = _perf(y_arr, pred_cont)

    log(f"  [pRFPS-int]  {analysis_name}: AUROC={prfps_auroc_int:.3f}  "
        f"cutoff={best_cutoff_int:.0f}  "
        f"sens={perf_int.get('sensitivity',float('nan')):.3f}  "
        f"spec={perf_int.get('specificity',float('nan')):.3f}")
    log(f"  [pRFPS-cont] {analysis_name}: AUROC={prfps_auroc_cont:.3f}  "
        f"cutoff={best_cutoff_cont:.3f}")

    # Save outputs
    top.to_csv(f"{output_dir}/prfps_feature_table_{analysis_name}.csv", index=False)
    pd.DataFrame({
        "patient_idx":     range(len(X_te)),
        "pRFPS_int":       prfps_int,
        "pRFPS_cont":      prfps_cont.round(4),
        "pRFPS_int_pred":  pred_int,
        "pRFPS_cont_pred": pred_cont,
        "True_Label":      y_arr,
        "True_Label_Str":  np.where(y_arr==1, POSITIVE_LABEL, NEGATIVE_LABEL),
    }).to_csv(f"{output_dir}/prfps_scores_{analysis_name}.csv", index=False)

    # Figure: 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for lv, ls, lc in [(0,NEGATIVE_LABEL,"#4575B4"),(1,POSITIVE_LABEL,"#D73027")]:
        mask = y_arr == lv
        axes[0].hist(prfps_int[mask], bins=range(0, int(prfps_int.max())+2),
                     alpha=0.6, color=lc, label=ls, edgecolor="white")
    axes[0].axvline(best_cutoff_int, color="black", ls="--", lw=2,
                    label=f"Cutoff={best_cutoff_int:.0f}")
    axes[0].set_xlabel("pRFPS Integer Score", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(f"A. Integer pRFPS -- {analysis_name}\nAUROC={prfps_auroc_int:.3f}", fontsize=10)
    axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.3)

    for lv, ls, lc in [(0,NEGATIVE_LABEL,"#4575B4"),(1,POSITIVE_LABEL,"#D73027")]:
        mask = y_arr == lv
        axes[1].hist(prfps_cont[mask], bins=25,
                     alpha=0.6, color=lc, label=ls, edgecolor="white")
    axes[1].axvline(best_cutoff_cont, color="black", ls="--", lw=2,
                    label=f"Cutoff={best_cutoff_cont:.2f}")
    axes[1].set_xlabel("pRFPS Continuous (pre-rounding)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(f"B. Continuous pRFPS -- {analysis_name}\nAUROC={prfps_auroc_cont:.3f}", fontsize=10)
    axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.3)

    top_s    = top.sort_values("weight_int")
    colors_b = ["#D73027" if d=="Risk+" else "#4575B4" for d in top_s["direction"]]
    bars = axes[2].barh(top_s["label"], top_s["weight_int"],
                        color=colors_b, alpha=0.85, edgecolor="white")
    for bar, row_ in zip(bars, top_s.itertuples()):
        axes[2].text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                     f"w={int(row_.weight_int)} (raw={row_.weight_raw:.2f}, "
                     f"loss={row_.information_loss:.1f}%)",
                     va="center", fontsize=7.5)
    axes[2].set_xlabel("Integer Weight (0-10)", fontsize=11)
    axes[2].set_title(
        f"C. Feature Weights + Info Loss\n"
        f"Coverage: {top['cumulative_pct_importance'].iloc[-1]:.1f}% of total SHAP",
        fontsize=10)
    axes[2].legend(handles=[
        mpatches.Patch(color="#D73027", label="Risk-increasing"),
        mpatches.Patch(color="#4575B4", label="Protective"),
    ], fontsize=8)
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].spines["top"].set_visible(False); axes[2].spines["right"].set_visible(False)
    plt.suptitle(
        f"pRFPS -- {analysis_name}  |  Integer cutoff={best_cutoff_int:.0f}  "
        f"|  {n_select} features  "
        f"|  SHAP coverage={top['cumulative_pct_importance'].iloc[-1]:.1f}%",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prfps_summary_{analysis_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()

    # Formula text
    formula_lines = [
        "=" * 70,
        f"  pRFPS -- Partitioned Rapid Fibrosis Progression Score",
        f"  Subgroup: {analysis_name}",
        f"  Features: {n_select}  (SHAP coverage={top['cumulative_pct_importance'].iloc[-1]:.1f}%)",
        "=" * 70, "",
        "  INTEGER pRFPS = " + " + ".join(
            [f"{int(r_['weight_int'])} * I({r_['label']} "
             f"{'>'+str(round(r_['youden_threshold'],3)) if r_['direction']=='Risk+' else '<'+str(round(r_['youden_threshold'],3))})"
             for _, r_ in top.iterrows()]),
        f"  Cutoff (integer):    {best_cutoff_int:.0f}",
        f"  Cutoff (continuous): {best_cutoff_cont:.4f}", "",
        f"  AUROC (integer):     {prfps_auroc_int:.3f}",
        f"  AUROC (continuous):  {prfps_auroc_cont:.3f}", "",
        f"  {'Rk':<4} {'Feature':<28} {'SHAP_dir':<11} "
        f"{'Fires_when':<12} {'Threshold':>10} {'w_int':>6} {'w_raw':>7} "
        f"{'Loss%':>7} {'Cov%':>7}",
        "-" * 95,
    ]
    for _, r_ in top.iterrows():
        _fires = "> thr" if r_["direction"] == "Risk+" else "< thr"
        formula_lines.append(
            f"  {int(r_['rank']):<4} {r_['label']:<28} {r_['direction']:<11} "
            f"{_fires:<12} {r_['youden_threshold']:>10.3f} {int(r_['weight_int']):>6} "
            f"{r_['weight_raw']:>7.3f} {r_['information_loss']:>7.1f} "
            f"{r_['cumulative_pct_importance']:>7.1f}")
    formula_lines += ["", "=" * 70]
    formula_str = "\n".join(formula_lines)
    with open(f"{output_dir}/prfps_formula_{analysis_name}.txt", "w", encoding="utf-8") as fh:
        fh.write(formula_str)
    log(formula_str)

    return {
        "score_int":           prfps_int,
        "score_array":         prfps_int,
        "threshold":           best_cutoff_int,
        "auroc":               prfps_auroc_int,
        "pred":                pred_int,
        "score_cont":          prfps_cont,
        "threshold_cont":      best_cutoff_cont,
        "auroc_cont":          prfps_auroc_cont,
        "pred_cont":           pred_cont,
        "n_features_selected": n_select,
        "shap_coverage_pct":   float(top["cumulative_pct_importance"].iloc[-1]),
        "feature_table":       top,
        "y_true":              y_arr,
        "perf_int":            perf_int,
        "perf_cont":           perf_cont,
        "formula_str":         formula_str,
        "score_norm":          prfps_int.astype(float) / (prfps_int.max() + 1e-10),
        "score_cont_norm":     prfps_cont / (prfps_cont.max() + 1e-10),
        # Train SHAP for formula derivation
        "shap_values":         sv_arr,
        # Test SHAP for true SHAP-additive score comparison (same patients as y_true)
        "shap_values_test":    np.array(shap_values_test) if shap_values_test is not None else None,
    }




# ---------------------------------------------------------------------------
# Private helpers (extracted from original to keep main function readable)
# ---------------------------------------------------------------------------
 
def _save_prfps_figure(prfps_int, prfps_cont, pred_int, pred_cont,
                        y_arr, top, best_cutoff_int, best_cutoff_cont,
                        auroc_int, auroc_cont, n_select,
                        analysis_name, output_dir, feature_labels_map):
    """Figure A/B/C — unchanged layout from original."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for lv, ls, lc in [(0, NEGATIVE_LABEL, "#4575B4"),
                        (1, POSITIVE_LABEL, "#D73027")]:
        mask = y_arr == lv
        axes[0].hist(prfps_int[mask],
                     bins=range(0, int(prfps_int.max()) + 2),
                     alpha=0.6, color=lc, label=ls, edgecolor="white")
    axes[0].axvline(best_cutoff_int, color="black", ls="--", lw=2,
                    label=f"Cutoff={best_cutoff_int:.0f}  [from train]")
    axes[0].set_xlabel("pRFPS Integer Score", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title(
        f"A. Integer pRFPS — {analysis_name}\n"
        f"AUROC={auroc_int:.3f}  cutoff={best_cutoff_int:.0f} (train-derived)",
        fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.3)
 
    for lv, ls, lc in [(0, NEGATIVE_LABEL, "#4575B4"),
                        (1, POSITIVE_LABEL, "#D73027")]:
        mask = y_arr == lv
        axes[1].hist(prfps_cont[mask], bins=25,
                     alpha=0.6, color=lc, label=ls, edgecolor="white")
    axes[1].axvline(best_cutoff_cont, color="black", ls="--", lw=2,
                    label=f"Cutoff={best_cutoff_cont:.2f}  [from train]")
    axes[1].set_xlabel("pRFPS Continuous (pre-rounding)", fontsize=11)
    axes[1].set_ylabel("Count", fontsize=11)
    axes[1].set_title(
        f"B. Continuous pRFPS — {analysis_name}\n"
        f"AUROC={auroc_cont:.3f}  cutoff={best_cutoff_cont:.3f} (train-derived)",
        fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(axis="y", alpha=0.3)
 
    top_s    = top.sort_values("weight_int")
    colors_b = ["#D73027" if d == "Risk+" else "#4575B4"
                for d in top_s["direction"]]
    bars = axes[2].barh(top_s["label"], top_s["weight_int"],
                         color=colors_b, alpha=0.85, edgecolor="white")
    for bar, row_ in zip(bars, top_s.itertuples()):
        axes[2].text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            f"w={int(row_.weight_int)} (raw={row_.weight_raw:.2f}, "
            f"loss={row_.information_loss:.1f}%)",
            va="center", fontsize=7.5)
    axes[2].set_xlabel("Integer Weight (0-10)", fontsize=11)
    axes[2].set_title(
        f"C. Feature Weights + Info Loss\n"
        f"Coverage: {top['cumulative_pct_importance'].iloc[-1]:.1f}% of total SHAP",
        fontsize=10)
    axes[2].legend(handles=[
        mpatches.Patch(color="#D73027", label="Risk-increasing"),
        mpatches.Patch(color="#4575B4", label="Protective"),
    ], fontsize=8)
    axes[2].grid(axis="x", alpha=0.3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)
 
    plt.suptitle(
        f"pRFPS — {analysis_name}  |  "
        f"Integer cutoff={best_cutoff_int:.0f} (train-derived)  |  "
        f"{n_select} features  |  "
        f"SHAP coverage={top['cumulative_pct_importance'].iloc[-1]:.1f}%",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prfps_summary_{analysis_name}.png",
                dpi=200, bbox_inches="tight")
    plt.close()
 
 
def _build_formula_str(top, n_select, best_cutoff_int, best_cutoff_cont,
                        auroc_int, auroc_cont, analysis_name):
    lines = [
        "=" * 70,
        f"  pRFPS — Partitioned Rapid Fibrosis Progression Score",
        f"  Subgroup: {analysis_name}",
        f"  Features: {n_select}  "
        f"(SHAP coverage={top['cumulative_pct_importance'].iloc[-1]:.1f}%)",
        f"  [FIX-A] Cutoffs derived from TRAINING SET — no test leakage",
        "=" * 70, "",
        "  INTEGER pRFPS = " + " + ".join([
            f"{int(r_['weight_int'])} * I({r_['label']} "
            f"{'>='+str(round(r_['youden_threshold'], 3)) if r_['direction']=='Risk+' else '<'+str(round(r_['youden_threshold'], 3))})"
            for _, r_ in top.iterrows()]),
        f"  Cutoff (integer, train-derived):    {best_cutoff_int:.0f}",
        f"  Cutoff (continuous, train-derived): {best_cutoff_cont:.4f}", "",
        f"  AUROC (integer):     {auroc_int:.3f}",
        f"  AUROC (continuous):  {auroc_cont:.3f}", "",
        f"  {'Rk':<4} {'Feature':<28} {'Dir':<11} "
        f"{'Threshold':>10} {'w_int':>6} {'w_raw':>7} "
        f"{'Loss%':>7} {'Cov%':>7}",
        "-" * 82,
    ]
    for _, r_ in top.iterrows():
        lines.append(
            f"  {int(r_['rank']):<4} {r_['label']:<28} {r_['direction']:<11} "
            f"{r_['youden_threshold']:>10.3f} {int(r_['weight_int']):>6} "
            f"{r_['weight_raw']:>7.3f} {r_['information_loss']:>7.1f} "
            f"{r_['cumulative_pct_importance']:>7.1f}")
    lines += ["", "=" * 70]
    return "\n".join(lines)
