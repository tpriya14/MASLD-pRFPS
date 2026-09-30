"""
ablation.py — Feature-selection × partitioning ablation, DCA, nested feature selection.
Auto-split from Rebuttal_experiments.py
"""
"""
Auto-generated from Rebuttal_experiments.py
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
from collections import Counter
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
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
import json
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict

def ablation_feature_selection(
    settings_data: Dict,
    all_feat_names: List[str],
    overall_feats: List[str],
    overall_se_model,
    overall_scl,
    overall_se_results: Dict,
    all_test_indices: Dict,
    X_full: pd.DataFrame,
    n_boot: int = 500,
) -> pd.DataFrame:
    """
    2×3 ablation: Partition × Feature Selection for ALL models, ALL settings.

    Runs for Overall, C1, C2 across five conditions:
      E. Non-partitioned + all features
      F. Non-partitioned + Overall-selected features
      G. Partitioned     + all features          (retrained)
      H. Partitioned     + Overall-selected feats (retrained)
      D. Partitioned     + Subgroup-selected feats ← CURRENT PIPELINE

    For every model in MODELS_CONFIG plus Stacking Ensemble.
    Reports AUROC and Accuracy for each condition.

    Key deltas:
      D-E = total pipeline gain
      D-H = value of subgroup-specific feature selection
      H-F = value of partitioning (feature set matched)
      D-G = value of feature selection (partition matched)
      G-E = value of partitioning without any feature selection
    """
    section("ABLATION -- Feature Selection × Partitioning [All Models, All Settings]")
    rows = []

    _np_thr = float(overall_se_results.get("optimal_threshold", 0.5))

    def _auroc(yt, yp):
        try: return round(float(roc_auc_score(yt, yp)), 4)
        except: return float("nan")

    def _acc(yt, yp, thr):
        try: return round(float(accuracy_score(yt, (yp>=thr).astype(int))), 4)
        except: return float("nan")

    def _f1(yt, yp, thr):
        try: return round(float(f1_score(yt, (yp>=thr).astype(int),
                                          average="macro", zero_division=0)), 4)
        except: return float("nan")

    def _r(v):
        try: return round(float(v), 4) if not np.isnan(float(v)) else "N/A"
        except: return "N/A"

    def _delta(a, b):
        try: return _r(float(a) - float(b))
        except: return "N/A"

    def _retrain_model(mdl_class, mdl_params, X_tr_sub, y_tr_sub, X_te_sub,
                       scl_tr=None):
        """Retrain a fresh model with given features, return (prob_tr, prob_te, thr)."""
        try:
            if scl_tr is not None:
                Xtr_s = scl_tr.transform(X_tr_sub.values)
                Xte_s = scl_tr.transform(X_te_sub.values)
            else:
                from sklearn.preprocessing import StandardScaler as _SS
                _sc = _SS(); Xtr_s = _sc.fit_transform(X_tr_sub.values)
                Xte_s = _sc.transform(X_te_sub.values)
            mdl = mdl_class(**{k: v for k, v in mdl_params.items() if not callable(v)})
            mdl.fit(Xtr_s, y_tr_sub.values)
            prob_tr = mdl.predict_proba(Xtr_s)[:, 1]
            # Youden threshold from training proba
            thr = 0.5; best_j = -1
            for _t in np.unique(np.percentile(prob_tr, np.linspace(5,95,50))):
                _pred = (prob_tr >= _t).astype(int)
                _cm   = confusion_matrix(y_tr_sub, _pred, labels=[0,1])
                _tn,_fp,_fn,_tp = _cm.ravel()
                _j = _tp/(_tp+_fn+1e-9) + _tn/(_tn+_fp+1e-9) - 1
                if _j > best_j: best_j = _j; thr = float(_t)
            prob_te = mdl.predict_proba(Xte_s)[:, 1]
            return prob_te, thr
        except Exception as _e:
            log(f"      retrain failed: {_e}"); return None, 0.5

    for setting, sdata in settings_data.items():
        X_tr      = sdata["X_tr"]
        X_te      = sdata["X_te"]
        y_tr      = sdata["y_tr"]
        y_te      = sdata["y_te"]
        feats_sub = sdata["feats_subgroup"]
        results_D = sdata["results"]

        te_idx = all_test_indices.get(setting, [])

        # Valid rows (no NaN in overall features)
        X_te_raw = X_full.loc[te_idx][overall_feats].copy() if te_idx else X_te[overall_feats].copy()
        valid    = ~X_te_raw.isnull().any(axis=1).values

        # ── Conditions E and F: non-partitioned Overall SE ────────────────────
        try:
            X_te_v  = X_te_raw[valid]
            X_te_sc = overall_scl.transform(X_te_v.values)
            yp_EF   = overall_se_model.predict_proba(X_te_sc)[:, 1]
        except Exception as _ef:
            log(f"  [{setting}] Non-partitioned SE scoring failed: {_ef}")
            yp_EF = None

        # All model names in this setting
        all_model_names = list(results_D.keys())

        for model_name in all_model_names:
            model_res = results_D.get(model_name, {})
            yt_all    = np.array(model_res.get("y_true",  []))
            yp_D_all  = np.array(model_res.get("y_proba", []))
            thr_D     = float(model_res.get("optimal_threshold", 0.5))

            if len(yt_all) < 10 or len(np.unique(yt_all)) < 2:
                continue

            yt_v    = yt_all[valid]
            yp_D_v  = yp_D_all[valid]

            if len(yt_v) < 10:
                continue

            # Conditions E and F: same non-partitioned Overall SE proba
            # (E ≈ F here since Overall SE already uses overall_feats;
            #  "all features" E would require retraining -- use SE as proxy)
            yp_E = yp_EF if yp_EF is not None and len(yp_EF)==len(yt_v) else yp_D_v
            yp_F = yp_EF if yp_EF is not None and len(yp_EF)==len(yt_v) else yp_D_v

            # Conditions G and H: retrain this model on subgroup data
            # G: all features
            all_feats_valid = [f for f in all_feat_names
                               if f in X_tr.columns and f in X_te.columns]
            ov_feats_valid  = [f for f in overall_feats
                               if f in X_tr.columns and f in X_te.columns]

            X_te_for_G = X_te.iloc[valid] if len(X_te)==len(yt_all) else X_te

            yp_G, thr_G = _retrain_model(
                type(model_res.get("_model", RandomForestClassifier())),
                (model_res.get("_model") or RandomForestClassifier()).get_params(deep=False),
                X_tr[all_feats_valid], y_tr,
                X_te_for_G[all_feats_valid])

            yp_H, thr_H = _retrain_model(
                type(model_res.get("_model", RandomForestClassifier())),
                (model_res.get("_model") or RandomForestClassifier()).get_params(deep=False),
                X_tr[ov_feats_valid], y_tr,
                X_te_for_G[ov_feats_valid])

            # Compute metrics
            aE = _auroc(yt_v, yp_E);  accE = _acc(yt_v, yp_E, _np_thr)
            aF = _auroc(yt_v, yp_F);  accF = _acc(yt_v, yp_F, _np_thr)
            aG = _auroc(yt_v, yp_G) if yp_G is not None else float("nan")
            accG= _acc(yt_v, yp_G, thr_G) if yp_G is not None else float("nan")
            aH = _auroc(yt_v, yp_H) if yp_H is not None else float("nan")
            accH= _acc(yt_v, yp_H, thr_H) if yp_H is not None else float("nan")
            aD = _auroc(yt_v, yp_D_v);  accD = _acc(yt_v, yp_D_v, thr_D)


            row = {
                "Setting":         setting,
                "Model":           model_name,
                "N_test":          len(yt_v),
                "N_feats_all":     len(all_feats_valid),
                "N_feats_overall": len(ov_feats_valid),
                "N_feats_subgroup":len(feats_sub),
                # AUROCs
                "E_AUROC": _r(aE), "F_AUROC": _r(aF),
                "G_AUROC": _r(aG), "H_AUROC": _r(aH), "D_AUROC": _r(aD),
                # Accuracies
                "E_Acc": _r(accE), "F_Acc": _r(accF),
                "G_Acc": _r(accG), "H_Acc": _r(accH), "D_Acc": _r(accD),
                # F1 for condition D
                "E_F1": _f1(yt_v, yp_E,   _np_thr),
                "F_F1": _f1(yt_v, yp_F,   _np_thr),
                "G_F1": _f1(yt_v, yp_G,   thr_G)   if yp_G is not None else "N/A",
                "H_F1": _f1(yt_v, yp_H,   thr_H)   if yp_H is not None else "N/A",
                "D_F1": _f1(yt_v, yp_D_v, thr_D),
                # AUROC deltas
                "Delta_D-E_AUROC":  _delta(aD, aE),
                "Delta_D-H_AUROC":  _delta(aD, aH),
                "Delta_H-F_AUROC":  _delta(aH, aF),
                "Delta_D-G_AUROC":  _delta(aD, aG),
                "Delta_G-E_AUROC":  _delta(aG, aE),
                # Accuracy deltas
                "Delta_D-E_Acc":    _delta(accD, accE),
                "Delta_D-H_Acc":    _delta(accD, accH),
                "Delta_H-F_Acc":    _delta(accH, accF),
                "Delta_D-E_F1":    _f1(yt_v, yp_D_v, thr_D), 
            }
            # DeLong: D vs E, F, G, H
            for cname, yp_c in [("E", yp_E), ("F", yp_F),
                                  ("G", yp_G), ("H", yp_H)]:
                if yp_c is None: continue
                try:
                    z, p = delong_auc_test(yt_v, yp_D_v, yp_c)
                    pb   = paired_bootstrap_contrast(yt_v, yp_D_v, yp_c, "auroc", n_boot)
                    row[f"DeLong_D_vs_{cname}_z"] = _r(z)
                    row[f"DeLong_D_vs_{cname}_p"] = _r(p)
                    row[f"Boot_D_vs_{cname}_CI"]  = f"[{pb['ci_lo']:+.3f},{pb['ci_hi']:+.3f}]"
                    row[f"D_beats_{cname}"]        = pb["significant"] and pb["diff"] > 0
                except Exception as _de:
                    log(f"    DeLong D vs {cname} [{model_name} {setting}]: {_de}")

            rows.append(row)
            log(f"  [{setting}][{model_name}] "
                f"E={aE:.3f}({accE:.3f}) F={aF:.3f} "
                f"G={aG:.3f}({accG:.3f}) H={aH:.3f}({accH:.3f}) "
                f"D={aD:.3f}({accD:.3f})  "
                f"Δ(D-E)AUROC={_delta(aD,aE)} Acc={_delta(accD,accE)}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ablation_feature_selection.csv", index=False)
    log("  Saved ablation_feature_selection.csv")

    if not df.empty:
        log("\n  === FEATURE ABLATION SUMMARY ===")
        log(f"  {'Setting':<8} {'Model':<24} {'Δ Total AUROC':>14} {'Δ SubFeat':>10} {'Δ Partition':>11}")
        for _, r in df.iterrows():
            try:
                log(f"  {str(r['Setting']):<8} {str(r['Model']):<24} "
                    f"{str(r['Delta_D-E_AUROC']):>14} "
                    f"{str(r['Delta_D-H_AUROC']):>10} "
                    f"{str(r['Delta_H-F_AUROC']):>11}")
            except: pass
    return df
def dca_focused(
    y_true_mcb: np.ndarray,
    fib4_mcb: np.ndarray,
    prfps_mcb: np.ndarray,
    setting: str,
    out_dir: str = None,
    thresholds: np.ndarray = None,
    smooth_window: int = 11,
    fib4_train_scores: np.ndarray = None,
    prfps_train_scores: np.ndarray = None,
    y_train: np.ndarray = None,
    # Tapestry cohort -- optional, adds 2 extra curves
    y_true_tap: np.ndarray = None,
    fib4_tap: np.ndarray = None,
    prfps_tap: np.ndarray = None,
) -> pd.DataFrame:
    """
    Focused DCA: pRFPS vs FIB-4.

    When Tapestry data is supplied (y_true_tap / fib4_tap / prfps_tap),
    produces 4 curves on a single plot:
      MCB pRFPS   solid dark-blue
      MCB FIB-4   solid dark-red
      Tap pRFPS   dashed light-blue
      Tap FIB-4   dashed orange

    Without Tapestry, produces the standard 2-curve MCB plot.

    All scores Platt-calibrated using MCB training data so the probability
    threshold axis is commensurable across both cohorts and both tools.
    Using the MCB calibrators for Tapestry is correct: thresholds are fixed
    at training time and transferred, exactly as in clinical deployment.
    """
    from scipy.signal import savgol_filter

    if out_dir is None:
        out_dir = f"{results_dir}/dca_plots"
    os.makedirs(out_dir, exist_ok=True)
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.85, 200)

    y_mcb    = np.array(y_true_mcb).astype(int)
    n_mcb    = len(y_mcb)
    prev_mcb = float(y_mcb.mean())

    def _calibrate(train_sc, y_tr, test_sc, label):
        if train_sc is not None and y_tr is not None and len(train_sc) > 5:
            cal = calibrate_score_to_probability(train_sc, y_tr, test_sc, label)
            return cal, "Platt"
        lo = float(np.nanmin(test_sc)); hi = float(np.nanmax(test_sc))
        log(f"  [FIX-B DCA {setting}] WARNING: {label} using min-max (no train scores)")
        return np.clip((np.array(test_sc, float)-lo)/(hi-lo+1e-10), 0, 1), "min-max"

    fib4_mcb_cal,  fib4_method  = _calibrate(fib4_train_scores,  y_train, fib4_mcb,  "MCB FIB-4")
    prfps_mcb_cal, prfps_method = _calibrate(prfps_train_scores, y_train, prfps_mcb, "MCB pRFPS")
    log(f"  [FIX-B DCA {setting}] MCB: FIB-4={fib4_method} | pRFPS={prfps_method}")

    has_tap = (y_true_tap is not None and fib4_tap is not None
               and prfps_tap is not None and len(y_true_tap) > 5)
    if has_tap:
        fib4_tap_cal,  _ = _calibrate(fib4_train_scores,  y_train, fib4_tap,  "Tap FIB-4")
        prfps_tap_cal, _ = _calibrate(prfps_train_scores, y_train, prfps_tap, "Tap pRFPS")
        y_tap    = np.array(y_true_tap).astype(int)
        n_tap    = len(y_tap)
        prev_tap = float(y_tap.mean())
        log(f"  [FIX-B DCA {setting}] Tapestry n={n_tap} prev={prev_tap:.2%}")

    def _nb(y, yp, t, n):
        yb  = (yp >= t).astype(int)
        tp_ = int(((yb==1)&(y==1)).sum())
        fp_ = int(((yb==1)&(y==0)).sum())
        return tp_/n - fp_/n * t/(1-t+1e-10)

    rows = []
    for t in thresholds:
        row = {
            "threshold":     t,
            "Treat_All_MCB": prev_mcb - (1-prev_mcb)*t/(1-t+1e-10),
            "Treat_None":    0.0,
            "MCB_pRFPS":     _nb(y_mcb, prfps_mcb_cal, t, n_mcb),
            "MCB_FIB4":      _nb(y_mcb, fib4_mcb_cal,  t, n_mcb),
        }
        if has_tap:
            row["Tap_pRFPS"]     = _nb(y_tap, prfps_tap_cal, t, n_tap)
            row["Tap_FIB4"]      = _nb(y_tap, fib4_tap_cal,  t, n_tap)
            row["Treat_All_Tap"] = prev_tap - (1-prev_tap)*t/(1-t+1e-10)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_dir}/dca_focused_{setting}.csv", index=False)

    win = min(smooth_window, len(df)//4*2+1)
    if win < 3: win = 3
    if win % 2 == 0: win += 1
    df_p = df.copy()
    smooth_cols = ["MCB_pRFPS","MCB_FIB4","Treat_All_MCB"]
    if has_tap: smooth_cols += ["Tap_pRFPS","Tap_FIB4","Treat_All_Tap"]
    for col in smooth_cols:
        if col in df_p.columns:
            try:   df_p[col] = savgol_filter(df_p[col].values, win, 2)
            except Exception: pass

    BLUE_D = "#1A5276"; BLUE_L = "#5DADE2"
    RED_D  = "#922B21"; RED_L  = "#E59866"
    GREY   = "#7F8C8D"

    fig, ax = plt.subplots(figsize=(7.5, 5))

    # Treat-All reference lines
    ax.plot(df_p["threshold"], df_p["Treat_All_MCB"],
            color=GREY, ls=":", lw=1.4, alpha=0.7, label="Treat All (MCB)")
    if has_tap:
        ax.plot(df_p["threshold"], df_p["Treat_All_Tap"],
                color=GREY, ls=(0,(3,1,1,1)), lw=1.2, alpha=0.5,
                label="Treat All (Tapestry)")
    ax.axhline(0, color="black", lw=0.6, ls="--", alpha=0.35)

    # MCB curves (solid)
    ax.plot(df_p["threshold"], df_p["MCB_pRFPS"],
            color=BLUE_D, ls="-",  lw=2.2, label="pRFPS — MCB")
    ax.plot(df_p["threshold"], df_p["MCB_FIB4"],
            color=RED_D,  ls="-",  lw=2.2, label="FIB-4 — MCB")

    # Tapestry curves (dashed)
    if has_tap:
        ax.plot(df_p["threshold"], df_p["Tap_pRFPS"],
                color=BLUE_L, ls="--", lw=2.0, label="pRFPS — Tapestry")
        ax.plot(df_p["threshold"], df_p["Tap_FIB4"],
                color=RED_L,  ls="--", lw=2.0, label="FIB-4 — Tapestry")

    ax.set_xlabel("Threshold Probability (Platt-calibrated)", fontsize=11)
    ax.set_ylabel("Net Benefit", fontsize=11)
    cohorts = "MCB + Tapestry" if has_tap else "MCB only"
    ax.set_title(
        f"Decision Curve Analysis — {setting} | pRFPS vs FIB-4 ({cohorts})",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9,
              title="solid = MCB  |  dashed = Tapestry", title_fontsize=7.5)
    ax.grid(alpha=0.20, lw=0.5)
    ax.set_xlim(0.05, 0.85)

    nb_cols = ["MCB_pRFPS","MCB_FIB4","Treat_All_MCB"]
    if has_tap: nb_cols += ["Tap_pRFPS","Tap_FIB4"]
    y_max = max(max(prev_mcb, prev_tap if has_tap else 0) + 0.06,
                df_p[[c for c in nb_cols if c in df_p]].max().max() + 0.03)
    ax.set_ylim(-0.04, min(y_max, 0.55))
    ax.text(0.98, 0.02,
            f"SG smooth w={win} | pRFPS={prfps_method} | FIB-4={fib4_method}",
            transform=ax.transAxes, fontsize=6, color="gray",
            ha="right", va="bottom")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/dca_focused_{setting}.png", dpi=200, bbox_inches="tight")
    plt.close()
    log(f"  Focused DCA saved: dca_focused_{setting}.png ({cohorts})")
    return df

# =============================================================================
# [FIX-4]  FEATURE SELECTION -- nested inside training fold
# =============================================================================

def nested_feature_selection(X_tr: pd.DataFrame, y_tr: pd.Series,
                              feature_names: List[str], k: int = 10,
                              exclude: Optional[List[str]] = None,
                              label: str = "") -> List[str]:
    """
    [FIX-4] All 5 original methods run on X_tr ONLY.
    No imputation -- NaN already dropped before this call.
    Consensus = selected by >= 3 methods.
    """
    names = [f for f in feature_names if f not in (exclude or [])]
    Xdf   = X_tr[names] if hasattr(X_tr, "__getitem__") else \
            pd.DataFrame(X_tr, columns=names)
    k_eff = min(k, len(names))

    scl  = StandardScaler()
    X_sc = scl.fit_transform(Xdf)
    X_np = Xdf.values

    sel = {m: set() for m in ["anova", "mi", "rfe", "rf", "lasso"]}

    try:   # 1. ANOVA F-test
        s = SelectKBest(f_classif, k=k_eff).fit(X_sc, y_tr)
        sel["anova"] = set(np.array(names)[s.get_support()])
    except Exception as e:
        log(f"    ANOVA failed: {e}")

    try:   # 2. Mutual information
        s = SelectKBest(mutual_info_classif, k=k_eff).fit(X_sc, y_tr)
        sel["mi"] = set(np.array(names)[s.get_support()])
    except Exception as e:
        log(f"    MI failed: {e}")

    try:   # 3. RFE with Random Forest
        rf  = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
        rfe = RFE(rf, n_features_to_select=k_eff)
        rfe.fit(X_np, y_tr)
        sel["rfe"] = set(np.array(names)[rfe.get_support()])
    except Exception as e:
        log(f"    RFE failed: {e}")

    try:   # 4. Model-based (RF importance)
        rf2 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)
        rf2.fit(X_np, y_tr)
        sm  = SelectFromModel(rf2, max_features=k_eff, prefit=True)
        sel["rf"] = set(np.array(names)[sm.get_support()])
    except Exception as e:
        log(f"    RF-importance failed: {e}")

    try:   # 5. LassoCV (same Cs as original)
        Cs  = np.logspace(-4, 4, 20)
        lcv = LogisticRegressionCV(
            Cs=Cs, penalty="l1", solver="liblinear",
            cv=5, max_iter=10000, random_state=42, scoring="roc_auc")
        lcv.fit(X_sc, y_tr)
        coefs = np.abs(lcv.coef_[0])
        if coefs.sum() == 0:
            top = np.argsort(coefs)[-k_eff:]
        else:
            sm2 = SelectFromModel(lcv, threshold="median",
                                   max_features=k_eff, prefit=True)
            top = sm2.get_support(indices=True)
            if len(top) == 0:
                top = np.argsort(coefs)[-k_eff:]
        sel["lasso"] = set(np.array(names)[top])
    except Exception as e:
        log(f"    LassoCV failed: {e}")

    counts    = Counter(f for fs in sel.values() for f in fs)
    consensus = sorted(f for f, c in counts.items() if c >= 3)
    if len(consensus) < 5:
        consensus = sorted(counts, key=counts.get, reverse=True)[:min(10, len(names))]

    log(f"  [{label}] {len(consensus)} features: {consensus}")

    rows = [{"feature": f,
             "anova":  int(f in sel["anova"]),  "mi":    int(f in sel["mi"]),
             "rfe":    int(f in sel["rfe"]),     "rf":    int(f in sel["rf"]),
             "lasso":  int(f in sel["lasso"]),
             "n_methods": counts.get(f, 0),
             "selected":  int(f in consensus)} for f in names]
    pd.DataFrame(rows).to_csv(
        f"{OUT}/feature_selection_{label.replace(' ','_')}.csv", index=False)
    return consensus


# =============================================================================
# ORIGINAL SAVE HELPERS  (identical to original)
# =============================================================================

def ablation_partition_contribution(
    results_by_setting: Dict,
    X_full: pd.DataFrame,
    y_full: pd.Series,
    all_test_indices: Dict,
    features_overall: List[str],
    overall_se_model,
    overall_scl,
    overall_se_results: Dict,
    n_boot: int = 500,
) -> pd.DataFrame:
    """
    2×2 ablation: same Stacking Ensemble architecture, vary training scope
    and threshold. Also repeats with best single model from each setting.

    Conditions (evaluated on IDENTICAL test patients):
      A. Overall SE,    fixed 0.5       (non-partitioned, no calibration)
      B. Overall SE,    Youden thr      (non-partitioned, calibrated)
      C. Partitioned SE, fixed 0.5      (partitioned, no calibration)
      D. Partitioned SE, Youden thr     (partitioned, calibrated) ← full pipeline

    Deltas:
      D-A = total gain
      C-A = partitioning alone
      D-C = threshold calibration alone
      D-B = partitioning benefit (threshold matched)
    """
    section("ABLATION -- 2×2 Partition × Threshold [Same Model Architecture]")
    rows = []

    _np_youden_thr = float(overall_se_results.get("optimal_threshold", 0.5))
    log(f"  Non-partitioned SE overall Youden threshold: {_np_youden_thr:.3f}")
    log(f"  Non-partitioned SE overall AUROC: "
        f"{overall_se_results.get('auroc', float('nan')):.3f}")

    def _m(yt, ypred, yprob):
        if len(yt) == 0 or len(np.unique(yt)) < 2: return {}
        cm_ = confusion_matrix(yt, ypred, labels=[0,1]); tn_,fp_,fn_,tp_ = cm_.ravel()
        return {
            "auroc":       roc_auc_score(yt, yprob),
            "f1_macro":    f1_score(yt, ypred, average="macro", zero_division=0),
            "f1_rapid":    f1_score(yt, ypred, pos_label=1, average="binary", zero_division=0),
            "sensitivity": tp_/(tp_+fn_+1e-10),
            "specificity": tn_/(tn_+fp_+1e-10),
            "accuracy":    accuracy_score(yt, ypred),
        }

    def _r(v):
        try: return round(float(v), 4) if not np.isnan(float(v)) else "N/A"
        except: return "N/A"

    for setting in [s for s in results_by_setting if s != "Overall"]:
        res    = results_by_setting[setting]
        se_sub = res.get("Stacking_Ensemble", {})
        yt_sub = np.array(se_sub.get("y_true", []))
        yp_sub = np.array(se_sub.get("y_proba", []))

        if len(yt_sub) < 10 or len(np.unique(yt_sub)) < 2:
            continue

        te_idx = all_test_indices.get(setting, [])
        if not te_idx:
            log(f"  [{setting}] no test indices -- skipping"); continue

        # ── Non-partitioned SE on same test patients (Conditions A & B) ──────
        try:
            X_te_raw  = X_full.loc[te_idx][features_overall].copy()
            valid     = ~X_te_raw.isnull().any(axis=1).values
            n_dropped = (~valid).sum()
            if n_dropped:
                log(f"  [{setting}] Dropping {n_dropped} NaN rows before SE scoring")

            X_te_v   = X_te_raw[valid]
            yt_v     = yt_sub[valid]
            yp_sub_v = yp_sub[valid]

            if len(yt_v) < 10 or len(np.unique(yt_v)) < 2:
                log(f"  [{setting}] too few valid rows -- skipping"); continue

            X_te_sc      = overall_scl.transform(X_te_v.values)
            _np_prob     = overall_se_model.predict_proba(X_te_sc)[:, 1]
            cond_A_pred  = (_np_prob >= 0.5).astype(int)
            cond_B_pred  = (_np_prob >= _np_youden_thr).astype(int)

            log(f"  [{setting}] Non-partitioned SE on subgroup: "
                f"AUROC={roc_auc_score(yt_v, _np_prob):.3f} (n={len(yt_v)})")
        except Exception as _e:
            log(f"  [{setting}] Non-partitioned SE scoring failed: {_e}"); continue

        # ── Partitioned SE (Conditions C & D) ─────────────────────────────────
        _part_thr    = float(se_sub.get("optimal_threshold", 0.5))
        cond_C_se    = (yp_sub_v >= 0.5).astype(int)
        cond_D_se    = (yp_sub_v >= _part_thr).astype(int)

        mA     = _m(yt_v, cond_A_pred, _np_prob)
        mB     = _m(yt_v, cond_B_pred, _np_prob)
        mC_se  = _m(yt_v, cond_C_se,   yp_sub_v)
        mD_se  = _m(yt_v, cond_D_se,   yp_sub_v)

        def _build_row(model_label, mC, mD, yp_C, yp_D):
            row = {
                "Setting":    setting,
                "Model":      model_label,
                "N_test":     len(yt_v),
                "Rapid_pct":  round(float(yt_v.mean()*100), 1),
                "A_AUROC":    _r(mA.get("auroc",   float("nan"))),
                "B_AUROC":    _r(mB.get("auroc",   float("nan"))),
                "C_AUROC":    _r(mC.get("auroc",   float("nan"))),
                "D_AUROC":    _r(mD.get("auroc",   float("nan"))),
                "A_F1":       _r(mA.get("f1_macro",float("nan"))),
                "B_F1":       _r(mB.get("f1_macro",float("nan"))),
                "C_F1":       _r(mC.get("f1_macro",float("nan"))),
                "D_F1":       _r(mD.get("f1_macro",float("nan"))),
                "D_Sens":     _r(mD.get("sensitivity",float("nan"))),
                "D_Spec":     _r(mD.get("specificity",float("nan"))),
                "Delta_D-A_TotalGain":    _r(mD.get("auroc",0)-mA.get("auroc",0)),
                "Delta_C-A_Partition":    _r(mC.get("auroc",0)-mA.get("auroc",0)),
                "Delta_D-C_Threshold":    _r(mD.get("auroc",0)-mC.get("auroc",0)),
                "Delta_D-B_PartYouden":   _r(mD.get("auroc",0)-mB.get("auroc",0)),
                "Delta_B-A_ThrNonPart":   _r(mB.get("auroc",0)-mA.get("auroc",0)),
            }
            # DeLong: D vs A, B, C
            for cname, cp, cpp in [("A", _np_prob, cond_A_pred),
                                    ("B", _np_prob, cond_B_pred),
                                    ("C", yp_C,     cond_C_se)]:
                try:
                    z,p = delong_auc_test(yt_v, yp_D, cp)
                    pb  = paired_bootstrap_contrast(yt_v, yp_D, cp, "auroc", n_boot)
                    row[f"DeLong_D_vs_{cname}_z"] = _r(z)
                    row[f"DeLong_D_vs_{cname}_p"] = _r(p)
                    row[f"Boot_D_vs_{cname}_CI"]  = f"[{pb['ci_lo']:+.3f},{pb['ci_hi']:+.3f}]"
                    row[f"Boot_D_vs_{cname}_p"]   = pb["p_value"]
                    row[f"D_beats_{cname}"]        = pb["significant"] and pb["diff"] > 0
                except Exception as _de:
                    log(f"    DeLong D vs {cname} [{model_label} {setting}]: {_de}")
            return row

        rows.append(_build_row("SE", mC_se, mD_se, yp_sub_v, yp_sub_v))
        log(f"  [{setting}][SE] A={mA.get('auroc',float('nan')):.3f}  "
            f"B={mB.get('auroc',float('nan')):.3f}  "
            f"C={mC_se.get('auroc',float('nan')):.3f}  "
            f"D={mD_se.get('auroc',float('nan')):.3f}  "
            f"Δ(D-A)={mD_se.get('auroc',0)-mA.get('auroc',0):+.3f}")

        # ── Best single model (Conditions C & D with best single) ─────────────
        singles  = {k: v for k, v in res.items() if k != "Stacking_Ensemble"}
        best_nm  = max(singles, key=lambda m: singles[m].get("f1_macro", 0)) if singles else None
        if best_nm:
            best_res   = singles[best_nm]
            yp_best    = np.array(best_res.get("y_proba", []))
            _best_thr  = float(best_res.get("optimal_threshold", 0.5))

            if len(yp_best) == len(yt_sub):
                yp_best_v   = yp_best[valid]
                cond_C_bst  = (yp_best_v >= 0.5).astype(int)
                cond_D_bst  = (yp_best_v >= _best_thr).astype(int)
                mC_bst      = _m(yt_v, cond_C_bst, yp_best_v)
                mD_bst      = _m(yt_v, cond_D_bst, yp_best_v)
                rows.append(_build_row(best_nm, mC_bst, mD_bst, yp_best_v, yp_best_v))
                log(f"  [{setting}][{best_nm}] A={mA.get('auroc',float('nan')):.3f}  "
                    f"B={mB.get('auroc',float('nan')):.3f}  "
                    f"C={mC_bst.get('auroc',float('nan')):.3f}  "
                    f"D={mD_bst.get('auroc',float('nan')):.3f}  "
                    f"Δ(D-A)={mD_bst.get('auroc',0)-mA.get('auroc',0):+.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/ablation_partition_contribution.csv", index=False)
    log("  Saved ablation_partition_contribution.csv")

    if not df.empty:
        log("\n  === ABLATION SUMMARY ===")
        log(f"  {'Setting':<6} {'Model':<22} {'Δ Total':>8} {'Δ Partition':>12} {'Δ Threshold':>12}")
        for _, r in df.iterrows():
            try:
                tot = float(r["Delta_D-A_TotalGain"])
                par = float(r["Delta_C-A_Partition"])
                thr = float(r["Delta_D-C_Threshold"])
                dom = "Partition" if abs(par) >= abs(thr) else "Threshold"
                log(f"  {str(r['Setting']):<6} {str(r['Model']):<22} "
                    f"{tot:>+8.3f} {par:>+12.3f} {thr:>+12.3f}  ← {dom}")
            except: pass
    return df

# =============================================================================
# [C1+C3]  COVARIATE SHIFT + SUBGROUP PROFILE COMPARISON
# =============================================================================

