"""
validate.py — External validation (Tapestry), pRFPS + FIB-4 comparison.
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
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
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

def validate_external(setting_artifacts,
                       X_val, y_val_num, subgroups_val,
                       df_val_raw=None,
                       all_prfps_results=None,
                       mcb_fib4_by_setting=None,
                       mcb_train_data_by_setting=None,
                       n_boot=500):
    """
    External validation on Tapestry using setting-specific MCB artifacts.

    Critical rules:
      - ML models use setting-specific features + setting-specific scaler.
      - pRFPS uses raw/processed feature values, NOT scaled values.
      - FIB-4 uses raw Tapestry data.
      - C1 uses C1 model/scaler/features/pRFPS formula.
      - C2 uses C2 model/scaler/features/pRFPS formula.
      - Overall uses Overall model/scaler/features/pRFPS formula.
    """
    section("EXTERNAL VALIDATION -- Tapestry")

    val_settings = ["Overall"] + sorted(subgroups_val.dropna().unique().tolist())
    ext_rows = []
    cal_rows = []
    pf_rows = []
    dl_rows = []
    _tap_prfps_by_setting = {}
    _tap_fib4_by_setting  = {}

    tap_out = f"{results_dir}/patient_level/Tapestry"
    os.makedirs(tap_out, exist_ok=True)

    def _raw_subset(idx):
        if df_val_raw is None:
            return None
        try:
            return df_val_raw.loc[idx].copy().reset_index(drop=True)
        except Exception:
            return df_val_raw.iloc[list(idx)].copy().reset_index(drop=True)

    def _metrics(yt, ypred, yscore):
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

    for vsetting in val_settings:
        artifact = setting_artifacts.get(vsetting, setting_artifacts.get("Overall"))
        if artifact is None:
            log(f"  [{vsetting}] No MCB artifact available -- skipping")
            continue

        models_s  = artifact["models"]
        results_s = artifact["results"]
        scl_s     = artifact["scaler"]
        feats_s   = artifact["features"]

        missing_ml = [f for f in feats_s if f not in X_val.columns]
        if missing_ml:
            log(f"  [{vsetting}] Missing ML features: {missing_ml} -- skipping")
            continue

        idx_s = X_val.index if vsetting == "Overall"                 else X_val.index[subgroups_val == vsetting]

        if len(idx_s) < 20:
            log(f"  [{vsetting}] too few ({len(idx_s)}) -- skip")
            continue

        Xv_raw_s = X_val.loc[idx_s, feats_s].copy()
        yv_s     = y_val_num.loc[idx_s].copy()
        sg_s     = subgroups_val.loc[idx_s].copy()

        complete_mask = ~Xv_raw_s.isnull().any(axis=1)
        if (~complete_mask).sum() > 0:
            log(f"  [{vsetting}] Dropping {(~complete_mask).sum()} NaN rows")

        Xv_raw_s = Xv_raw_s.loc[complete_mask]
        yv_s     = yv_s.loc[complete_mask]
        sg_s     = sg_s.loc[complete_mask]
        idx_s    = Xv_raw_s.index

        if len(yv_s) < 20 or len(np.unique(yv_s.values)) < 2:
            log(f"  [{vsetting}] insufficient data after filtering -- skip")
            continue

        Xv_np_s = scl_s.transform(Xv_raw_s[feats_s].values)
        y_arr   = np.array(yv_s).astype(int).flatten()

        log(f"  [{vsetting}] Tapestry n={len(y_arr)} using {len(feats_s)} features")

        # ── ML model validation ───────────────────────────────────────────────
        model_probs = {}
        for nm, mdl in models_s.items():
            try:
                pv  = mdl.predict_proba(Xv_np_s)[:, 1]
                thr = results_s[nm]["optimal_threshold"]
                pd_ = (pv >= thr).astype(int)
                pt, ci = compute_comprehensive_metrics(y_arr, pd_, pv, n_boot=n_boot)
                def fmt(m): return fmt_ci(m, ci, pt)
                ext_rows.append({
                    "Val_Setting": vsetting, "Model": nm,
                    "Threshold": f"{thr:.3f}",
                    "Accuracy": fmt("accuracy"),   "F1_macro": fmt("f1_macro"),
                    "AUROC":    fmt("auroc"),       "AUPRC":    fmt("auprc"),
                    "Brier":    fmt("brier"),       "Sensitivity": fmt("sensitivity"),
                    "Specificity": fmt("specificity"), "PPV": fmt("ppv"), "NPV": fmt("npv"),
                    f"Precision_{NEGATIVE_LABEL}": fmt("precision_0"),
                    f"Recall_{NEGATIVE_LABEL}":    fmt("recall_0"),
                    f"F1_{NEGATIVE_LABEL}":        fmt("f1_0"),
                    f"Precision_{POSITIVE_LABEL}": fmt("precision_1"),
                    f"Recall_{POSITIVE_LABEL}":    fmt("recall_1"),
                    f"F1_{POSITIVE_LABEL}":        fmt("f1_1"),
                })
                model_probs[nm] = pv
                log(f"  [{vsetting}][{nm}] AUROC={pt['auroc']:.3f}")
                cal_rows.append(calibration_analysis(
                    y_arr, pv, nm, f"Tapestry_{vsetting}",
                    out_dir=f"{results_dir}/calibration_plots"))
            except Exception as e:
                log(f"  [{vsetting}][{nm}] failed: {e}")

        # ── pRFPS on Tapestry (raw values, MCB formula, no refitting) ─────────
        prfps_res_tap = None
        if all_prfps_results is not None:
            mcb_formula = (all_prfps_results.get(vsetting)
                           or all_prfps_results.get("Overall"))
            if mcb_formula is not None:
                try:
                    top_feats = mcb_formula["feature_table"]["feature"].tolist()
                    missing_p = [f for f in top_feats if f not in Xv_raw_s.columns]
                    if missing_p:
                        log(f"  [pRFPS-Tap {vsetting}] Missing: {missing_p}")
                    prfps_res_tap = apply_prfps_formula(
                        mcb_formula, Xv_raw_s, y_arr, vsetting)
                    log(f"  [{vsetting}] pRFPS-Tap: "
                        f"n_features={prfps_res_tap['n_features']} "
                        f"cutoff={prfps_res_tap['cutoff_int']:.0f} (MCB-derived)")
                except Exception as e:
                    log(f"  [{vsetting}] pRFPS-Tap failed: {e}")

        # ── FIB-4 on Tapestry ─────────────────────────────────────────────────
        fib4_res_tap = None
        fib4_cont    = None
        df_sub_raw   = _raw_subset(idx_s)

        if df_sub_raw is not None and (
            "first_FIB4" in df_sub_raw.columns
            or all(c in df_sub_raw.columns
                   for c in ["AST.x", "ALT.x", "PLATELET_COUNT", "diagnosis_age"])
        ):
            try:
                fib4_cont = compute_fib4_score(df_sub_raw)
                fib4_age  = (df_sub_raw["diagnosis_age"].astype(float)
                             if "diagnosis_age" in df_sub_raw.columns
                             else pd.Series(50.0, index=df_sub_raw.index))
                fib4_bin  = fib4_binary_label(fib4_cont, fib4_age)
                n_low = int((fib4_bin==0).sum())
                n_hi  = int((fib4_bin==1).sum())
                n_int = int(fib4_bin.isna().sum())
                log(f"  FIB-4 Tap [{vsetting}]: low={n_low} high={n_hi} excl={n_int}")
                fib4_res_tap = {
                    "fib4_continuous": fib4_cont.values.astype(float),
                    "fib4_binary":     fib4_bin.values,
                    "n_low": n_low, "n_high": n_hi, "n_intermediate": n_int,
                }
            except Exception as e:
                log(f"  FIB-4 Tap [{vsetting}] failed: {e}")
        else:
            log(f"  FIB-4 Tap [{vsetting}]: required columns unavailable")
        
        # ── pRFPS vs FIB-4 comparison (EASL-eligible only) ───────────────────
        if prfps_res_tap is not None and fib4_res_tap is not None:
            try:
                fib4_binary_arr = fib4_res_tap["fib4_binary"]
                fib4_score_arr  = fib4_res_tap["fib4_continuous"]
                prfps_score_arr = prfps_res_tap["score_cont"]
                prfps_int_arr   = prfps_res_tap["score_int"]
                prfps_cutoff    = prfps_res_tap["cutoff_int"]
                eligible        = ~np.isnan(fib4_binary_arr)

                if eligible.sum() < 10:
                    log(f"  [{vsetting}] Too few EASL-eligible -- skip comparison")
                else:
                    yt_e         = y_arr[eligible]
                    fib4_pred_e  = fib4_binary_arr[eligible].astype(int)
                    fib4_cont_e  = fib4_score_arr[eligible]
                    prfps_pred_e = (prfps_int_arr[eligible] >= prfps_cutoff).astype(int)
                    prfps_cont_e = prfps_score_arr[eligible]
                    pm = _metrics(yt_e, prfps_pred_e, prfps_cont_e)
                    fm = _metrics(yt_e, fib4_pred_e,  fib4_cont_e)

                    pf_rows.append({
                        "Cohort": "Tapestry", "Setting": vsetting,
                        "N_EASL_eligible": int(eligible.sum()),
                        "N_intermediate_excl": int((~eligible).sum()),
                        "pRFPS_cutoff_MCB": round(float(prfps_cutoff), 3),
                        "FIB4_low_thresh_lt65": 1.3, "FIB4_low_thresh_gte65": 2.0,
                        "FIB4_high_thresh": 2.67,
                        "pRFPS_AUROC":       round(float(pm.get("auroc",   float("nan"))), 4),
                        "pRFPS_AUPRC":       round(float(pm.get("auprc",   float("nan"))), 4),
                        "pRFPS_Accuracy":    round(float(pm.get("accuracy", float("nan"))), 4),
                        "pRFPS_F1_macro":    round(float(pm.get("f1_macro", float("nan"))), 4),
                        "pRFPS_F1_Rapid":    round(float(pm.get("f1_rapid", float("nan"))), 4),
                        "pRFPS_Sensitivity": round(float(pm.get("sensitivity", float("nan"))), 4),
                        "pRFPS_Specificity": round(float(pm.get("specificity", float("nan"))), 4),
                        "pRFPS_PPV": round(float(pm.get("ppv", float("nan"))), 4),
                        "pRFPS_NPV": round(float(pm.get("npv", float("nan"))), 4),
                        "pRFPS_TP": pm.get("tp",""), "pRFPS_FP": pm.get("fp",""),
                        "pRFPS_TN": pm.get("tn",""), "pRFPS_FN": pm.get("fn",""),
                        "FIB4_AUROC":        round(float(fm.get("auroc",   float("nan"))), 4),
                        "FIB4_AUPRC":        round(float(fm.get("auprc",   float("nan"))), 4),
                        "FIB4_Accuracy":     round(float(fm.get("accuracy", float("nan"))), 4),
                        "FIB4_F1_macro":     round(float(fm.get("f1_macro", float("nan"))), 4),
                        "FIB4_F1_Rapid":     round(float(fm.get("f1_rapid", float("nan"))), 4),
                        "FIB4_Sensitivity":  round(float(fm.get("sensitivity", float("nan"))), 4),
                        "FIB4_Specificity":  round(float(fm.get("specificity", float("nan"))), 4),
                        "FIB4_PPV": round(float(fm.get("ppv", float("nan"))), 4),
                        "FIB4_NPV": round(float(fm.get("npv", float("nan"))), 4),
                        "FIB4_TP": fm.get("tp",""), "FIB4_FP": fm.get("fp",""),
                        "FIB4_TN": fm.get("tn",""), "FIB4_FN": fm.get("fn",""),
                    })

                    if len(np.unique(yt_e)) > 1:
                        z_dl, p_dl   = delong_auc_test(yt_e, prfps_cont_e, fib4_cont_e)
                        pb_auroc = paired_bootstrap_contrast(yt_e, prfps_cont_e, fib4_cont_e, "auroc",       n_boot)
                        pb_f1    = paired_bootstrap_contrast(yt_e, prfps_cont_e, fib4_cont_e, "f1_1",        n_boot)
                        pb_sens  = paired_bootstrap_contrast(yt_e, prfps_cont_e, fib4_cont_e, "sensitivity", n_boot)
                        pb_spec  = paired_bootstrap_contrast(yt_e, prfps_cont_e, fib4_cont_e, "specificity", n_boot)
                        log(f"  [{vsetting}] DeLong Tap pRFPS={pm.get('auroc',float('nan')):.3f} "
                            f"FIB-4={fm.get('auroc',float('nan')):.3f} z={z_dl:.3f} p={p_dl:.4f}")
                        dl_rows.append({
                            "Cohort": "Tapestry", "Setting": vsetting,
                            "N_EASL_eligible": int(eligible.sum()),
                            "N_intermediate_excluded": int((~eligible).sum()),
                            "pRFPS_cutoff_MCB": round(float(prfps_cutoff), 3),
                            "pRFPS_AUROC": round(float(pm.get("auroc", float("nan"))), 4),
                            "FIB4_AUROC":  round(float(fm.get("auroc", float("nan"))), 4),
                            "AUROC_diff_pRFPS_FIB4": round(float(pm.get("auroc",0)-fm.get("auroc",0)), 4),
                            "DeLong_z": round(float(z_dl),3) if not np.isnan(z_dl) else "N/A",
                            "DeLong_p": round(float(p_dl),4) if not np.isnan(p_dl) else "N/A",
                            "AUROC_Boot_CI_lo": pb_auroc["ci_lo"], "AUROC_Boot_CI_hi": pb_auroc["ci_hi"],
                            "AUROC_Boot_p": pb_auroc["p_value"],  "AUROC_Significant": pb_auroc["significant"],
                            "pRFPS_F1_Rapid": round(float(pm.get("f1_rapid", float("nan"))), 4),
                            "FIB4_F1_Rapid":  round(float(fm.get("f1_rapid", float("nan"))), 4),
                            "F1_Rapid_diff": pb_f1["diff"],   "F1_Rapid_CI_lo": pb_f1["ci_lo"],
                            "F1_Rapid_CI_hi": pb_f1["ci_hi"],"F1_Rapid_p": pb_f1["p_value"],
                            "F1_Rapid_Significant": pb_f1["significant"],
                            "pRFPS_Sensitivity": round(float(pm.get("sensitivity", float("nan"))), 4),
                            "FIB4_Sensitivity":  round(float(fm.get("sensitivity", float("nan"))), 4),
                            "Sens_diff": pb_sens["diff"], "Sens_CI_lo": pb_sens["ci_lo"],
                            "Sens_CI_hi": pb_sens["ci_hi"], "Sens_p": pb_sens["p_value"],
                            "Sens_Significant": pb_sens["significant"],
                            "pRFPS_Specificity": round(float(pm.get("specificity", float("nan"))), 4),
                            "FIB4_Specificity":  round(float(fm.get("specificity", float("nan"))), 4),
                            "Spec_diff": pb_spec["diff"], "Spec_CI_lo": pb_spec["ci_lo"],
                            "Spec_CI_hi": pb_spec["ci_hi"], "Spec_p": pb_spec["p_value"],
                            "Spec_Significant": pb_spec["significant"],
                        })

            except Exception as e:
                log(f"  [{vsetting}] Tapestry pRFPS/FIB-4 comparison failed: {e}")
                import traceback; log(traceback.format_exc())

         # Accumulate Tapestry data for combined DCA (run after loop)
        if prfps_res_tap is not None:
            _tap_prfps_by_setting[vsetting] = prfps_res_tap
        if fib4_res_tap is not None:
            _tap_fib4_by_setting[vsetting] = {
                "fib4_continuous": fib4_res_tap["fib4_continuous"],
                "y_arr":           y_arr,
            }
        # ── Patient-level export ──────────────────────────────────────────────
        try:
            tap_tag = f"Tapestry_{vsetting}"
            tap_df  = pd.DataFrame()
            tap_df["patient_idx"]    = range(len(Xv_raw_s))
            tap_df["split"]          = "validation"
            tap_df["setting"]        = vsetting
            tap_df["cohort"]         = "Tapestry"
            tap_df["original_index"] = idx_s.tolist()
            tap_df["subgroup"]       = sg_s.values
            tap_df["true_label_num"] = y_arr
            tap_df["true_label_str"] = np.where(y_arr==1, POSITIVE_LABEL, NEGATIVE_LABEL)

            for feat in feats_s:
                tap_df[feat] = Xv_raw_s[feat].values

            best_nm_t = max(results_s, key=lambda m: results_s[m].get("auroc", 0))
            for nm, pv in model_probs.items():
                thr = results_s[nm]["optimal_threshold"]
                tap_df[f"proba_{nm}"] = pv.round(4)
                if nm == best_nm_t:
                    pred_best = (pv >= thr).astype(int)
                    tap_df["best_model_name"]      = nm
                    tap_df["best_model_proba"]     = pv.round(4)
                    tap_df["best_model_threshold"] = round(float(thr), 4)
                    tap_df["best_model_pred"]      = pred_best
                    tap_df["best_model_correct"]   = (pred_best == y_arr).astype(int)

            if prfps_res_tap is not None:
                tap_df["pRFPS_int_score"]   = prfps_res_tap["score_int"]
                tap_df["pRFPS_cont_score"]  = prfps_res_tap["score_cont"].round(4)
                tap_df["pRFPS_int_pred"]    = prfps_res_tap["pred_int"].astype(int)
                tap_df["pRFPS_cont_pred"]   = prfps_res_tap.get(
                    "pred_cont", prfps_res_tap["pred_int"]).astype(int)
                tap_df["pRFPS_int_cutoff"]  = round(float(prfps_res_tap["cutoff_int"]), 3)
                tap_df["pRFPS_cont_cutoff"] = round(float(prfps_res_tap.get("cutoff_cont", np.nan)), 4)
                tap_df["pRFPS_correct"]     = (prfps_res_tap["pred_int"].astype(int)==y_arr).astype(int)
                tap_df["pRFPS_note"]        = "MCB-derived formula, no refitting"

            if fib4_res_tap is not None:
                fib4_b = pd.Series(fib4_res_tap["fib4_binary"])
                tap_df["FIB4_score"]         = np.round(fib4_res_tap["fib4_continuous"], 3)
                tap_df["FIB4_binary_label"]  = fib4_res_tap["fib4_binary"]
                tap_df["FIB4_risk_category"] = np.where(
                    fib4_b.values==0, "Low",
                    np.where(fib4_b.values==1, "High", "Intermediate"))
                tap_df["FIB4_pred_binary"]   = np.where(fib4_b.notna(), fib4_b.values, np.nan)
                tap_df["FIB4_correct"]       = np.where(
                    fib4_b.notna(),
                    (fib4_b.values.astype(float)==y_arr.astype(float)).astype(float), np.nan)

            tp = f"{tap_out}/{tap_tag}.csv"
            tap_df.to_csv(tp, index=False)
            log(f"  [patient export] Tap {vsetting}: {len(tap_df)} rows -> {tp}")
        except Exception as e:
            log(f"  Tapestry patient export failed [{vsetting}]: {e}")
    # ── Combined MCB + Tapestry DCA (4 curves: pRFPS vs FIB-4 × 2 cohorts) ───
    for _vs in val_settings:
        _mcb_pr   = (all_prfps_results or {}).get(_vs)
        _train    = (mcb_train_data_by_setting or {}).get(_vs, {})
        _tap_pr   = _tap_prfps_by_setting.get(_vs)
        _tap_f4   = _tap_fib4_by_setting.get(_vs)

        _mcb_y     = _mcb_pr.get("y_true")    if _mcb_pr else None
        _mcb_prfps = _mcb_pr.get("score_int") if _mcb_pr else None

        # MCB FIB-4: try direct dict first, then fallback to prfps_result storage
        _mcb_fib4 = (mcb_fib4_by_setting or {}).get(_vs)
        if _mcb_fib4 is None and _mcb_pr is not None:
            _mcb_fib4 = _mcb_pr.get("mcb_fib4_cont")  # stored by MCB DCA loop

        # Tap data: log what's available to help debug
        log(f"  [DCA {_vs}] mcb_pr={'OK' if _mcb_pr else 'None'}  "
            f"mcb_fib4={'OK' if _mcb_fib4 is not None else 'None'}  "
            f"tap_pr={'OK' if _tap_pr else 'None'}  "
            f"tap_f4={'OK' if _tap_f4 else 'None'}")

        missing = [k for k, v in [
            ("MCB y_true", _mcb_y), ("MCB pRFPS", _mcb_prfps),
            ("MCB FIB-4",  _mcb_fib4), ("Tap pRFPS", _tap_pr),
            ("Tap FIB-4",  _tap_f4)
        ] if v is None]
        if missing:
            log(f"  [DCA {_vs}] Missing {missing} -- skipping"); continue
        try:
            dca_focused(
                y_true_mcb         = _mcb_y,
                fib4_mcb           = np.array(_mcb_fib4, dtype=float),
                prfps_mcb          = np.array(_mcb_prfps, dtype=float),
                setting            = _vs,
                out_dir            = f"{results_dir}/dca_plots",
                fib4_train_scores  = _train.get("fib4_train"),
                prfps_train_scores = _train.get("prfps_train"),
                y_train            = _train.get("y_train"),
                y_true_tap  = _tap_f4["y_arr"],
                fib4_tap    = _tap_f4["fib4_continuous"],
                prfps_tap   = _tap_pr["score_int"].astype(float),
            )
            log(f"  [DCA {_vs}] Combined MCB+Tapestry 4-curve plot saved")
        except Exception as _de:
            log(f"  Combined DCA [{_vs}] failed: {_de}")
            import traceback; log(traceback.format_exc())
    # ── Save all outputs ──────────────────────────────────────────────────────
    pd.DataFrame(ext_rows).to_csv(f"{OUT}/tapestry_validation.csv", index=False)
    pd.DataFrame(cal_rows).to_csv(f"{OUT}/tapestry_calibration.csv", index=False)
    if pf_rows:
        pd.DataFrame(pf_rows).to_csv(f"{OUT}/tapestry_prfps_fib4.csv", index=False)
    if dl_rows:
        pd.DataFrame(dl_rows).to_csv(f"{OUT}/tapestry_delong_prfps_fib4.csv", index=False)

    log("  Saved tapestry_validation.csv")
    if pf_rows: log("  Saved tapestry_prfps_fib4.csv")
    if dl_rows: log("  Saved tapestry_delong_prfps_fib4.csv")
    return pd.DataFrame(ext_rows)

# =============================================================================
# ORIGINAL build_shap_clinical_risk_score  (PRESERVED INTACT)
# =============================================================================

