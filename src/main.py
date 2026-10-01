"""
main.py — Main pipeline entry point.
"""
import os, sys

# Windows multiprocessing safety — must be before any sklearn/joblib import
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# Support both: python src/main.py (direct) and python scripts/run.py (package)
if __package__ is None or __package__ == "":
    # Running directly — add repo root to path and use absolute imports
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _repo not in sys.path:
        sys.path.insert(0, _repo)
    from src.utils import *
    from src.train import train_and_evaluate_setting, get_oof_probabilities, append_master, apply_prfps_formula, save_hyperparameter_info, save_stacking_ensemble_info
    from src.prfps import build_shap_clinical_risk_score, normalize_binary_shap_values, compute_continuous_prfps_variants
    from src.evaluate import compute_best_model_shap, save_patient_level_dataset, covariate_shift_analysis, plot_metrics_with_ci, shap_vs_prfps_delta, shap_vs_prfps_delta_from_result, compute_prfps_weight_stability
    from src.ablation import ablation_feature_selection, nested_feature_selection, ablation_partition_contribution
    from src.validate import validate_external
else:
    # Running as package via scripts/run.py — use relative imports
    from .utils import *
    from .train import train_and_evaluate_setting, get_oof_probabilities, append_master, apply_prfps_formula, save_hyperparameter_info, save_stacking_ensemble_info
    from .prfps import build_shap_clinical_risk_score, normalize_binary_shap_values, compute_continuous_prfps_variants
    from .evaluate import compute_best_model_shap, save_patient_level_dataset, covariate_shift_analysis, plot_metrics_with_ci, shap_vs_prfps_delta, shap_vs_prfps_delta_from_result, compute_prfps_weight_stability
    from .ablation import ablation_feature_selection, nested_feature_selection, ablation_partition_contribution
    from .validate import validate_external

def main(data_path: str, val_path: str, n_boot: int = 1000):
    section("MASLD REVISED PIPELINE -- START")

    # ── Load data ──────────────────────────────────────────────────────────────
    df_raw     = pd.read_csv(data_path, sep="\t", low_memory=False)
    df_val_raw = pd.read_csv(val_path,  sep="\t", low_memory=False)

    (X, y_orig, y_num, subgroups, feat_names,
     data_proc, _, _) = load_and_prepare_data(df_raw, "MCB")
    (X_val, y_val_orig, y_val_num, subgroups_val,
     _, _, _, _) = load_and_prepare_data(df_val_raw, "Tapestry")

    master_rows          = []
    all_settings_results = {}   # {setting: {model_name: eval_dict}}
    all_prfps_results    = {}   # {setting: prfps_result dict}
    all_test_indices     = {}   # {setting: list of original df_raw row indices}
    all_train_indices     = {}   # {setting: list of original df_raw row indices}
    all_test_subgroups   = {}   # {setting: subgroup labels for test patients}
    all_delta_rows       = []   # FIX-10: continuous vs integer pRFPS delta per setting
    all_settings_data    = {}   # feature selection ablation
    all_weight_stab_rows = []   # FIX-12: pRFPS weight stability per fold per setting
    all_mcb_fib4         = {}   # MCB FIB-4 continuous per setting for combined DCA
    all_mcb_train_data   = {}   # MCB train FIB-4/pRFPS scores for Platt calibration
    all_continuous_results        = []   # collect across settings
    all_continuous_feature_details = []  # collect across settings
    # ── One-time analyses ──────────────────────────────────────────────────────
    # lca_tables functions removed (not needed)
    # plot_temporal_diagram()
    # lca_class_sensitivity_and_justification(...)
    # lca_bootstrap_stability(...)
    # recalculate_cited_percentages(...)
    # test_progression_time_difference(df_raw)
    # fixed_vs_median_sensitivity(df_raw)
    # comparison_with_prior_studies()

    # ── Included vs Excluded Table 1 (Part 6) ──────────────────────────────────
    cont_vars = ["diagnosis_age","Avg_BMI","HbA1C","ALP","ALT.x","AST.x",
                 "PLATELET_COUNT","ALBUMIN","TRIGLYCERIDE","HDL","LDL","BUN",
                 "first_FIB4","NFS","AST_PT_ratio"]
    cat_vars  = ["Sex","Diabetes","hypertension","dyslipidemia","depression",
                 "heart","ckd","PNPLA3","TM6SF2","HSD17B13"]
    # included_vs_excluded_table(df_raw, X.index, cont_vars, cat_vars)

    # ── Settings loop: Overall + each subgroup ─────────────────────────────────
    settings = ["Overall"] + sorted(subgroups.unique().tolist())
    global_results = global_models = global_scl = global_feats = None
    all_setting_artifacts = {}
    for setting in settings:
        section(f"SETTING: {setting}")

        if setting == "Overall":
            Xs, ys = X, y_num
        else:
            mask = (subgroups == setting)
            Xs   = X[mask].reset_index(drop=True)
            ys   = y_num[mask].reset_index(drop=True)

        if len(ys.unique()) < 2 or len(ys) < 50:
            log(f"  Skipping {setting} (n={len(ys)})"); continue

        tag = setting.replace(" ","_")

        # Drop NaN (no imputation)
        Xs, _, ys, _ = drop_nan_rows(Xs, None, ys, label=setting)

        # 80/20 split -- threshold via OOF on training set (correct for small n)
        X_tr, X_te, y_tr, y_te = two_way_split(Xs, ys)

        # [FIX-4] Feature selection on TRAIN only
        feats = nested_feature_selection(X_tr, y_tr, feat_names, k=10, label=tag)

        # Train + evaluate (OOF threshold, no leakage)
        results, models, scl = train_and_evaluate_setting(
            X_tr, X_te, y_tr, y_te,
            feats, label=setting, n_boot=n_boot)
        all_setting_artifacts[setting] = {
            "results": results,
            "models": models,
            "scaler": scl,
            "features": feats,
        }
        # if setting != "Overall":
        all_settings_data[setting] = {
            "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
            "feats_subgroup": feats, "results": results,
        }    
        # [Part 2] Publication-ready metrics table
        results_to_publication_table(
            results, setting, f"{OUT}/results_{tag}.csv")

        # [FIX-2/14] Figures
        plot_roc_pr(results, tag)
        plot_calibration_curves(results, tag)

        # [Part 3] Calibration per model
        cal_rows = []
        for nm, r in results.items():
            yt, yp = r.get("y_true"), r.get("y_proba")
            if yt is not None and len(np.unique(yt)) > 1:
                cal_rows.append(calibration_analysis(
                    yt, yp, nm, setting,
                    out_dir=f"{results_dir}/calibration_plots"))
        if cal_rows:
            pd.DataFrame(cal_rows).to_csv(
                f"{OUT}/calibration_summary_{tag}.csv", index=False)

        # [Part 4] Decision Curve Analysis
        dca_preds = {nm: r["y_proba"] for nm, r in results.items()
                     if r.get("y_proba") is not None}
        decision_curve_analysis(y_te.values, dca_preds, setting)

        # Master table
        append_master(master_rows, results, setting)

        # Accumulate for combined metrics plot
        all_settings_results[setting] = results
        # Store test indices and subgroup labels for FIB-4 lookup in DCA/DeLong
        all_test_indices[setting]   = (X_te.index.tolist()
                                       if hasattr(X_te, "index")
                                       else list(range(len(y_te))))
        
        all_train_indices[setting]   = (X_tr.index.tolist()
                                       if hasattr(X_te, "index")
                                       else list(range(len(y_te))))
        all_test_subgroups[setting] = (subgroups.loc[X_te.index].values
                                       if hasattr(subgroups, "loc")
                                       else np.full(len(y_te), setting))

        # [FIX-3] Circularity -- disabled
        # if any(f in feats for f in FIB4_COMPONENTS):
        #     circularity_sensitivity(
        #         X_tr, X_te, y_tr, y_te,
        #         feats, tag, n_boot=min(n_boot, 300))

        # [FIX-12 / Part 7] SHAP stability across folds AND cohorts
        # Scale train features
        _shap_scl = StandardScaler().fit(X_tr[feats].values)
        Ntr_shap  = _shap_scl.transform(X_tr[feats].values)

        # Prepare Tapestry data for cross-cohort comparison (same features)
        _tap_feats = [f for f in feats if f in X_val.columns]
        Nval_shap  = None; Yval_shap = None
        if len(_tap_feats) == len(feats):
            # Filter Tapestry to same setting if subgroup
            if setting == "Overall":
                Xv_shap = X_val[_tap_feats].dropna()
                Yv_shap = y_val_num.loc[Xv_shap.index]
            else:
                _vmask  = (subgroups_val == setting)
                Xv_shap = X_val[_vmask][_tap_feats].dropna()
                Yv_shap = y_val_num[_vmask].loc[Xv_shap.index]
            if len(Xv_shap) > 20:
                Nval_shap = _shap_scl.transform(Xv_shap.values)
                Yval_shap = Yv_shap.values

        # shap_stability_cv -- disabled
        # shap_stability_cv(
        #     Ntr_shap, y_tr.values, feats, tag,
        #     n_splits=5, cv_threshold=0.30,
        #     X_val_np=Nval_shap, y_val_np=Yval_shap,
        #     val_tag="Tapestry")

        # [FIX-10] SHAP vs pRFPS
        best_nm = max(results, key=lambda n: results[n].get("f1_macro", 0))
        log(f"  Best Model [{setting}]: {best_nm}")
        SHAP_TREE_PREFERENCE = [
            "XGBoost", "LightGBM", "Random_Forest",
            "GradientBoosting", "DecisionTree",
        ]

        if best_nm in models:
            shap_nm = best_nm
            scl2   = StandardScaler().fit(X_tr[feats].values)
            Ntr_np = scl2.transform(X_tr[feats].values)
            Nte_np = scl2.transform(X_te[feats].values)
            shap_vs_prfps_delta(
                models[best_nm], Ntr_np, Nte_np, y_te.values, feats, tag)
        elif XGBOOST_AVAILABLE and "XGBoost" in models:
            scl2   = StandardScaler().fit(X_tr[feats].values)
            Ntr_np = scl2.transform(X_tr[feats].values)
            Nte_np = scl2.transform(X_te[feats].values)
            shap_vs_prfps_delta(
                models["XGBoost"], Ntr_np, Nte_np, y_te.values, feats, tag)
        else:
            raise ValueError("No suitable model available for SHAP")

        # SHAP + pRFPS score for best model
        if best_nm in models:
            shap_nm = best_nm
        elif XGBOOST_AVAILABLE and "XGBoost" in models:
            shap_nm = "XGBoost"
        else:
            raise ValueError("No suitable model available for SHAP")
        # shap_nm = next(
        #     (m for m in SHAP_TREE_PREFERENCE if m in models),
        #     max(results, key=lambda n: results[n].get("auroc", 0))  # fallback
        # )
        # log(f"  SHAP model for pRFPS: {shap_nm} (tree-preferred for exact SHAP)")
        # [FIX-12 EXTENDED] pRFPS weight stability using the same SHAP model class
        _stab_rows = compute_prfps_weight_stability(
            X_tr_np=Ntr_shap,
            y_tr_np=y_tr.values,
            feature_names=feats,
            model_name=shap_nm,
            fitted_model=models.get(shap_nm),
            tag=tag,
            n_splits=5,
            shap_coverage_threshold=0.80,
            max_features=10,
        )

        if _stab_rows:
            all_weight_stab_rows.extend(_stab_rows)

        shap_out_dir = f"{results_dir}/best_model_shap/{tag}"
        X_tr_df = X_tr[feats].copy()
        X_te_df = X_te[feats].copy()
        shap_result = compute_best_model_shap(
            shap_nm, models.get(shap_nm), X_tr_df, X_te_df,
            feats, CLASS_NAMES, tag, shap_out_dir, scaler=None, scaled=False)

        prfps_result = None
        if shap_result is not None:
            prfps_result = build_shap_clinical_risk_score(
                shap_values   = shap_result["shap_values"],
                shap_values_test = shap_result["shap_values_test"],
                X_test        = X_te_df,
                y_test        = y_te.values,
                X_train       = X_tr[feats].copy(),
                y_train       = y_tr.values,        # TRAINING labels -- same length as X_train
                feature_names = feats,
                analysis_name = tag,
                output_dir    = f"{results_dir}/clinical_risk_score/{tag}",
                top_n_features= 5,
                feature_labels_map = FEATURE_LABELS,
                raw_unit_map  = {
                    "diagnosis_age": "years", "Avg_BMI": "kg/m2",
                    "ALP": "U/L", "AST.x": "U/L",
                    "PLATELET_COUNT": "x10^9/L", "ALBUMIN": "g/dL",
                    "BUN": "mg/dL", "AST_PT_ratio": "score", "NFS": "score",
                })
            # Store pRFPS scores for focused DCA + DeLong
            if prfps_result is not None:
                all_prfps_results[setting] = prfps_result

        # ── Save patient-level dataset (train + test) ─────────────────────────
        # All features + best model predictions + pRFPS + FIB-4 per patient.
        # FIX-10 corrected delta is computed AFTER this call (prfps_result needed).
        try:
            save_patient_level_dataset(
                X_tr          = X_tr,
                X_te          = X_te,
                y_tr          = y_tr,
                y_te          = y_te,
                feats         = feats,
                results       = results,
                models        = models,
                best_nm       = max(results, key=lambda n: results[n].get("auroc", 0)),
                prfps_result  = prfps_result,
                df_raw        = df_raw,
                setting       = setting,
                tag           = tag,
            )
        except Exception as _pe:
            log(f"  Patient-level export failed [{setting}]: {_pe}")

        # ── FIX-10 corrected: delta using real pRFPS result ───────────────────
        if prfps_result is not None:
            _delta_row = shap_vs_prfps_delta_from_result(
                prfps_result, tag, n_boot=min(n_boot, 500))
            if _delta_row:
                all_delta_rows.append(_delta_row)
        # ============================================================
        # *** ADD THE CONTINUOUS pRFPS BLOCK HERE ***
        # Paste immediately after the block above, still inside the
        # settings loop, before the `if setting == "Overall":` block
        # ============================================================
    
        # ── Continuous pRFPS variant comparison (addresses reviewer comment) ──────
        if prfps_result is not None:
            try:
                #from prfps_continuous_extension import compute_continuous_prfps_variants
    
                df_variants, df_feat_detail = compute_continuous_prfps_variants(
                    top           = prfps_result['feature_table'],
                    X_te          = X_te[feats].copy(),
                    X_tr          = X_tr[feats].copy(),
                    y_te          = y_te.values,
                    y_tr          = y_tr.values,
                    analysis_name = tag,
                    output_dir    = f'{results_dir}/prfps_continuous_variants',
                )
                df_variants['setting'] = tag
                df_feat_detail['setting'] = tag
                all_continuous_results.append(df_variants)
                all_continuous_feature_details.append(df_feat_detail)
    
                log(f'  [Continuous pRFPS] {tag}:')
                log(f'    Binary int  AUROC = {df_variants.loc[0,"auroc"]:.4f}')
                log(f'    Binary raw  AUROC = {df_variants.loc[1,"auroc"]:.4f}')
                log(f'    Continuous  AUROC = {df_variants.loc[2,"auroc"]:.4f}  '
                    f'Delta = {df_variants.loc[2,"delta_auroc_vs_binary_int"]:+.4f}  '
                    f'DeLong p = {df_variants.loc[2,"delong_p_vs_binary_int"]}')
    
            except Exception as _ce:
                log(f'  [Continuous pRFPS] {tag} failed: {_ce}')
                import traceback; log(traceback.format_exc())
        if setting == "Overall":
            global_results = results
            global_models  = models
            global_scl     = scl
            global_feats   = feats
    

    # ── Master table ────────────────────────────────────────────────────────────
    pd.DataFrame(master_rows).to_csv(f"{OUT}/MASTER_all_results.csv", index=False)
    log("Saved MASTER_all_results.csv")
    if all_continuous_results:
        os.makedirs(f'{results_dir}/prfps_continuous_variants', exist_ok=True)
    
        df_all = pd.concat(all_continuous_results, ignore_index=True)
        df_all.to_csv(
            f'{results_dir}/prfps_continuous_variants/SUMMARY_all_settings.csv',
            index=False)
        log('Saved: SUMMARY_all_settings.csv (continuous pRFPS variants)')
    
        # Print the key table
        cols = ['setting', 'variant', 'auroc', 'delta_auroc_vs_binary_int',
                'delong_p_vs_binary_int', 'f1_macro', 'sensitivity', 'specificity']
        log('\n' + df_all[cols].to_string(index=False))
    
    if all_continuous_feature_details:
        pd.concat(all_continuous_feature_details, ignore_index=True).to_csv(
            f'{results_dir}/prfps_continuous_variants/SUMMARY_feature_indicator_detail.csv',
            index=False)
        log('Saved: SUMMARY_feature_indicator_detail.csv')

    # ── Feature selection ablation (partition × feature set) ─────────────────
    if all_settings_data and global_models is not None and global_feats is not None:
        log("Running feature selection ablation ...")
        try:
            ablation_feature_selection(
                settings_data      = all_settings_data,
                all_feat_names     = feat_names,
                overall_feats      = global_feats,
                overall_se_model   = global_models.get("Stacking_Ensemble"),
                overall_scl        = global_scl,
                overall_se_results = global_results.get("Stacking_Ensemble", {}),
                all_test_indices   = all_test_indices,
                X_full             = X,
                n_boot             = min(n_boot, 300),
            )
        except Exception as _fe:
            log(f"  Feature selection ablation failed: {_fe}")
            import traceback; log(traceback.format_exc())

    # ── [C2+C5] Ablation: partition contribution analysis ─────────────────────
    if all_settings_results and len(all_settings_results) > 1 and global_models is not None:
        log("Running 2x2 ablation -- partition × threshold ...")
        # Reconstruct Overall train split
        _ov_te_idx = set(all_test_indices.get("Overall", []))
        _ov_all_X, _, _ov_all_y, _ = drop_nan_rows(X, None, y_num, label="ablation")
        _ov_tr_mask = ~_ov_all_X.index.isin(_ov_te_idx)
        _X_tr_ov = _ov_all_X[_ov_tr_mask]
        _y_tr_ov = _ov_all_y[_ov_tr_mask]
        try:
            _ov_se_model   = global_models.get("Stacking_Ensemble")
            _ov_se_results = global_results.get("Stacking_Ensemble", {})
            if _ov_se_model is None:
                log("  Ablation skipped: Overall Stacking Ensemble not available")
            else:
                ablation_partition_contribution(
                    results_by_setting  = all_settings_results,
                    X_full              = X,
                    y_full              = y_num,
                    all_test_indices    = all_test_indices,
                    features_overall    = global_feats,
                    overall_se_model    = _ov_se_model,
                    overall_scl         = global_scl,
                    overall_se_results  = _ov_se_results,
                    n_boot              = min(n_boot, 1000),
                )
        except Exception as _abl_e:
            log(f"  Ablation failed: {_abl_e}")
            import traceback; log(traceback.format_exc())

    # ── [C1+C3] Covariate shift + subgroup profile comparison ─────────────────
    log("Running covariate shift + subgroup profile analysis ...")
    try:
        Xm_clean, _, ym_clean, sgm_clean = drop_nan_rows(
            X[global_feats], None, y_num, subgroups, "mcb_shift")
        Xt_clean, _, yt_clean, sgt_clean = drop_nan_rows(
            X_val[[f for f in global_feats if f in X_val.columns]],
            None, y_val_num, subgroups_val, "tap_shift")
        covariate_shift_analysis(
            Xm_clean, ym_clean, sgm_clean,
            Xt_clean, yt_clean, sgt_clean,
            global_feats)
    except Exception as _e:
        log(f"  Covariate shift analysis failed: {_e}")

    # ── FIX-10: Save continuous vs integer pRFPS delta summary table ─────────
    if all_delta_rows:
        delta_df = pd.DataFrame(all_delta_rows)
        delta_df.to_csv(f"{OUT}/continuous_vs_integer_prfps_delta.csv", index=False)
        log("  Saved continuous_vs_integer_prfps_delta.csv")
        log(delta_df[["Setting","AUROC_SHAP_additive","AUROC_weight_raw_pRFPS", "AUROC_integer_pRFPS","AUROC_delta_SHAP_vs_int","AUROC_delta_cont_vs_int","DeLong_p_SHAP_vs_int"]].to_string(index=False))

    # ── FIX-12: Save pRFPS weight stability table ──────────────────────────────
    if all_weight_stab_rows:
        pd.DataFrame(all_weight_stab_rows).to_csv(
            f"{OUT}/prfps_weight_stability.csv", index=False)
        log("  Saved prfps_weight_stability.csv")

    # ── Pairwise statistical contrasts (Reviewer 59JH) ────────────────────────
    if all_settings_results:
        log("Running pairwise statistical contrasts ...")
        run_pairwise_contrasts(
            all_results_dict = all_settings_results,
            n_boot           = n_boot,
        )

    # ── Focused DCA: FIB-4 + pRFPS + SE + Treat-All/None ─────────────────────
    # Uses REAL pRFPS integer scores from build_shap_clinical_risk_score
    # and FIB-4 as the clinical comparator.
    # DeLong test between pRFPS and FIB-4 is also computed here.

    if all_settings_results:
        log("Generating focused DCA curves with real pRFPS + FIB-4 ...")
        all_delong_rows = []   # accumulate across settings; written ONCE at end
        for _setting, _res in all_settings_results.items():
            _se  = _res.get("Stacking_Ensemble", {})
            _yt  = np.array(_se.get("y_true", []))
            _yp  = np.array(_se.get("y_proba", []))
            if len(_yt) < 10:
                continue

            # ── Real pRFPS (normalised 0-1 for DCA) ───────────────────────────
            _prfps_res = all_prfps_results.get(_setting)
            if _prfps_res is not None:
                _prfps_norm = _prfps_res["score_norm"]   # 0-1 integer score
                log(f"  [{_setting}] pRFPS AUROC={_prfps_res['auroc']:.3f}  "
                    f"cutoff={_prfps_res['threshold']:.0f}  "
                    f"features={_prfps_res.get('n_features_selected','?')}")
            else:
                _prfps_norm = _yp
                log(f"  [{_setting}] pRFPS not available -- using SE proba")

            # ── Real FIB-4 (guideline-aligned binary -> continuous for DCA) ───
            # Compute actual FIB-4 from AST/ALT/Platelet/Age in raw data.
            # Normalise 0-1 so DCA threshold axis is comparable to probability.
            _fib4_norm = _yp  # safe fallback if columns missing
            if all(c in df_raw.columns for c in
                   ["AST.x", "ALT.x", "PLATELET_COUNT", "diagnosis_age"]):
                try:
                    _fib4_info = prepare_fib4_comparator(
                        df_raw         = df_raw,
                        test_index     = all_test_indices.get(_setting, []),
                        y_true_test    = _yt,
                        setting        = _setting,
                        subgroups_test = all_test_subgroups.get(_setting),
                    )
                    _fib4_norm = _fib4_info["fib4_norm"]
                    # align length
                    if len(_fib4_norm) != len(_yt):
                        _fib4_norm = _yp
                except Exception as _ef:
                    log(f"  FIB-4 computation failed [{_setting}]: {_ef}")
                    _fib4_norm = _yp
            else:
                log(f"  [{_setting}] FIB-4 columns missing -- using SE proba as proxy")

            # ── Focused DCA with Platt-calibrated scores (FIX-B) ─────────────
            # Pass RAW scores (not normalised) so Platt scaling can convert them
            # to probabilities using training data. This makes the DCA threshold
            # axis commensurable with the SE's predict_proba output.
            try:
                # Raw FIB-4 continuous scores on training patients
                _fib4_train_raw = None
                try:
                    _tr_idx = all_test_indices.get(_setting, [])
                    # Training indices = all in setting minus test indices
                    if _setting == "Overall":
                        _all_idx = X.index.tolist()
                    else:
                        _sg_mask = (subgroups == _setting)
                        _all_idx = X[_sg_mask].index.tolist()
                    _train_idx_set = set(_all_idx) - set(_tr_idx)
                    _train_idx = [i for i in _all_idx if i in _train_idx_set]
                    if len(_train_idx) > 10:
                        _df_tr_raw = df_raw.loc[_train_idx].copy().reset_index(drop=True)
                        _fib4_tr   = compute_fib4_score(_df_tr_raw)
                        _fib4_train_raw = _fib4_tr.values.astype(float)
                except Exception as _ef:
                    log(f"  FIB-4 train scores failed [{_setting}]: {_ef}")

                # Raw pRFPS integer scores on training patients (from pRFPS result)
                _prfps_train_raw = None
                _y_tr_dca        = None
                if _prfps_res is not None:
                    try:
                        # Re-score training patients using the same pRFPS formula
                        _top = _prfps_res["feature_table"]
                        _prfps_tr_int = np.zeros(len(_train_idx) if _fib4_train_raw is not None else 0)
                        if _fib4_train_raw is not None and len(_train_idx) > 0:
                            _X_tr_dca = df_raw.loc[_train_idx][
                                [f for f in _top["feature"].tolist() if f in df_raw.columns]
                            ].copy().reset_index(drop=True)
                            _prfps_tr_scores = np.zeros(len(_X_tr_dca))
                            for _, _r in _top.iterrows():
                                _f = _r["feature"]
                                if _f not in _X_tr_dca.columns: continue
                                _v = _X_tr_dca[_f].values.astype(float)
                                _t = _r["youden_threshold"]
                                _ind = (_v > _t if _r["direction"] == "Risk+" else _v < _t).astype(float)
                                _prfps_tr_scores += int(_r["weight_int"]) * _ind
                            _prfps_train_raw = _prfps_tr_scores.astype(int)
                            # Labels aligned to training indices
                            _y_tr_dca = y_num.loc[_train_idx].values.astype(int)
                    except Exception as _ep:
                        log(f"  pRFPS train scores failed [{_setting}]: {_ep}")

                # dca_focused -- disabled
                # dca_focused(
                #     y_true_mcb         = _yt,
                #     fib4_mcb           = _fib4_info["fib4_continuous"] if _fib4_info is not None else _fib4_norm,
                #     prfps_mcb          = _prfps_res["score_int"].astype(float) if _prfps_res is not None else _prfps_norm,
                #     setting            = _setting,
                #     out_dir            = f"{results_dir}/dca_plots",
                #     fib4_train_scores  = _fib4_train_raw,
                #     prfps_train_scores = _prfps_train_raw,
                #     y_train            = _y_tr_dca,
                # )
            except Exception as _e:
                log(f"  Focused DCA failed for {_setting}: {_e}")
                import traceback; log(traceback.format_exc())

            # ── Full comparison: pRFPS vs FIB-4 (AUROC + binary metrics) ─────
            # Uses ONLY EASL-eligible patients (intermediate FIB-4 excluded).
            # pRFPS binarised at its Youden-optimal cutoff derived from training.
            # FIB-4 binarised using age-adjusted EASL thresholds:
            #   Low-risk  < 1.3 (age<65) / < 2.0 (age>=65) -> Slow Progression
            #   High-risk > 2.67 -> Rapid Progression
            # All metrics: AUROC (DeLong), F1-macro, Accuracy, Precision, Recall,
            # Sensitivity, Specificity (paired bootstrap with 95% CI).
            if len(np.unique(_yt)) > 1:
                try:
                    delong_rows = []

                    # Get FIB-4 binary labels and eligible mask
                    _fib4_binary_all = (_fib4_info["fib4_binary"]
                                        if _fib4_info is not None else None)

                    if _fib4_binary_all is None:
                        log(f"  [{_setting}] No FIB-4 binary labels -- skipping DeLong")
                    else:
                        # ── Length alignment guard ────────────────────────────
                        # All three arrays must have identical length = len(_yt)
                        n_yt   = len(_yt)
                        n_fib4 = len(_fib4_binary_all)
                        n_norm = len(_fib4_norm)
                        n_prfps= len(_prfps_norm)

                        if not (n_fib4 == n_yt == n_norm == n_prfps):
                            log(f"  [{_setting}] Length mismatch -- "
                                f"_yt={n_yt} fib4_binary={n_fib4} "
                                f"fib4_norm={n_norm} prfps_norm={n_prfps} "
                                f"-- skipping DeLong")
                        else:
                        # ── EASL-eligible subset (intermediate excluded) ───────
                            _eligible = ~np.isnan(_fib4_binary_all)
                            if _eligible.sum() < 10:
                                log(f"  [{_setting}] Too few EASL-eligible ({_eligible.sum()}) "
                                    f"-- skipping DeLong")
                            else:
                                # Ground truth for eligible patients
                                _yt_e  = _yt[_eligible]

                            # FIB-4 binary prediction (0=Slow, 1=Rapid)
                            _fib4_pred = _fib4_binary_all[_eligible].astype(int)

                            # FIB-4 continuous (normalised) for AUROC
                            _fib4_cont_e = _fib4_norm[_eligible]

                            # pRFPS continuous (normalised) for AUROC + binary pred
                            _prfps_cont_e = _prfps_norm[_eligible]

                            # pRFPS binary at training-derived Youden cutoff
                            _prfps_cutoff = (_prfps_res["threshold"]
                                             if _prfps_res is not None else 0.5)
                            # threshold was on raw integer score; score_norm = score/max
                            # so normalised cutoff = cutoff / max(score_int)
                            _score_max = float(_prfps_res["score_int"].max()) if _prfps_res else 1.0
                            _prfps_cutoff_norm = (_prfps_cutoff / (_score_max + 1e-10)
                                                  if _score_max > 0 else 0.5)
                            _prfps_pred = ((_prfps_res["score_int"][_eligible]
                                            >= _prfps_cutoff)
                                           if _prfps_res is not None
                                           else (_prfps_cont_e >= 0.5)).astype(int)

                            # ── Helper: compute all binary metrics ────────────
                            def _metrics(yt, ypred, ycont):
                                if len(np.unique(yt)) < 2:
                                    return {}
                                cm_ = confusion_matrix(yt, ypred, labels=[0,1])
                                tn_,fp_,fn_,tp_ = cm_.ravel()
                                return {
                                    "accuracy":    accuracy_score(yt, ypred),
                                    "brier_score": brier_score_loss(yt, ypred),
                                    "f1_macro":    f1_score(yt, ypred, average="macro", zero_division=0),
                                    "f1_rapid":    f1_score(yt, ypred, pos_label=1, average="binary", zero_division=0),
                                    "precision":   precision_score(yt, ypred, pos_label=1, zero_division=0),
                                    "recall":      recall_score(yt, ypred, pos_label=1, zero_division=0),
                                    "sensitivity": tp_/(tp_+fn_+1e-10),
                                    "specificity": tn_/(tn_+fp_+1e-10),
                                    "ppv":         tp_/(tp_+fp_+1e-10),
                                    "npv":         tn_/(tn_+fn_+1e-10),
                                    "auroc":       roc_auc_score(yt, ycont),
                                    "tp": int(tp_), "fp": int(fp_),
                                    "tn": int(tn_), "fn": int(fn_),
                                }

                            pm = _metrics(_yt_e, _prfps_pred, _prfps_cont_e)
                            fm = _metrics(_yt_e, _fib4_pred, _fib4_cont_e)

                            # ── DeLong: AUROC comparison ──────────────────────
                            z_dl, p_dl = delong_auc_test(
                                _yt_e, _prfps_cont_e, _fib4_cont_e)
                            pb_auroc = paired_bootstrap_contrast(
                                _yt_e, _prfps_cont_e, _fib4_cont_e, "auroc", n_boot)

                            # ── Paired bootstrap: F1, Accuracy, Sens, Spec ────
                            pb_f1   = paired_bootstrap_contrast(
                                _yt_e, _prfps_cont_e, _fib4_cont_e, "f1_1", n_boot)
                            pb_sens = paired_bootstrap_contrast(
                                _yt_e, _prfps_cont_e, _fib4_cont_e, "sensitivity", n_boot)
                            pb_spec = paired_bootstrap_contrast(
                                _yt_e, _prfps_cont_e, _fib4_cont_e, "specificity", n_boot)

                            log(f"  [{_setting}] pRFPS vs FIB-4 (n={_eligible.sum()} EASL-eligible):")
                            log(f"    AUROC:  pRFPS={pm.get('auroc',float('nan')):.3f}  "
                                f"FIB-4={fm.get('auroc',float('nan')):.3f}  "
                                f"diff={pm.get('auroc',0)-fm.get('auroc',0):+.3f}  "
                                f"z={z_dl:.3f}  p={p_dl:.4f}")
                            log(f"    Accuracy: pRFPS={pm.get('accuracy',float('nan')):.3f}  "
                                f"FIB-4={fm.get('accuracy',float('nan')):.3f}")
                            log(f"    F1-macro: pRFPS={pm.get('f1_macro',float('nan')):.3f}  "
                                f"FIB-4={fm.get('f1_macro',float('nan')):.3f}")
                            log(f"    F1-Rapid: pRFPS={pm.get('f1_rapid',float('nan')):.3f}  "
                                f"FIB-4={fm.get('f1_rapid',float('nan')):.3f}")
                            log(f"    Sens:  pRFPS={pm.get('sensitivity',float('nan')):.3f}  "
                                f"FIB-4={fm.get('sensitivity',float('nan')):.3f}  "
                                f"diff={pb_sens['diff']:+.3f} [{pb_sens['ci_lo']:.3f},{pb_sens['ci_hi']:.3f}] p={pb_sens['p_value']:.4f}")
                            log(f"    Spec:  pRFPS={pm.get('specificity',float('nan')):.3f}  "
                                f"FIB-4={fm.get('specificity',float('nan')):.3f}  "
                                f"diff={pb_spec['diff']:+.3f} [{pb_spec['ci_lo']:.3f},{pb_spec['ci_hi']:.3f}] p={pb_spec['p_value']:.4f}")

                            delong_rows.append({
                                # Setting & design
                                "Setting":                   _setting,
                                "N_EASL_eligible":           int(_eligible.sum()),
                                "N_intermediate_excluded":   int((~_eligible).sum()),
                                "FIB4_low_thresh_lt65":      1.3,
                                "FIB4_low_thresh_gte65":     2.0,
                                "FIB4_high_thresh":          2.67,
                                "pRFPS_cutoff":              round(float(_prfps_cutoff), 3),
                                # pRFPS performance
                                "pRFPS_AUROC":               round(float(pm.get("auroc",float("nan"))),4),
                                "pRFPS_brier":               round(float(pm.get("brier",float("nan"))),4),
                                "pRFPS_Accuracy":            round(float(pm.get("accuracy",float("nan"))),4),
                                "pRFPS_F1_macro":            round(float(pm.get("f1_macro",float("nan"))),4),
                                "pRFPS_F1_Rapid":            round(float(pm.get("f1_rapid",float("nan"))),4),
                                "pRFPS_Precision":           round(float(pm.get("precision",float("nan"))),4),
                                "pRFPS_Recall":              round(float(pm.get("recall",float("nan"))),4),
                                "pRFPS_Sensitivity":         round(float(pm.get("sensitivity",float("nan"))),4),
                                "pRFPS_Specificity":         round(float(pm.get("specificity",float("nan"))),4),
                                "pRFPS_PPV":                 round(float(pm.get("ppv",float("nan"))),4),
                                "pRFPS_NPV":                 round(float(pm.get("npv",float("nan"))),4),
                                "pRFPS_TP":                  pm.get("tp",""),
                                "pRFPS_FP":                  pm.get("fp",""),
                                "pRFPS_TN":                  pm.get("tn",""),
                                "pRFPS_FN":                  pm.get("fn",""),
                                # FIB-4 performance
                                "FIB4_AUROC":                round(float(fm.get("auroc",float("nan"))),4),
                                "FIB4_brier":                round(float(fm.get("brier",float("nan"))),4),
                                "FIB4_Accuracy":             round(float(fm.get("accuracy",float("nan"))),4),
                                "FIB4_F1_macro":             round(float(fm.get("f1_macro",float("nan"))),4),
                                "FIB4_F1_Rapid":             round(float(fm.get("f1_rapid",float("nan"))),4),
                                "FIB4_Precision":            round(float(fm.get("precision",float("nan"))),4),
                                "FIB4_Recall":               round(float(fm.get("recall",float("nan"))),4),
                                "FIB4_Sensitivity":          round(float(fm.get("sensitivity",float("nan"))),4),
                                "FIB4_Specificity":          round(float(fm.get("specificity",float("nan"))),4),
                                "FIB4_PPV":                  round(float(fm.get("ppv",float("nan"))),4),
                                "FIB4_NPV":                  round(float(fm.get("npv",float("nan"))),4),
                                "FIB4_TP":                   fm.get("tp",""),
                                "FIB4_FP":                   fm.get("fp",""),
                                "FIB4_TN":                   fm.get("tn",""),
                                "FIB4_FN":                   fm.get("fn",""),
                                # AUROC comparison (DeLong)
                                "AUROC_diff_pRFPS_minus_FIB4": round(float(pm.get("auroc",0)-fm.get("auroc",0)),4),
                                "DeLong_z":                  round(float(z_dl),3) if not np.isnan(z_dl) else "N/A",
                                "DeLong_p":                  round(float(p_dl),4) if not np.isnan(p_dl) else "N/A",
                                "AUROC_Boot_CI_lo":          pb_auroc["ci_lo"],
                                "AUROC_Boot_CI_hi":          pb_auroc["ci_hi"],
                                "AUROC_Boot_p":              pb_auroc["p_value"],
                                "AUROC_Significant":         pb_auroc["significant"],
                                # F1 Rapid Progression comparison
                                "F1_Rapid_diff":             pb_f1["diff"],
                                "F1_Rapid_CI_lo":            pb_f1["ci_lo"],
                                "F1_Rapid_CI_hi":            pb_f1["ci_hi"],
                                "F1_Rapid_p":                pb_f1["p_value"],
                                "F1_Rapid_Significant":      pb_f1["significant"],
                                # Sensitivity comparison
                                "Sens_diff":                 pb_sens["diff"],
                                "Sens_CI_lo":                pb_sens["ci_lo"],
                                "Sens_CI_hi":                pb_sens["ci_hi"],
                                "Sens_p":                    pb_sens["p_value"],
                                "Sens_Significant":          pb_sens["significant"],
                                # Specificity comparison
                                "Spec_diff":                 pb_spec["diff"],
                                "Spec_CI_lo":                pb_spec["ci_lo"],
                                "Spec_CI_hi":                pb_spec["ci_hi"],
                                "Spec_p":                    pb_spec["p_value"],
                                "Spec_Significant":          pb_spec["significant"],
                            })

                    # Collect rows into session accumulator (written once at end)
                    if delong_rows:
                        all_delong_rows.extend(delong_rows)
                        log(f"  [{_setting}] DeLong rows queued ({len(delong_rows)})")

                except Exception as _e:
                    log(f"  DeLong/comparison failed [{_setting}]: {_e}")
                    import traceback; log(traceback.format_exc())

    # ── Save DeLong results (written ONCE -- no stale append) ────────────────
    if all_delong_rows if "all_delong_rows" in dir() else False:
        pd.DataFrame(all_delong_rows).to_csv(
            f"{OUT}/delong_prfps_vs_fib4.csv", index=False)
        log(f"  Saved delong_prfps_vs_fib4.csv  ({len(all_delong_rows)} rows)")

    # ── Metrics visualisation: CI plots, ROC, PR, Calibration, Heatmap ───────
    if all_settings_results:
        log("Generating metrics visualisation figures ...")
        plot_metrics_with_ci(
            all_results_dict = all_settings_results,
            out_dir          = f"{results_dir}/figures/metrics_plots",
            baseline_model   = "LightGBM",
        )

    # ── External validation (Tapestry) -- includes pRFPS + FIB-4 comparison ────
    if global_models is not None:
        
        validate_external(
            setting_artifacts         = all_setting_artifacts,
            X_val                     = X_val,
            y_val_num                 = y_val_num,
            subgroups_val             = subgroups_val,
            df_val_raw                = df_val_raw,
            all_prfps_results         = all_prfps_results,
            mcb_fib4_by_setting       = all_mcb_fib4,
            mcb_train_data_by_setting = all_mcb_train_data,
            n_boot                    = min(n_boot, 300),
        )

    section("ALL EXPERIMENTS COMPLETE")
    log(f"Outputs -> {OUT}/")
    for root, _, files in os.walk(OUT):
        for f_name in sorted(files):
            log(f"  {os.path.relpath(os.path.join(root,f_name), OUT)}")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="MASLD Revised Pipeline -- all reviewer concerns addressed")
    parser.add_argument("--data",   nargs="+", required=False, default=["mcb_data.tsv"])
    parser.add_argument("--val",    nargs="+", required=True)
    parser.add_argument("--n_boot", type=int,  default=1000,
                        help="Bootstrap resamples (1000 for submission, 50 for quick test)")
    parser.add_argument("--sep",    type=str,  default=None, choices=["tab",","])

    args      = parser.parse_args()
    args.data = " ".join(args.data) if isinstance(args.data, list) else args.data
    args.val  = " ".join(args.val)  if isinstance(args.val,  list) else args.val
    if args.sep == "tab": args.sep = "\t"

    main(data_path=args.data, val_path=args.val, n_boot=args.n_boot)
