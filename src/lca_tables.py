"""
lca_tables.py — LCA sensitivity, included/excluded table, comparison with prior studies.

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

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/comparison_prior_studies.csv", index=False)
    log("  Saved comparison_prior_studies.csv")
    return df

