# =============================================================================
# pRFPS Subgroup-Specific Score Computation + HCC / Transplant Analysis
# =============================================================================

# ── 0. Libraries ─────────────────────────────────────────────────────────────
required_packages <- c("dplyr", "ggplot2", "tidyr")
missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)]
if (length(missing_packages) > 0) {
  stop("Install the required R packages first: ", paste(missing_packages, collapse = ", "), call. = FALSE)
}
suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(tidyr)
})

# ── 1. Load data ─────────────────────────────────────────────────────────────
# Store local, non-versioned files as data/mcb.tsv and data/tapestry.tsv.
# Set DATA_DIR or OUTPUT_DIR to override the default directories.
data_dir <- Sys.getenv("DATA_DIR", unset = "data")
out_dir <- Sys.getenv("OUTPUT_DIR", unset = "outputs")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

mcb_data_path <- file.path(data_dir, "mcb.tsv")
tap_data_path <- file.path(data_dir, "tapestry.tsv")
if (!all(file.exists(c(mcb_data_path, tap_data_path)))) {
  stop("Input files not found. Expected: ", mcb_data_path, " and ", tap_data_path, call. = FALSE)
}

MCB <- read.delim(mcb_data_path, header = TRUE, quote = "", check.names = FALSE)
Tapestry <- read.delim(tap_data_path, header = TRUE, quote = "", check.names = FALSE)
cat("MCB dimensions:      ", nrow(MCB), " rows x ", ncol(MCB), " cols\n", sep = "")
cat("Tapestry dimensions: ", nrow(Tapestry), " rows x ", ncol(Tapestry), " cols\n", sep = "")

# ── 2. Helper: compute pRFPS and classify risk ────────────────────────────────
# Expected score columns: Subgroup_5, diagnosis_age, BUN, ALP, PLATELET_COUNT,
# HDL, LDL, ALBUMIN, and AST.x. Update names here if your source data differ.
required_score_columns <- c("Subgroup_5", "diagnosis_age", "BUN", "ALP", "PLATELET_COUNT", "HDL", "LDL", "ALBUMIN", "AST.x")

check_columns <- function(data, columns, data_name) {
  missing_columns <- setdiff(columns, names(data))
  if (length(missing_columns) > 0) {
    stop(data_name, " is missing required column(s): ", paste(missing_columns, collapse = ", "), call. = FALSE)
  }
}

compute_pRFPS <- function(data, cohort_name) {
  check_columns(data, required_score_columns, cohort_name)
  data %>%
    mutate(
      # C1 = 10*I(Age >45) + 5*I(BUN >21) + 5*I(ALP >109) + 3*I(Platelet <125) + 3*I(HDL <61)
      pRFPS_C1 = if_else(Subgroup_5 == "C1",
                         10 * as.integer(diagnosis_age > 45) + 5 * as.integer(BUN > 21) +
                           5 * as.integer(ALP > 109) + 3 * as.integer(PLATELET_COUNT < 125) +
                           3 * as.integer(HDL < 61), NA_real_),
      pRFPS_C1_risk = case_when(is.na(pRFPS_C1) ~ NA_character_, pRFPS_C1 >= 15 ~ "High", TRUE ~ "Low"),
      # C2 = 10*I(ALP >97) + 6*I(Platelet <313) + I(LDL >64) + I(Albumin <13) + I(AST >39)
      pRFPS_C2 = if_else(Subgroup_5 == "C2",
                         10 * as.integer(ALP > 97) + 6 * as.integer(PLATELET_COUNT < 313) +
                           as.integer(LDL > 64) + as.integer(ALBUMIN < 13) + as.integer(AST.x > 39), NA_real_),
      pRFPS_C2_risk = case_when(is.na(pRFPS_C2) ~ NA_character_, pRFPS_C2 >= 18 ~ "High", TRUE ~ "Low"),
      pRFPS_risk = coalesce(pRFPS_C1_risk, pRFPS_C2_risk)
    )
}

MCB <- compute_pRFPS(MCB, "MCB")
Tapestry <- compute_pRFPS(Tapestry, "Tapestry")

# ── 3. Score distribution ────────────────────────────────────────────────────
score_summary <- function(data, cohort_name) {
  cat("\n", strrep("=", 60), "\nCOHORT: ", cohort_name, "\n", strrep("=", 60), "\n", sep = "")
  for (subgroup in c("C1", "C2")) {
    score_column <- paste0("pRFPS_", subgroup)
    risk_column <- paste0("pRFPS_", subgroup, "_risk")
    subset_data <- filter(data, Subgroup_5 == subgroup)
    if (nrow(subset_data) == 0) next
    cat("\n--- Subgroup ", subgroup, " (n = ", nrow(subset_data), ") ---\n", sep = "")
    cat("Score range: ", min(subset_data[[score_column]], na.rm = TRUE), "–", max(subset_data[[score_column]], na.rm = TRUE), "\n", sep = "")
    cat("Score mean (SD): ", round(mean(subset_data[[score_column]], na.rm = TRUE), 2), " (", round(sd(subset_data[[score_column]], na.rm = TRUE), 2), ")\n", sep = "")
    print(table(subset_data[[risk_column]], useNA = "ifany"))
  }
}
score_summary(MCB, "MCB")
score_summary(Tapestry, "Tapestry")

# ── 4. HCC and transplant rates by pRFPS risk group ─────────────────────────
to_binary_outcome <- function(x) {
  values <- toupper(trimws(as.character(x)))
  if_else(is.na(x), NA_integer_, as.integer(values %in% c("1", "YES", "Y", "TRUE")))
}

outcome_analysis <- function(data, cohort_name, hcc_col = "hcc", transplant_col = "transplant") {
  outcomes <- c(HCC = hcc_col, Transplant = transplant_col)
  outcomes <- outcomes[outcomes %in% names(data)]
  cat("\n", strrep("=", 60), "\nOUTCOME ANALYSIS: ", cohort_name, "\n", strrep("=", 60), "\n", sep = "")
  if (length(outcomes) == 0) return(message("No outcome columns found; skipping."))
  for (outcome in unique(outcomes)) data[[outcome]] <- to_binary_outcome(data[[outcome]])
  for (subgroup in c("C1", "C2")) {
    risk_column <- paste0("pRFPS_", subgroup, "_risk")
    subset_data <- filter(data, Subgroup_5 == subgroup, !is.na(.data[[risk_column]]))
    cat("\n--- Subgroup ", subgroup, " ---\n", sep = "")
    for (outcome_name in names(outcomes)) {
      outcome <- outcomes[[outcome_name]]
      summary_table <- subset_data %>% group_by(Risk = .data[[risk_column]]) %>%
        summarise(N = n(), Events = sum(.data[[outcome]], na.rm = TRUE), Event_pct = round(mean(.data[[outcome]], na.rm = TRUE) * 100, 1), .groups = "drop")
      cat(outcome_name, " by pRFPS risk:\n", sep = "")
      print(summary_table)
    }
  }
}

# ── 5. Odds-ratio analysis ───────────────────────────────────────────────────
fit_odds_ratio <- function(data, outcome, covariates = character()) {
  analysis_columns <- c(outcome, "Risk_bin", covariates)
  analysis_data <- data[complete.cases(data[, analysis_columns, drop = FALSE]), , drop = FALSE]
  if (nrow(analysis_data) == 0 || length(unique(analysis_data[[outcome]])) < 2) return(NULL)
  model <- tryCatch(glm(reformulate(c("Risk_bin", covariates), outcome), data = analysis_data, family = binomial()), error = function(e) NULL)
  if (is.null(model) || !"Risk_bin" %in% names(coef(model))) return(NULL)
  confidence_interval <- exp(confint.default(model)["Risk_bin", ])
  list(OR = exp(coef(model)["Risk_bin"]), CI_low = confidence_interval[1], CI_high = confidence_interval[2], p_value = coef(summary(model))["Risk_bin", "Pr(>|z|)"], N = nrow(analysis_data))
}

odds_ratio_analysis <- function(data, cohort_name, hcc_col = "hcc", transplant_col = "transplant") {
  outcomes <- c(HCC = hcc_col, Transplant = transplant_col)
  outcomes <- outcomes[outcomes %in% names(data)]
  for (outcome in unique(outcomes)) data[[outcome]] <- to_binary_outcome(data[[outcome]])
  demographic_covariates <- c("diagnosis_age", "Sex")
  genetic_candidates <- c("PRS_3", "PNPLA3", "TM6SF2", "HSD17B13", "PNPLA3_TM6SF2_MBOAT7_GCKR_HSD17B13", "DISCORD_3_SNP_SUM")
  results <- list()
  cat("\n", strrep("=", 60), "\nODDS-RATIO ANALYSIS: ", cohort_name, "\n", strrep("=", 60), "\n", sep = "")
  for (subgroup in c("C1", "C2")) {
    risk_column <- paste0("pRFPS_", subgroup, "_risk")
    subset_data <- data %>% filter(Subgroup_5 == subgroup, !is.na(.data[[risk_column]])) %>% mutate(Risk_bin = as.integer(.data[[risk_column]] == "High"))
    if (nrow(subset_data) < 10) next
    demographics <- intersect(demographic_covariates, names(subset_data))
    genetics <- intersect(genetic_candidates, names(subset_data))
    genetics <- genetics[vapply(genetics, function(x) mean(!is.na(subset_data[[x]])) >= 0.80, logical(1))]
    models <- list(Unadjusted = character(), `Adjusted (age + sex)` = demographics, `Adjusted (age + sex + genetics)` = c(demographics, genetics))
    cat("\n--- Subgroup ", subgroup, " (reference = low risk) ---\n", sep = "")
    for (outcome_name in names(outcomes)) {
      outcome <- outcomes[[outcome_name]]
      if (sum(subset_data[[outcome]], na.rm = TRUE) < 3) next
      for (model_name in names(models)) {
        if (model_name == "Adjusted (age + sex + genetics)" && length(genetics) == 0) next
        result <- fit_odds_ratio(subset_data, outcome, models[[model_name]])
        if (is.null(result)) next
        cat(sprintf("%s, %s: OR %.2f (95%% CI %.2f–%.2f), p = %.4f\n", outcome_name, model_name, result$OR, result$CI_low, result$CI_high, result$p_value))
        results[[length(results) + 1]] <- cbind(data.frame(Cohort = cohort_name, Subgroup = subgroup, Outcome = outcome_name, Model = model_name), as.data.frame(result))
      }
    }
  }
  bind_rows(results)
}

outcome_analysis(MCB, "MCB")
outcome_analysis(Tapestry, "Tapestry")
mcb_or <- odds_ratio_analysis(MCB, "MCB")
tapestry_or <- odds_ratio_analysis(Tapestry, "Tapestry")

# ── 6. Fibrosis-stage odds-ratio analysis ────────────────────────────────────
odds_ratio_stage_analysis <- function(data, cohort_name, hcc_col = "hcc", transplant_col = "transplant", stage_col = "first_stage", subgroup_col = "Subgroup_5") {
  check_columns(data, c(stage_col, subgroup_col), cohort_name)
  outcomes <- c(HCC = hcc_col, Transplant = transplant_col)
  outcomes <- outcomes[outcomes %in% names(data)]
  for (outcome in unique(outcomes)) data[[outcome]] <- to_binary_outcome(data[[outcome]])
  data <- data %>% mutate(stage_clean = toupper(trimws(as.character(.data[[stage_col]]))),
                          Stage_risk = case_when(
                            stage_clean %in% c("F0", "0", "F0.0") ~ "Low",
                            stage_clean %in% c("F1", "1", "F1.0", "F2", "2", "F2.0", "F1-F2") ~ "Intermediate",
                            stage_clean %in% c("F3", "3", "F3.0", "F4", "4", "F4.0", "F3-F4") ~ "High",
                            TRUE ~ NA_character_))
  cat("\nFIBROSIS-STAGE OR ANALYSIS: ", cohort_name, " (F3–F4 versus F0)\n", sep = "")
  for (subgroup in c("C1", "C2")) {
    subset_data <- data %>% filter(.data[[subgroup_col]] == subgroup, Stage_risk %in% c("Low", "High")) %>% mutate(Risk_bin = as.integer(Stage_risk == "High"))
    for (outcome_name in names(outcomes)) {
      result <- fit_odds_ratio(subset_data, outcomes[[outcome_name]])
      if (!is.null(result)) cat(sprintf("%s, %s: OR %.2f (95%% CI %.2f–%.2f), p = %.4f\n", subgroup, outcome_name, result$OR, result$CI_low, result$CI_high, result$p_value))
    }
  }
}
odds_ratio_stage_analysis(MCB, "MCB")

# ── 7. Save scored datasets and odds-ratio results ───────────────────────────
write.table(MCB, file.path(out_dir, "MCB_with_pRFPS.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(Tapestry, file.path(out_dir, "Tapestry_with_pRFPS.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
or_results <- bind_rows(mcb_or, tapestry_or)
write.csv(or_results, file.path(out_dir, "pRFPS_odds_ratios.csv"), row.names = FALSE)

# =============================================================================
# FOREST PLOT — OR for HCC and Transplant by pRFPS risk group
# =============================================================================
forest_plot_pRFPS <- function(or_results, output_directory = out_dir) {
  if (nrow(or_results) == 0) return(warning("No estimable odds ratios; forest plot was not created."))
  colors <- c("Unadjusted" = "#2A78D6", "Adjusted (age + sex)" = "#EB6834", "Adjusted (age + sex + genetics)" = "#1BAF7A")
  shapes <- c("Unadjusted" = 16, "Adjusted (age + sex)" = 15, "Adjusted (age + sex + genetics)" = 17)
  plot_data <- or_results %>% mutate(Model = factor(Model, levels = names(colors)), Label = paste(Subgroup, Outcome, sep = " | "), Facet = paste(Cohort, "cohort"), OR_text = sprintf("%.2f [%.2f–%.2f]", OR, CI_low, CI_high))
  annotation_x <- max(plot_data$CI_high, na.rm = TRUE) * 1.10
  plot <- ggplot(plot_data, aes(x = OR, y = Label, colour = Model, shape = Model)) +
    geom_vline(xintercept = 1, linetype = "dashed", colour = "grey60") +
    geom_errorbar(aes(xmin = CI_low, xmax = CI_high), orientation = "y", width = 0.25, position = position_dodge(width = 0.6)) +
    geom_point(size = 3, position = position_dodge(width = 0.6)) +
    geom_text(aes(x = annotation_x, label = OR_text), hjust = 0, size = 2.8, colour = "grey30", position = position_dodge(width = 0.6)) +
    scale_x_log10(expand = expansion(mult = c(0.02, 0.30))) + scale_colour_manual(values = colors, name = NULL) + scale_shape_manual(values = shapes, name = NULL) +
    facet_wrap(~Facet, ncol = 2) + labs(title = "pRFPS High Risk versus Low Risk: HCC and Liver Transplant", subtitle = "Odds ratios (95% CI); reference = low risk", x = "Odds ratio (log scale)", y = NULL, caption = "C1 cutoff ≥15; C2 cutoff ≥18") +
    theme_bw(base_size = 11) + theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(), legend.position = "bottom", plot.title = element_text(face = "bold"))
  ggsave(file.path(output_directory, "pRFPS_forest_plot.pdf"), plot, width = 14, height = 7, units = "in")
  ggsave(file.path(output_directory, "pRFPS_forest_plot.png"), plot, width = 14, height = 7, units = "in", dpi = 300, bg = "white")
}

# ── Run forest plot ──────────────────────────────────────────────────────────
forest_plot_pRFPS(or_results)
