# MASLD subgroup analyses: descriptive tables and publication figures
#
# Usage:
#   1. Put the two tab-delimited input files in data/ (or set DATA_DIR).
#   2. Run: Rscript masld_figures.R
#
# Outputs are written to outputs/. This script does not contain patient data.

# ---- Package checks ---------------------------------------------------------

required_packages <- c(
  "dplyr", "fmsb", "ggalluvial", "ggplot2", "ggpubr", "ggsci",
  "glue", "gtsummary", "gt", "purrr", "rlang", "scales", "tibble"
)

missing_packages <- required_packages[!vapply(
  required_packages,
  requireNamespace,
  logical(1),
  quietly = TRUE
)]

if (length(missing_packages) > 0) {
  stop(
    "Install the required R packages first: ",
    paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(fmsb)
  library(ggalluvial)
  library(ggplot2)
  library(ggpubr)
  library(ggsci)
  library(gtsummary)
  library(gt)
  library(purrr)
  library(scales)
  library(tibble)
})

# ---- Project configuration --------------------------------------------------

data_dir <- Sys.getenv("DATA_DIR", unset = "data")
output_dir <- Sys.getenv("OUTPUT_DIR", unset = "outputs")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

input_files <- list(
  MCB = file.path(data_dir, "mcb.tsv"),
  Tapestry = file.path(data_dir, "tapestry.tsv")
)

if (!all(file.exists(unlist(input_files)))) {
  stop(
    "Input files not found. Expected: ",
    paste(unlist(input_files), collapse = ", "),
    "\nSet DATA_DIR to use a different data directory.",
    call. = FALSE
  )
}

palette <- list(
  subgroup = c(C1 = "#E76F51", C2 = "#2A9D8F"),
  timing = c("ICD-based" = "#4A90A4", "FIB-4-based" = "#E9C46A", Unclassified = "#CCCCCC"),
  progression = c(Rapid = "#E63946", Slow = "#457B9D", No = "#A8DADC")
)

read_cohort <- function(path) {
  read.delim(path, header = TRUE, quote = "", check.names = FALSE)
}

prepare_cohort <- function(data, dataset_name) {
  data %>%
    filter(Sex %in% c("Male", "Female")) %>%
    mutate(
      Dataset = dataset_name,
      Subgroup_label = if_else(Subgroup_5 == "C1", "C1", "C2"),
      Subgroup_label = factor(Subgroup_label, levels = c("C1", "C2"))
    )
}

safe_select <- function(data, variables) {
  select(data, any_of(variables))
}

characteristic_labels <- list(
  diagnosis_age ~ "Age (years)", Avg_BMI ~ "BMI (kg/m²)",
  HbA1C ~ "HbA1c (%)", ALT.x ~ "ALT (U/L)", AST.x ~ "AST (U/L)",
  ALP ~ "ALP (U/L)", PLATELET_COUNT ~ "Platelet count (×10⁹/L)",
  ALBUMIN ~ "Albumin (g/dL)", TRIGLYCERIDE ~ "Triglycerides (mg/dL)",
  HDL ~ "HDL (mg/dL)", LDL ~ "LDL (mg/dL)", BUN ~ "BUN (mg/dL)",
  first_FIB4 ~ "Initial FIB-4 stage", NFS ~ "NFS", PRS_3 ~ "PRS-3",
  fibrosis_cirrhosis ~ "Fibrosis/cirrhosis (ICD-based)",
  diabetes ~ "Diabetes", hypertension ~ "Hypertension",
  dyslipidemia ~ "Dyslipidemia", depression ~ "Depression",
  heart ~ "Heart disease", ckd ~ "Kidney disease",
  PNPLA3 ~ "PNPLA3", TM6SF2 ~ "TM6SF2", HSD17B13 ~ "HSD17B13",
  GCKR ~ "GCKR", MBOAT7 ~ "MBOAT7", CMR_count ~ "CMR count",
  n_followups ~ "Number of follow-ups"
)

# ---- Table 1 ----------------------------------------------------------------

make_characteristics_table <- function(data, dataset_name) {
  variables <- c(
    "diagnosis_age", "Sex", "Avg_BMI", "HbA1C", "MASH", "ALT.x", "AST.x",
    "ALP", "PLATELET_COUNT", "ALBUMIN", "TRIGLYCERIDE", "HDL", "LDL", "BUN",
    "hcc", "transplant", "first_FIB4", "NFS", "fibrosis_cirrhosis", "PRS_3",
    "diabetes", "hypertension", "dyslipidemia", "depression", "heart", "ckd",
    "PNPLA3", "TM6SF2", "HSD17B13", "GCKR", "MBOAT7", "CMR_count", "n_followups"
  )
  
  table_data <- data %>%
    filter(overall_progression_category2 %in% c("Rapid Progression", "Slow Progression")) %>%
    safe_select(c(variables, "Subgroup_label", "overall_progression_category2")) %>%
    mutate(
      across(where(is.character), as.factor),
      overall_progression_category2 = factor(
        overall_progression_category2,
        levels = c("Rapid Progression", "Slow Progression")
      )
    )
  
  table_data %>%
    tbl_strata(
      strata = Subgroup_label,
      .tbl_fun = ~ .x %>%
        tbl_summary(
          by = overall_progression_category2,
          missing = "no",
          statistic = list(
            all_continuous() ~ "{median} ({p25}, {p75})",
            all_categorical() ~ "{n} ({p}%)"
          ),
          digits = all_continuous() ~ 1,
          label = characteristic_labels
        ) %>%
        add_overall(last = FALSE) %>%
        add_p(test = list(
          all_continuous() ~ "wilcox.test",
          all_categorical() ~ "chisq.test"
        )) %>%
        bold_p(t = 0.05) %>%
        modify_header(
          label ~ "**Characteristic**", stat_0 ~ "**Overall**",
          stat_1 ~ "**Rapid progression**", stat_2 ~ "**Slow progression**",
          p.value ~ "**p-value**"
        ) %>%
        modify_footnote(
          all_stat_cols() ~ "Median (IQR) for continuous variables; n (%) for categorical variables.",
          p.value ~ "Wilcoxon rank-sum test (continuous); chi-square test (categorical)."
        )
    ) %>%
    modify_caption(glue::glue("**Table 1. Patient characteristics — {dataset_name}**"))
}

save_characteristics_table <- function(data, dataset_name) {
  table <- make_characteristics_table(data, dataset_name)
  output_file <- file.path(output_dir, paste0("table1_", tolower(dataset_name), ".html"))
  gtsave(as_gt(table), output_file)
  message("Saved: ", output_file)
  invisible(table)
}

# ---- Radar chart ------------------------------------------------------------

radar_variables <- c("HbA1C", "ALT.x", "diagnosis_age", "Avg_BMI", "LDL", "TRIGLYCERIDE")
radar_labels <- c("HbA1c (%)", "ALT (U/L)", "Age (years)", "BMI (kg/m²)", "LDL (mg/dL)", "Triglycerides (mg/dL)")

make_radar_chart <- function(data, dataset_name) {
  cohort <- data %>% filter(!is.na(Subgroup_label))
  group_means <- cohort %>%
    safe_select(c("Subgroup_label", radar_variables)) %>%
    group_by(Subgroup_label) %>%
    summarise(across(all_of(radar_variables), ~ mean(.x, na.rm = TRUE)), .groups = "drop") %>%
    column_to_rownames("Subgroup_label")
  
  ranges <- cohort %>%
    safe_select(radar_variables) %>%
    summarise(across(everything(), list(min = ~ min(.x, na.rm = TRUE), max = ~ max(.x, na.rm = TRUE))))
  
  scaled <- group_means
  for (variable in radar_variables) {
    minimum <- ranges[[paste0(variable, "_min")]]
    maximum <- ranges[[paste0(variable, "_max")]]
    scaled[[variable]] <- if (maximum > minimum) (scaled[[variable]] - minimum) / (maximum - minimum) else 0
  }
  
  plot_data <- rbind(maximum = rep(1, ncol(scaled)), minimum = rep(0, ncol(scaled)), scaled)
  colnames(plot_data) <- radar_labels
  output_file <- file.path(output_dir, paste0("radar_", tolower(dataset_name), ".png"))
  
  png(output_file, width = 1800, height = 1600, res = 220)
  on.exit(dev.off(), add = TRUE)
  par(mar = c(2, 2, 4, 2), family = "sans")
  radarchart(
    plot_data, axistype = 1, seg = 4,
    pcol = unname(palette$subgroup[rownames(scaled)]),
    pfcol = alpha(unname(palette$subgroup[rownames(scaled)]), 0.25),
    plwd = 3.5, plty = 1, cglcol = "grey85", cglty = 1, cglwd = 0.6,
    caxislabels = c("0", "0.25", "0.5", "0.75", "1"), axislabcol = "grey35",
    calcex = 0.75, vlcex = 0.95,
    title = paste("Subgroup profile comparison —", dataset_name)
  )
  legend("topright", legend = rownames(scaled), col = unname(palette$subgroup[rownames(scaled)]),
         lty = 1, lwd = 4, bty = "n", cex = 1.05, text.col = "grey20")
  message("Saved: ", output_file)
}

# ---- Progression timing -----------------------------------------------------

make_progression_timing_plots <- function(data_list, dataset_names) {
  timing_summary <- map2_dfr(data_list, dataset_names, ~ .x %>%
                               filter(!is.na(overall_progression_category2)) %>%
                               transmute(Dataset = .y, Subgroup_label, timing_source = age_diff_diagnosis) %>%
                               count(Dataset, Subgroup_label, timing_source) %>%
                               group_by(Dataset, Subgroup_label) %>%
                               mutate(pct = 100 * n / sum(n)) %>%
                               ungroup()
  )
  
  bar_data <- timing_summary %>%
    mutate(
      Subgroup_label = factor(Subgroup_label, levels = c("C1", "C2")),
      timing_source = factor(timing_source, levels = names(palette$timing))
    )
  
  timing_plot <- ggplot(bar_data, aes(Subgroup_label, pct, fill = timing_source)) +
    geom_col(width = 0.5, colour = "white", linewidth = 0.4) +
    geom_text(aes(label = if_else(pct >= 5, paste0(round(pct, 1), "%"), "")),
              position = position_stack(vjust = 0.5), size = 3.5, fontface = "bold", colour = "white") +
    scale_fill_manual(values = palette$timing, name = NULL) +
    scale_y_continuous(labels = (x) paste0(x, "%"), limits = c(0, 100), expand = c(0, 0)) +
    labs(x = "Subgroup", y = "Percentage of patients") +
    theme_classic(base_size = 12) +
    theme(legend.position = "top")
  
  raw_data <- map2_dfr(data_list, dataset_names, ~ bind_rows(
    .x %>% filter(age_diff_diagnosis >= 0.5) %>% transmute(Dataset = .y, Subgroup_label, age_diff_diagnosis),
    .x %>% filter(age_diff_diagnosis >= 0.5) %>% transmute(Dataset = .y, Subgroup_label = "Overall", age_diff_diagnosis)
  )) %>%
    mutate(Subgroup_label = factor(Subgroup_label, levels = c("Overall", "C1", "C2")))
  
  comparisons <- list(c("Overall", "C1"), c("Overall", "C2"), c("C1", "C2"))
  time_plot <- ggplot(raw_data, aes(Subgroup_label, age_diff_diagnosis, fill = Subgroup_label)) +
    geom_boxplot(outlier.shape = NA, alpha = 0.6, width = 0.5) +
    geom_jitter(aes(colour = Subgroup_label), width = 0.15, size = 1, alpha = 0.35) +
    stat_compare_means(comparisons = comparisons, method = "wilcox.test", p.adjust.method = "bonferroni",
                       label = "p.format", tip.length = 0.01, size = 3.5) +
    scale_fill_manual(values = setNames(pal_jama()(3), c("Overall", "C1", "C2"))) +
    scale_colour_manual(values = setNames(pal_jama()(3), c("Overall", "C1", "C2"))) +
    labs(x = NULL, y = "Progression time (years)") +
    theme_classic(base_size = 12) +
    theme(legend.position = "none")
  
  suffix <- tolower(paste(dataset_names, collapse = "_"))
  ggsave(file.path(output_dir, paste0("timing_source_", suffix, ".png")), timing_plot,
         width = 3 * length(data_list), height = 3, dpi = 400, bg = "white")
  ggsave(file.path(output_dir, paste0("progression_time_", suffix, ".png")), time_plot,
         width = 3 * length(data_list), height = 3.5, dpi = 400, bg = "white")
  invisible(list(timing_summary = timing_summary, timing_plot = timing_plot, time_plot = time_plot))
}

# ---- Alluvial plot ----------------------------------------------------------

make_alluvial_plot <- function(data, dataset_name) {
  stage_levels <- c("Low Risk", "Intermediate Risk", "High Risk")
  progression_levels <- c("Rapid Progression", "Slow Progression", "No Progression")
  
  alluvial_data <- data %>%
    mutate(fib4_stage_3 = case_when(
      first_stage == "F0" ~ "Low Risk",
      first_stage == "F1-F2" ~ "Intermediate Risk",
      first_stage == "F3-F4" ~ "High Risk",
      TRUE ~ NA_character_
    )) %>%
    filter(!is.na(fib4_stage_3), !is.na(overall_progression_category2), !is.na(Subgroup_label)) %>%
    mutate(
      Subgroup_label = factor(Subgroup_label, levels = c("C1", "C2")),
      fib4_stage_3 = factor(fib4_stage_3, levels = stage_levels),
      overall_progression_category2 = factor(overall_progression_category2, levels = progression_levels)
    ) %>%
    count(Subgroup_label, fib4_stage_3, overall_progression_category2) %>%
    group_by(Subgroup_label) %>%
    mutate(pct = 100 * n / sum(n), flow_label = if_else(pct >= 1, paste0(round(pct, 1), "%"), "")) %>%
    ungroup()
  
  plot <- ggplot(alluvial_data, aes(
    axis1 = Subgroup_label, axis2 = fib4_stage_3, axis3 = overall_progression_category2, y = pct
  )) +
    stat_alluvium(aes(fill = overall_progression_category2), width = 0.3, alpha = 0.65, knot.pos = 0.45) +
    stat_stratum(aes(fill = after_stat(stratum)), width = 0.5, colour = "black", linewidth = 0.1, alpha = 0.92) +
    geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3, lineheight = 0.7) +
    scale_fill_manual(values = c(
      C1 = palette$subgroup[["C1"]], C2 = palette$subgroup[["C2"]],
      "Low Risk" = "#2A9D8F", "Intermediate Risk" = "#E9C46A", "High Risk" = "#E63946",
      "Rapid Progression" = palette$progression[["Rapid"]],
      "Slow Progression" = palette$progression[["Slow"]],
      "No Progression" = palette$progression[["No"]]
    ), guide = "none") +
    scale_x_discrete(limits = c("Subgroup", "Initial FIB-4 stage", "Progression category"),
                     expand = expansion(add = c(0.02, 0.05))) +
    scale_y_continuous(labels = (x) paste0(x, "%"), expand = c(0, 0)) +
    labs(x = NULL, y = "Percentage of patients") +
    theme_minimal(base_size = 9) +
    theme(panel.grid = element_blank(), axis.text.x = element_text(colour = "black"))
  
  output_file <- file.path(output_dir, paste0("alluvial_", tolower(dataset_name), ".png"))
  ggsave(output_file, plot, width = 4.5, height = 3.2, dpi = 400, bg = "white")
  message("Saved: ", output_file)
  invisible(plot)
}

# ---- Run analysis -----------------------------------------------------------

mcb <- prepare_cohort(read_cohort(input_files$MCB), "MCB")
tapestry <- prepare_cohort(read_cohort(input_files$Tapestry), "Tapestry")
combined <- bind_rows(mcb, tapestry)

walk2(list(mcb, tapestry), c("MCB", "Tapestry"), save_characteristics_table)
walk2(list(mcb, tapestry), c("MCB", "Tapestry"), make_radar_chart)
make_progression_timing_plots(list(mcb, tapestry), c("MCB", "Tapestry"))
walk2(list(mcb, tapestry, combined), c("MCB", "Tapestry", "Combined"), make_alluvial_plot)
