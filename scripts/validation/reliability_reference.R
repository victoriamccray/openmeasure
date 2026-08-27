# One-time generator for the R (psych::alpha) leg of the Reliability
# reference-validation numbers in docs/validation/reference-validation.md.
#
# Dev-only tool, not part of the OpenMeasure application. R and the psych
# package are NOT OpenMeasure dependencies; this script is run manually,
# once, to produce numbers that are then hardcoded into the validation
# doc and into automated regression tests (which do not call R).
#
# Run from the repo root after reliability_reference.py has written the
# dataset_*.csv files:
#     Rscript scripts/validation/reliability_reference.R

library(psych)

run_dataset <- function(name, path, split_half) {
  cat("\n===", name, "===\n")
  data <- read.csv(path)
  print(data)

  # check.keys = FALSE (also psych's own default) so items are never
  # auto reverse-scored - OpenMeasure never reverse-codes either, so
  # this keeps both sides comparing the same, un-recoded items.
  result <- psych::alpha(data, check.keys = FALSE)
  cat("\nraw_alpha:", result$total$raw_alpha, "\n")

  cat("\nCorrected item-total correlation (r.drop):\n")
  print(result$item.stats$r.drop)

  cat("\nAlpha if item dropped (raw_alpha column of alpha.drop):\n")
  print(result$alpha.drop$raw_alpha)

  if (split_half) {
    # Independently reproduce OpenMeasure's ODD/EVEN positional split
    # (not psych's random splitHalf), so the comparison uses the same
    # definition rather than a differently-defined statistic.
    n_items <- ncol(data)
    odd_cols <- seq(1, n_items, by = 2)
    even_cols <- seq(2, n_items, by = 2)
    odd_score <- rowSums(data[, odd_cols, drop = FALSE])
    even_score <- rowSums(data[, even_cols, drop = FALSE])
    r <- cor(odd_score, even_score)
    sb <- (2 * r) / (1 + r)
    cat("\nOdd/even split-half correlation:", r, "\n")
    cat("Spearman-Brown corrected:", sb, "\n")
  }
}

run_dataset(
  "Dataset A (3 items x 4 participants)",
  "scripts/validation/dataset_a.csv",
  split_half = FALSE
)

run_dataset(
  "Dataset B (4 items x 8 participants)",
  "scripts/validation/dataset_b.csv",
  split_half = TRUE
)

run_dataset(
  "Dataset C - high reliability (5 items x 8 participants)",
  "scripts/validation/dataset_c.csv",
  split_half = TRUE
)

run_dataset(
  "Dataset D - low/negative reliability (4 items x 8 participants)",
  "scripts/validation/dataset_d.csv",
  split_half = TRUE
)

run_dataset(
  "Dataset E - problematic reverse item (5 items x 8 participants)",
  "scripts/validation/dataset_e.csv",
  split_half = TRUE
)

run_dataset(
  "Dataset G - small edge case (2 items x 5 participants)",
  "scripts/validation/dataset_g.csv",
  split_half = FALSE
)

# Dataset F (missing data) is handled separately: psych::alpha()'s
# default use = "pairwise" does NOT match OpenMeasure's listwise
# deletion (analyze()'s documented missing-data handling). Both are
# run and printed explicitly so the difference is visible rather than
# silently picking whichever agrees.
cat("\n=== Dataset F - missing data (4 items x 10 participants) ===\n")
data_f <- read.csv("scripts/validation/dataset_f.csv")
print(data_f)

result_pairwise <- psych::alpha(data_f, check.keys = FALSE, use = "pairwise")
cat("\nraw_alpha with use='pairwise' (psych's default, NOT matched to OpenMeasure):",
    result_pairwise$total$raw_alpha, "\n")

result_complete <- psych::alpha(data_f, check.keys = FALSE, use = "complete.obs")
cat("raw_alpha with use='complete.obs' (matches OpenMeasure's listwise deletion):",
    result_complete$total$raw_alpha, "\n")
