# One-time generator for the R (psych::alpha) leg of the Reliability
# reference-validation numbers in docs/validation/reference-validation.md.
#
# Dev-only tool, not part of the OpenMeasure application. R and the psych
# package are NOT OpenMeasure dependencies; this script is run manually,
# once, to produce numbers that are then hardcoded into the validation
# doc and into automated regression tests (which do not call R).
#
# Run from the repo root after reliability_reference.py has written
# dataset_a.csv / dataset_b.csv:
#     Rscript scripts/validation/reliability_reference.R

library(psych)

run_dataset <- function(name, path, split_half) {
  cat("\n===", name, "===\n")
  data <- read.csv(path)
  print(data)

  result <- psych::alpha(data)
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
