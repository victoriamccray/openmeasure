# OpenMeasure Reliability

A simple reliability analysis tool for surveys and questionnaires.

## Features

- Cronbach's alpha
- Corrected item-total correlations
- Alpha if item deleted
- Split-half reliability
- Spearman-Brown correction
- Missing data handling (listwise deletion)
- Multi-rater categorical agreement (`core/interrater.py`): Cohen's kappa
  (two raters), Fleiss' kappa (three or more raters, complete coverage
  only), and Krippendorff's alpha (any number of raters, tolerates
  missing/partial coverage). A different reliability question from the
  scale-level statistics above: whether independent raters apply the
  same categorical judgment consistently, not whether items measure one
  construct. Cohen's/Fleiss' kappa are computed via `statsmodels`;
  Krippendorff's alpha via the `krippendorff` package -- established,
  independently maintained implementations, not reimplemented here.

## Usage

1. Upload a CSV in wide format (one row per participant).
2. Select the survey items.
3. View reliability statistics and item diagnostics.

## Notes

Missing data is handled via listwise deletion.
Reverse-coded items should be recoded before uploading.
Alpha measures internal consistency, not validity. Use factor analysis to check dimensionality.


## References

- Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16(3), 297–334.
- Nunnally, J. C., & Bernstein, I. H. (1994). Psychometric Theory (3rd ed.). McGraw-Hill.
- DeVellis, R. F., & Thorpe, C. T. (2021). Scale Development: Theory and Applications (5th ed.). Sage.
- Tavakol, M., & Dennick, R. (2011). Making sense of Cronbach's alpha. International Journal of Medical Education, 2, 53–55.
- Cortina, J. M. (1993). What is coefficient alpha? Journal of Applied Psychology, 78(1), 98–104.
- Schmitt, N. (1996). Uses and abuses of coefficient alpha. Psychological Assessment, 8(4), 350–353.
- Revelle, W., & Zinbarg, R. E. (2009). Coefficients alpha, beta, omega, and the glb. Psychometrika, 74(1), 145–154.
- Cohen, J. (1960). A coefficient of agreement for nominal scales. Educational and Psychological Measurement, 20(1), 37–46.
- Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. Psychological Bulletin, 76(5), 378–382.
- Landis, J. R., & Koch, G. G. (1977). The measurement of observer agreement for categorical data. Biometrics, 33(1), 159–174.
- Krippendorff, K. (2004). Content Analysis: An Introduction to Its Methodology (2nd ed.). Sage. (Krippendorff's alpha)
- Hayes, A. F., & Krippendorff, K. (2007). Answering the call for a standard reliability measure for coding data. Communication Methods and Measures, 1(1), 77–89.


## License

MIT
