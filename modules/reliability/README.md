# OpenMeasure Reliability

A simple reliability analysis tool for surveys and questionnaires.

## Features

- Cronbach's alpha
- Corrected item-total correlations
- Alpha if item deleted
- Split-half reliability
- Spearman-Brown correction
- Missing data handling (listwise deletion)

## Usage

1. Upload a CSV in wide format (one row per participant).
2. Select the survey items.
3. View reliability statistics and item diagnostics.

## Notes

Missing data is handled via listwise deletion.
Reverse-coded items should be recoded before uploading.
Alpha measures internal consistency, not validity. Use factor analysis to check dimensionality.


## References

Cronbach, L. J. (1951). Coefficient alpha and the internal structure of tests. Psychometrika, 16(3), 297–334.
Nunnally, J. C., & Bernstein, I. H. (1994). Psychometric Theory (3rd ed.). McGraw-Hill.
DeVellis, R. F., & Thorpe, C. T. (2021). Scale Development: Theory and Applications (5th ed.). Sage.
Tavakol, M., & Dennick, R. (2011). Making sense of Cronbach's alpha. International Journal of Medical Education, 2, 53–55.
Cortina, J. M. (1993). What is coefficient alpha? Journal of Applied Psychology, 78(1), 98–104.
Schmitt, N. (1996). Uses and abuses of coefficient alpha. Psychological Assessment, 8(4), 350–353.
Revelle, W., & Zinbarg, R. E. (2009). Coefficients alpha, beta, omega, and the glb. Psychometrika, 74(1), 145–154.


## License

MIT
