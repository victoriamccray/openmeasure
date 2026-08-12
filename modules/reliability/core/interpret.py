"""
Turns raw statistics into plain-language interpretation.

Thresholds follow the commonly cited conventions (George & Mallery, 2003;
Nunnally, 1978). These are conventions, not laws, the docstrings say so.
"""

from __future__ import annotations


def interpret_alpha(alpha: float) -> str:
    """
    Standard Cronbach's alpha interpretation thresholds:
        >= 0.90  Excellent
        >= 0.80  Good
        >= 0.70  Acceptable
        >= 0.60  Questionable
        >= 0.50  Poor
        <  0.50  Unacceptable
    """
    if alpha >= 0.90:
        return "Excellent internal consistency"
    if alpha >= 0.80:
        return "Good internal consistency"
    if alpha >= 0.70:
        return "Acceptable internal consistency"
    if alpha >= 0.60:
        return "Questionable internal consistency"
    if alpha >= 0.50:
        return "Poor internal consistency"
    return "Unacceptable internal consistency"


def alpha_warnings(alpha: float) -> list[str]:
    warnings = []
    if alpha > 0.95:
        warnings.append(
            "Alpha above 0.95 may indicate item redundancy, consider whether "
            "some items are just restating each other rather than measuring "
            "distinct facets of the construct."
        )
    return warnings


def interpret_agreement(value: float) -> str:
    """
    Landis & Koch (1977) agreement thresholds, originally proposed for
    kappa statistics and conventionally applied to Krippendorff's alpha
    as well:
        >= 0.81  Almost perfect agreement
        >= 0.61  Substantial agreement
        >= 0.41  Moderate agreement
        >= 0.21  Fair agreement
        >= 0.00  Slight agreement
        <  0.00  Poor agreement

    Landis, J. R., & Koch, G. G. (1977). The measurement of observer
    agreement for categorical data. Biometrics, 33(1), 159-174.
    """
    if value >= 0.81:
        return "Almost perfect agreement"
    if value >= 0.61:
        return "Substantial agreement"
    if value >= 0.41:
        return "Moderate agreement"
    if value >= 0.21:
        return "Fair agreement"
    if value >= 0.00:
        return "Slight agreement"
    return "Poor agreement"


def item_warning(item_total_corr: float, threshold: float = 0.30) -> str | None:
    if item_total_corr != item_total_corr:  # NaN check without importing numpy
        return "Item has zero variance or could not be correlated."
    if item_total_corr < threshold:
        return f"Low item-total correlation ({item_total_corr:.2f}), consider reviewing or removing this item."
    return None
