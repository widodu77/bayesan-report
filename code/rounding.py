"""One rounding rule for every number we show: in the report, in its figures and in the notebook.

Python's f"{x:.3f}" rounds the binary value of x, so 0.0055 (stored as 0.005499...) gives 0.005, while
pandas' round(3) gives 0.006. A p-value of 0.0055 could then read 0.005 in the report and 0.006 in the
notebook. Here x is rounded from its shortest decimal form (repr), halves up: 0.0055 gives 0.006
everywhere.
"""
from decimal import ROUND_HALF_UP, Decimal


def rounded(x, digits, scale=1):
    """scale * x rounded to `digits` decimals, halves away from zero, as a Decimal."""
    value = (Decimal(repr(float(x))) * Decimal(repr(scale))).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return value if value != 0 else abs(value)  # no "-0.00"


def fmt(x, digits, scale=1):
    """scale * x as text with `digits` decimals and thousands separators, e.g. fmt(0.5274, 1, 100) = '52.7'."""
    return f"{rounded(x, digits, scale):,.{digits}f}"
