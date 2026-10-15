"""Score a predicted number against the label.

strict:    within 1% of the label (relative tolerance absorbs rounding and float noise)
lenient:   strict, OR the prediction matches after x100 or /100 (percent vs fraction, e.g. 15 vs 0.15)
sign_flip: wrong only because of the sign (tracked for error analysis, NOT counted as correct)
"""

from __future__ import annotations

REL_TOL = 0.01
ZERO_ABS_TOL = 1e-3  # relative tolerance is meaningless when the label is 0


def is_close(pred: float, gold: float) -> bool:
    if gold == 0:
        return abs(pred) <= ZERO_ABS_TOL
    return abs(pred - gold) <= REL_TOL * abs(gold)


def _close_any_scale(pred: float, gold: float) -> bool:
    return is_close(pred, gold) or is_close(pred * 100, gold) or is_close(pred / 100, gold)


def score(pred: float | None, gold: float) -> dict[str, bool]:
    if pred is None:
        return {"strict": False, "lenient": False, "sign_flip": False}
    strict = is_close(pred, gold)
    lenient = _close_any_scale(pred, gold)
    sign_flip = not lenient and gold != 0 and _close_any_scale(-pred, gold)
    return {"strict": strict, "lenient": lenient, "sign_flip": sign_flip}