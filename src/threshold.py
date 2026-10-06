"""Cost-based decision threshold. A missed defaulter (FN) costs more than a refused good customer (FP)."""
import numpy as np


def expected_cost(y, p, threshold, cost_fn=5.0, cost_fp=1.0):
    y, p = np.asarray(y), np.asarray(p)
    decline = p >= threshold
    fn = int(((~decline) & (y == 1)).sum())      # approved, then defaulted
    fp = int((decline & (y == 0)).sum())         # declined a good customer
    return cost_fn * fn + cost_fp * fp


def theoretical_threshold(cost_fn=5.0, cost_fp=1.0):
    """With HONEST (calibrated) probabilities, the best cutoff is cost_fp / (cost_fp + cost_fn)."""
    return cost_fp / (cost_fp + cost_fn)


def best_threshold(y, p, cost_fn=5.0, cost_fp=1.0, grid=None):
    grid = np.linspace(0.01, 0.99, 99) if grid is None else grid
    costs = np.array([expected_cost(y, p, t, cost_fn, cost_fp) for t in grid])
    i = int(costs.argmin())
    return float(grid[i]), float(costs[i])


def threshold_summary(y, p, cost_fn=5.0, cost_fp=1.0):
    """One row of facts for a given cost ratio."""
    y, p = np.asarray(y), np.asarray(p)
    thr, cost = best_threshold(y, p, cost_fn, cost_fp)
    decline = p >= thr
    return {
        "cost_ratio": cost_fn / cost_fp,
        "theory_threshold": round(theoretical_threshold(cost_fn, cost_fp), 3),
        "best_threshold": round(thr, 3),
        "pct_declined": round(100 * decline.mean(), 1),
        "defaulters_caught_pct": round(100 * (decline & (y == 1)).sum() / max((y == 1).sum(), 1), 1),
        "precision_pct": round(100 * (decline & (y == 1)).sum() / max(decline.sum(), 1), 1),
        "cost": cost,
        "cost_approve_all": expected_cost(y, p, 1.01, cost_fn, cost_fp),
        "cost_at_0.5": expected_cost(y, p, 0.5, cost_fn, cost_fp),
    }


def decision(p, decline_threshold):
    """approve / review / decline. 'Review' = grey zone between half the cutoff and the cutoff."""
    if p >= decline_threshold:
        return "decline"
    if p >= 0.5 * decline_threshold:
        return "review"
    return "approve"
def risk_band(p, decline_threshold):
    """Return a simple risk band consistent with the decision threshold."""
    if p >= decline_threshold:
        return "High"
    if p >= 0.5 * decline_threshold:
        return "Medium"
    return "Low"