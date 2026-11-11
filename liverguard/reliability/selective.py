"""Risk-coverage analysis / selective prediction (abstain on the most uncertain cases).

LIMITATIONS: the risk-coverage curve is measured on one finite split, so it is noisy at low coverage (few cases); abstention
lowers the error of the retained cases but does not make them clinically safe; thresholds for abstention must be chosen on
validation data, never on the split that is reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_coverage(errors, uncertainty) -> pd.DataFrame:
    """Keep the k most certain cases (ascending uncertainty; ties broken by index for determinism).

    coverage_k = k/n; risk_k = error rate among the kept k. Returns one row per k = 1..n.
    """
    err = np.asarray(errors, dtype=float)
    u = np.asarray(uncertainty, dtype=float)
    order = np.lexsort((np.arange(len(u)), u))
    cum = np.cumsum(err[order])
    k = np.arange(1, len(err) + 1)
    return pd.DataFrame({"coverage": k / len(err), "risk": cum / k, "n_kept": k, "uncertainty_cutoff": u[order]})


def aurc(rc: pd.DataFrame) -> float:
    """Area under the risk-coverage curve (lower is better) = mean of the selective risks over all k."""
    return float(rc["risk"].mean())


def oracle_aurc(errors) -> float:
    """AURC of a perfect ranker (all correct cases kept first)."""
    return aurc(risk_coverage(errors, np.asarray(errors, dtype=float)))


def random_aurc(errors) -> float:
    """Expected AURC of an uninformative ranker = overall error rate."""
    return float(np.mean(errors))


def risk_at_coverage(rc: pd.DataFrame, coverage: float) -> float:
    """Selective risk when keeping (approximately) the given fraction of cases."""
    k = max(1, int(round(coverage * len(rc))))
    return float(rc["risk"].iloc[k - 1])


def coverage_summary(errors, scores: dict[str, np.ndarray], coverages=(1.0, 0.9, 0.8, 0.7, 0.5)) -> pd.DataFrame:
    """One row per uncertainty score: AURC, normalised excess vs oracle, and risk at the given coverages."""
    errors = np.asarray(errors, dtype=float)
    base, orc = random_aurc(errors), oracle_aurc(errors)
    rows = []
    for name, u in scores.items():
        rc = risk_coverage(errors, u)
        a = aurc(rc)
        row = {"score": name, "aurc": a, "aurc_oracle": orc, "aurc_random": base,
               "excess_aurc_frac": (a - orc) / (base - orc) if base > orc else float("nan")}
        row.update({f"risk@{int(c * 100)}%": risk_at_coverage(rc, c) for c in coverages})
        rows.append(row)
    return pd.DataFrame(rows)
