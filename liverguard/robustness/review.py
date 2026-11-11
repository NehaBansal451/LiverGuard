"""Expert label-preservation review: a template for clinicians and a summariser for their answers.

Whether a distorted image still supports the original diagnosis can only be judged by an expert. The fidelity heuristics are a
screening proxy. This module writes the sheet an expert fills in and turns completed sheets into the share of images judged
"label preserved" per (condition, level), so stress-test levels can be justified (or dropped) with evidence.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from liverguard.robustness.config import RobustnessConfig

REVIEW_COLUMNS = ["image_id", "condition", "level", "tier", "fidelity_at_risk", "reviewer_label_preserved", "reviewer_notes"]
_YES, _NO, _UNSURE = {"yes", "y", "1", "true"}, {"no", "n", "0", "false"}, {"unsure", "?", "u"}


def review_template(image_ids: list[str], cfg: RobustnessConfig, at_risk: dict[tuple[str, str, int], bool]) -> pd.DataFrame:
    rows = [{"image_id": i, "condition": c.name, "level": k, "tier": cfg.tier(k),
             "fidelity_at_risk": bool(at_risk.get((i, c.name, k), False)),
             "reviewer_label_preserved": "", "reviewer_notes": ""}
            for i in image_ids for c in cfg.conditions for k in range(1, len(c.params) + 1)]
    return pd.DataFrame(rows, columns=REVIEW_COLUMNS)


def summarize_review(path: str | Path) -> pd.DataFrame:
    """Per (condition, level): number reviewed and share judged label-preserved (yes / [yes+no]); unsure counted separately.

    Rows with an empty or unrecognised answer are ignored (and counted in `n_unanswered`).
    """
    df = pd.read_csv(path, dtype=str).fillna("")
    ans = df["reviewer_label_preserved"].str.strip().str.lower()
    df["_a"] = ans.map(lambda a: "yes" if a in _YES else "no" if a in _NO else "unsure" if a in _UNSURE else "")
    out = []
    for (cond, level), g in df.groupby(["condition", "level"], sort=False):
        yes, no, uns = (g["_a"] == "yes").sum(), (g["_a"] == "no").sum(), (g["_a"] == "unsure").sum()
        out.append({"condition": cond, "level": int(level), "n_yes": int(yes), "n_no": int(no), "n_unsure": int(uns),
                    "n_unanswered": int((g["_a"] == "").sum()),
                    "share_preserved": float(yes / (yes + no)) if (yes + no) else float("nan")})
    return pd.DataFrame(out)
