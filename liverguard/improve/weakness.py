"""Turn baseline robustness-audit tables into a list of diagnosed weaknesses and a targeted experiment plan.

Rules (all thresholds are arguments and are written into the report; nothing is tuned on the test split because the audit
being read must be a VALIDATION audit):
- A condition is a WEAKNESS in one audit when, at some PLAUSIBLE-tier level whose fidelity heuristics do not flag it,
  (a) the AUROC drop is >= min_drop and its patient-bootstrap CI excludes 0, or (b) the lower CI of the flip rate >= flip_alert.
  Without CIs (n_boot=0) point estimates are used and the evidence is marked 'no_ci'.
- With several baseline seeds a condition is a weakness only if it is flagged in a majority of the audits (seed noise guard).
- Stress-tier failures never create a weakness: they may be label-destroying and are not clinical findings.
- Conditions flagged by the fidelity proxies at every plausible level are 'excluded' (we would be training against possibly
  non-label-preserving changes).
LIMITATION: the diagnosis inherits the audit's assumptions (a-priori tiers, simplified speckle, unverified label preservation).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from liverguard.improve.interventions import INTERVENTIONS, intervention_for


def _flags(summary: pd.DataFrame, min_drop: float, flip_alert: float) -> dict[str, dict]:
    out: dict[str, dict] = {}
    pl = summary[summary["tier"] == "plausible"]
    for cond, g in pl.groupby("condition", sort=False):
        usable = g[~g["fidelity_flag"].astype(bool)]
        ev, has_ci = [], bool(g["d_auroc_lo"].notna().all())
        for _, r in usable.iterrows():
            drop = -float(r["d_auroc"]) if pd.notna(r["d_auroc"]) else 0.0
            ci_ok = (float(r["d_auroc_hi"]) < 0) if has_ci else True
            a = drop >= min_drop and ci_ok
            lo = float(r["pfr_lo"]) if has_ci and pd.notna(r["pfr_lo"]) else float(r["pfr"])
            b = lo >= flip_alert
            if a or b:
                ev.append({"level": int(r["level"]), "auroc_drop": round(drop, 4), "pfr": round(float(r["pfr"]), 4),
                           "by_auroc": bool(a), "by_flip": bool(b)})
        out[cond] = {"flagged": bool(ev), "evidence": ev, "no_ci": not has_ci, "excluded": len(usable) == 0,
                     "mean_d_auroc_plausible": float(g["d_auroc"].mean()), "mean_pfr_plausible": float(g["pfr"].mean())}
    return out


def diagnose(audit_dirs: list[str | Path], *, min_drop: float = 0.02, flip_alert: float = 0.05, max_targeted: int = 4) -> dict:
    if not audit_dirs:
        raise ValueError("need at least one baseline audit directory")
    per_audit, meta = [], []
    for d in map(Path, audit_dirs):
        man = json.loads((d / "audit_manifest.json").read_text())
        if man.get("split") != "val":
            raise ValueError(f"{d}: weaknesses must be diagnosed on the VALIDATION split, got {man.get('split')!r}")
        per_audit.append(_flags(pd.read_csv(d / "summary_by_level.csv"), min_drop, flip_alert))
        meta.append({"dir": str(d), "valid_as_result": man.get("valid_as_result"), "watermark": man.get("watermark"),
                     "n_images": man["dataset"]["n_images"], "n_patients": man["dataset"]["n_patients"]})
    conds = list(per_audit[0])
    for m, a in zip(meta, per_audit):
        if set(a) != set(conds):
            raise ValueError(f"{m['dir']} covers different conditions than {meta[0]['dir']}; audit all baselines with the same config")
    need = math.ceil(len(per_audit) / 2 + 1e-9) if len(per_audit) > 1 else 1
    rows = []
    for c in conds:
        n_flag = sum(a[c]["flagged"] for a in per_audit)
        rows.append({"condition": c, "n_audits_flagged": n_flag, "n_audits": len(per_audit),
                     "weakness": n_flag >= need, "excluded_fidelity": all(a[c]["excluded"] for a in per_audit),
                     "mean_d_auroc_plausible": float(np.mean([a[c]["mean_d_auroc_plausible"] for a in per_audit])),
                     "mean_pfr_plausible": float(np.mean([a[c]["mean_pfr_plausible"] for a in per_audit])),
                     "no_ci": any(a[c]["no_ci"] for a in per_audit),
                     "evidence": [a[c]["evidence"] for a in per_audit]})
    # severity: bigger AUROC loss and flip rate first
    weak = sorted([r for r in rows if r["weakness"]], key=lambda r: (r["mean_d_auroc_plausible"], -r["mean_pfr_plausible"]))
    targeted: list[str] = []
    for r in weak:
        iv = intervention_for(r["condition"])
        if iv and iv not in targeted:
            targeted.append(iv)
    targeted = targeted[:max_targeted]
    weak_names = {r["condition"] for r in weak}
    # negative control: a single-family intervention aimed only at conditions that were NOT diagnosed as weaknesses
    control = next((n for n, cs in INTERVENTIONS.items()
                    if n not in ("baseline", "combined", "photometric") and cs and not (set(cs) & weak_names)
                    and n not in targeted), None)
    plan = ["baseline"] + targeted + (["combined"] if targeted else []) + ([control] if control else [])
    return {"thresholds": {"min_drop": min_drop, "flip_alert": flip_alert, "majority_needed": need, "max_targeted": max_targeted},
            "audits": meta, "valid_as_result": all(bool(m["valid_as_result"]) for m in meta),
            "weaknesses": [r["condition"] for r in weak], "conditions": rows, "targeted": targeted,
            "negative_control": control, "plan": plan,
            "note": ("No condition met the weakness criteria at a plausible severity: nothing to fix with augmentation. "
                     "Only 'baseline' (and 'combined' as a sanity ablation) will be run." if not weak else "")}


def write_report(diag: dict, out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(diag, indent=2, default=float), encoding="utf-8")
    L = ["# Weakness diagnosis and experiment plan", ""]
    if not diag["valid_as_result"]:
        L += ["> **NOT A RESULT**: the audited baseline is a smoke-test / control model. The 'weaknesses' below describe a toy model "
              "and only demonstrate the pipeline.", ""]
    t = diag["thresholds"]
    L += [f"Rule: plausible-tier level with AUROC drop >= {t['min_drop']} (CI excludes 0) or flip-rate lower CI >= {t['flip_alert']}; "
          f"flagged in >= {t['majority_needed']} of {len(diag['audits'])} baseline audit(s); fidelity-flagged levels ignored.", "",
          "| condition | audits flagged | mean dAUROC (plausible) | mean flip rate (plausible) | verdict |", "|---|---|---|---|---|"]
    for r in sorted(diag["conditions"], key=lambda r: r["mean_d_auroc_plausible"]):
        v = "WEAKNESS" if r["weakness"] else ("excluded (fidelity)" if r["excluded_fidelity"] else "ok")
        L.append(f"| {r['condition']} | {r['n_audits_flagged']}/{r['n_audits']} | {r['mean_d_auroc_plausible']:+.3f} | "
                 f"{r['mean_pfr_plausible']:.3f} | {v}{' (no CI)' if r['no_ci'] else ''} |")
    L += ["", f"**Planned interventions:** {', '.join(diag['plan'])}",
          f"(targeted: {diag['targeted'] or 'none'}; negative control: {diag['negative_control'] or 'none'})", ""]
    if diag["note"]:
        L += [diag["note"], ""]
    L += ["Caveat: tiers are a-priori, label preservation is unverified, and training on a family then auditing it measures "
          "adaptation to a known corruption; transfer is judged on conditions outside the trained family."]
    p = out / "weakness_report.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
