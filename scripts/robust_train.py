#!/usr/bin/env python
"""M6: diagnose weaknesses -> targeted robust training -> ablation on VALIDATION -> frozen selection -> one TEST re-audit.

  python scripts/robust_train.py diagnose  --workdir artifacts/m6 --config configs/baseline_efficientnet_b0.yaml \\
         --index-csv artifacts/dataset/image_index.csv --seeds 42,43,44
  python scripts/robust_train.py train     (same options; trains the planned interventions)
  python scripts/robust_train.py validate  (audits all models on val, paired comparison, writes validation/selection.json)
  python scripts/robust_train.py final --final-report   (test split, ONLY after validate; selection cannot change)
  python scripts/robust_train.py all       (diagnose + train + validate)
Exit codes: 0 ok, 2 could not run / protocol violation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from liverguard.improve.ablation import Workspace  # noqa: E402
from liverguard.ml.config import load_train_config  # noqa: E402
from liverguard.ml.data import LeakageError  # noqa: E402
from liverguard.robustness.config import load_robustness_config  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["diagnose", "train", "validate", "final", "all"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--config", required=True, help="baseline training YAML; every model uses it, only the intervention differs")
    ap.add_argument("--index-csv", required=True)
    ap.add_argument("--seeds", default="42", help="comma-separated training seeds (use >= 3 for a defensible claim)")
    ap.add_argument("--rob-config", help="robustness YAML (default configs/robustness.yaml)")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--split-file")
    ap.add_argument("--data-root")
    ap.add_argument("--interventions", help="override the plan, comma-separated (must include baseline)")
    ap.add_argument("--min-drop", type=float, default=0.02)
    ap.add_argument("--flip-alert", type=float, default=0.05)
    ap.add_argument("--max-clean-drop", type=float, default=0.02)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device")
    ap.add_argument("--pretrained-weights-path")
    ap.add_argument("--final-report", action="store_true", help="required for the final TEST re-audit")
    a = ap.parse_args(argv)
    try:
        cfg = load_train_config(a.config, epochs=a.epochs, device=a.device, pretrained_weights_path=a.pretrained_weights_path)
        rob = load_robustness_config(a.rob_config, n_boot=a.n_boot)
        ws = Workspace(a.workdir, cfg, a.index_csv, [int(s) for s in a.seeds.split(",")], rob_cfg=rob, split_file=a.split_file,
                       data_root=a.data_root, n_boot=a.n_boot, max_clean_drop=a.max_clean_drop)
        plan = [s.strip() for s in a.interventions.split(",")] if a.interventions else None
        if a.stage in ("diagnose", "all"):
            d = ws.diagnose(a.min_drop, a.flip_alert)
            print("weaknesses:", d["weaknesses"] or "none", "| plan:", d["plan"], "| report:", Path(a.workdir) / "diagnosis" / "weakness_report.md")
        if a.stage in ("train", "all"):
            print("trained:", ws.train_variants(plan))
        if a.stage in ("validate", "all"):
            r = ws.validate(plan)
            print("selected on validation:", r["selection"]["chosen"], "| report:", Path(a.workdir) / "validation" / "report.md")
        if a.stage == "final":
            r = ws.final(final_report=a.final_report)
            print("test re-audit done; selection was", r["selection"]["chosen"], "| report:", Path(a.workdir) / "final_test" / "report.md")
        if cfg.watermark:
            print(f"*** {cfg.watermark} ***")
    except (FileNotFoundError, ValueError, LeakageError, RuntimeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
