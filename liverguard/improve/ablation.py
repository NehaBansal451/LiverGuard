"""M6 orchestration: diagnose -> train variants -> validate (ablation + pre-declared selection) -> final test re-audit.

Protocol (enforced in code, not just documented)
1. `diagnose`: train the baseline(s) WITHOUT evaluating test, audit them on the VALIDATION split, derive weaknesses and a plan.
2. `train_variants`: train every planned intervention with the same seeds / split / architecture / epochs; only the augmentation
   differs. Test is never evaluated at training time.
3. `validate`: audit all models on VALIDATION, compare to the baseline with paired patient-bootstrap CIs, apply the pre-declared
   selection rule and write `selection.json` (with checkpoint hashes) BEFORE any test evaluation.
4. `final`: needs `final_report=True` and an intact `selection.json`; audits ALL models once on the TEST split and compares them
   with exactly the same code. The selection is not revisited; the other variants are reported descriptively.
Every stage is resumable (existing outputs are reused) and writes a stage manifest.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from liverguard.data.report import json_default
from liverguard.improve import compare as C
from liverguard.improve import plots as P
from liverguard.improve import weakness as W
from liverguard.improve.interventions import conditions_of
from liverguard.ml.config import TrainConfig
from liverguard.ml.trainer import run_experiment
from liverguard.robustness.config import RobustnessConfig, load_robustness_config
from liverguard.robustness.engine import run_robustness_audit
from liverguard.utils import file_sha256, get_logger

log = get_logger(__name__)


def model_name(intervention: str, seed: int) -> str:
    return f"{intervention}_s{seed}"


class Workspace:
    def __init__(self, workdir: str | Path, base_cfg: TrainConfig, index_csv: str | Path, seeds: list[int], *,
                 rob_cfg: RobustnessConfig | None = None, split_file=None, data_root=None, n_boot: int = 500,
                 max_clean_drop: float = 0.02, audit_batch_size: int = 64) -> None:
        if not seeds:
            raise ValueError("need at least one seed")
        self.dir, self.index_csv, self.seeds = Path(workdir), Path(index_csv), list(seeds)
        self.base, self.split_file, self.data_root = base_cfg, split_file, data_root
        self.rob = rob_cfg or load_robustness_config(base_cfg.robustness_config)
        self.n_boot, self.max_clean_drop, self.bs = n_boot, max_clean_drop, audit_batch_size
        self.dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def model_dir(self, name: str) -> Path:
        return self.dir / "models" / name

    def audit_dir(self, split: str, name: str) -> Path:
        return self.dir / "audits" / split / name

    @property
    def watermark_note(self) -> str | None:
        return self.base.watermark

    # ------------------------------------------------------------------ training
    def train(self, intervention: str, seed: int) -> Path:
        conditions_of(intervention)
        name = model_name(intervention, seed)
        out = self.model_dir(name)
        if (out / "best.pt").exists() and (out / "manifest.json").exists():
            log.info("reuse trained model %s", name)
            return out
        kind = "baseline" if intervention == "baseline" else "intervention"
        if self.base.experiment_kind == "smoke":
            kind = "smoke"
        cfg = self.base.model_copy(update={"intervention": intervention, "seed": seed, "run_name": name,
                                           "experiment_kind": kind, "evaluate_test": False})
        TrainConfig.model_validate(cfg.model_dump())          # re-validate the modified copy
        run_experiment(cfg, self.index_csv, out, split_file=self.split_file, data_root=self.data_root)
        return out

    # ------------------------------------------------------------------ audit
    def audit(self, name: str, split: str, final_report: bool = False) -> Path:
        out = self.audit_dir(split, name)
        if (out / "audit_manifest.json").exists():
            return out
        run_robustness_audit(self.model_dir(name) / "best.pt", self.index_csv, out, cfg=self.rob, split=split,
                             threshold="val_youden", batch_size=self.bs, final_report=final_report,
                             split_file=self.split_file, data_root=self.data_root, review_n=1, make_plots=False)
        return out

    def load_eval(self, intervention: str, seed: int, split: str) -> C.ModelEval:
        name = model_name(intervention, seed)
        return C.ModelEval.from_audit(name, intervention, seed, conditions_of(intervention), self.audit_dir(split, name))

    # ------------------------------------------------------------------ stages
    def diagnose(self, min_drop: float = 0.02, flip_alert: float = 0.05, max_targeted: int = 4) -> dict:
        dirs = []
        for s in self.seeds:
            self.train("baseline", s)
            dirs.append(self.audit(model_name("baseline", s), "val"))
        diag = W.diagnose(dirs, min_drop=min_drop, flip_alert=flip_alert, max_targeted=max_targeted)
        W.write_report(diag, self.dir / "diagnosis")
        return diag

    def plan(self) -> list[str]:
        p = self.dir / "diagnosis" / "plan.json"
        if not p.exists():
            raise FileNotFoundError("run the diagnose stage first (diagnosis/plan.json is missing)")
        return json.loads(p.read_text())["plan"]

    def train_variants(self, interventions: list[str] | None = None) -> list[str]:
        plan = interventions or self.plan()
        for iv in plan:
            for s in self.seeds:
                self.train(iv, s)
        return plan

    def _compare(self, split: str, plan: list[str], out: Path, title: str) -> dict:
        for iv in plan:
            for s in self.seeds:
                self.audit(model_name(iv, s), split, final_report=(split == "test"))
        models = [self.load_eval(iv, s, split) for iv in plan for s in self.seeds]
        out.mkdir(parents=True, exist_ok=True)
        points = C.point_table(models)
        per_cond = pd.concat([C.per_condition_auroc(m) for m in models], ignore_index=True)
        per_run, summary = C.compare_to_baseline(models, n_boot=self.n_boot, seed=self.rob.seed)
        for name, df in (("point_metrics", points), ("per_condition_auroc", per_cond), ("paired_diffs_per_run", per_run),
                         ("paired_diffs_summary", summary)):
            df.to_csv(out / f"{name}.csv", index=False)
        wm = self.watermark_note
        figs = {"tradeoff": P.plot_tradeoff(summary, out / "figures" / "tradeoff_clean_vs_robust.png", title, wm),
                "forest": P.plot_forest(summary, out / "figures" / "paired_differences.png", title, wm),
                "ablation_matrix": P.plot_ablation_matrix(per_cond, out / "figures" / "ablation_matrix.png", title, wm),
                "degradation": P.plot_degradation(per_cond, out / "figures" / "degradation_baseline_vs_best.png", title, wm,
                                                  highlight=None)}
        return {"models": models, "points": points, "per_cond": per_cond, "per_run": per_run, "summary": summary, "figures": figs}

    def validate(self, interventions: list[str] | None = None) -> dict:
        plan = interventions or self.plan()
        out = self.dir / "validation"
        res = self._compare("val", plan, out, f"validation split · {len(self.seeds)} seed(s)")
        nc = {iv: len(conditions_of(iv)) for iv in plan}
        sel = C.select_final(res["summary"], nc, max_clean_drop=self.max_clean_drop)
        try:
            sel["negative_control"] = json.loads((self.dir / "diagnosis" / "plan.json").read_text()).get("negative_control")
        except FileNotFoundError:
            sel["negative_control"] = None
        sel.update({"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "plan": plan, "seeds": self.seeds,
                    "selected_on": "validation split only", "n_boot": self.n_boot,
                    "checkpoint_sha256": {model_name(iv, s): file_sha256(self.model_dir(model_name(iv, s)) / "best.pt")
                                          for iv in plan for s in self.seeds},
                    "valid_as_result": self.base.valid_as_result, "watermark": self.watermark_note})
        (out / "selection.json").write_text(json.dumps(sel, indent=2, default=json_default), encoding="utf-8")
        # degradation figure highlighting the chosen model
        P.plot_degradation(res["per_cond"], out / "figures" / "degradation_baseline_vs_best.png",
                           "validation split", self.watermark_note, highlight=sel["chosen"], selected=sel["chosen"])
        write_report(out / "report.md", "VALIDATION (ablation and selection)", res, sel, self.base, self.rob, self.seeds)
        res["selection"] = sel
        return res

    def final(self, *, final_report: bool = False) -> dict:
        if not final_report:
            raise ValueError("The TEST re-audit requires final_report=True (--final-report). Run it once, after selection.")
        sp = self.dir / "validation" / "selection.json"
        if not sp.exists():
            raise FileNotFoundError("selection.json is missing: run the validate stage first. The test split is never evaluated "
                                    "before the selection is frozen.")
        sel = json.loads(sp.read_text())
        for name, h in sel["checkpoint_sha256"].items():
            if file_sha256(self.model_dir(name) / "best.pt") != h:
                raise ValueError(f"checkpoint {name} changed after selection was frozen; refusing the test evaluation")
        out = self.dir / "final_test"
        res = self._compare("test", sel["plan"], out, f"TEST split (untouched until now) · {len(self.seeds)} seed(s)")
        P.plot_degradation(res["per_cond"], out / "figures" / "degradation_baseline_vs_best.png",
                           "TEST split", self.watermark_note, highlight=sel["chosen"], selected=sel["chosen"])
        write_report(out / "report.md", "FINAL TEST RE-AUDIT (selection frozen on validation)", res, sel, self.base, self.rob,
                     self.seeds, final=True)
        (out / "final_manifest.json").write_text(json.dumps(
            {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "selection": sel["chosen"],
             "selection_file_sha256": file_sha256(sp), "valid_as_result": self.base.valid_as_result,
             "watermark": self.watermark_note, "n_boot": self.n_boot}, indent=2), encoding="utf-8")
        res["selection"] = sel
        return res


# ------------------------------------------------------------------ report
def _fmt(d: float, lo: float, hi: float, digits: int = 3) -> str:
    ci = "" if not np.isfinite(lo) else f" [{lo:+.{digits}f}, {hi:+.{digits}f}]"
    return f"{d:+.{digits}f}{ci}"


def write_report(path: Path, heading: str, res: dict, sel: dict, base: TrainConfig, rob: RobustnessConfig, seeds: list[int],
                 final: bool = False) -> None:
    pts, summ = res["points"], res["summary"]
    L = [f"# {heading}", ""]
    if base.watermark:
        L += [f"> **{base.watermark}** - these numbers validate the pipeline; they say nothing about liver ultrasound.", ""]
    L += [f"Models: {', '.join(sorted(pts['intervention'].unique()))} x seeds {seeds}. Images: {int(pts['n_images'].iloc[0])} / "
          f"{int(pts['n_patients'].iloc[0])} patients. Distortion tiers: levels <= {rob.plausible_max_level} plausible, above = stress "
          "(a-priori, not expert-validated). Differences are (model - baseline of the same seed) with paired patient-bootstrap 95% CIs; "
          "the range over seeds is shown where several seeds exist.", "",
          "## Absolute metrics (mean over seeds)", ""]
    cols = ["clean_auroc", "clean_accuracy", "clean_ece", "clean_brier", "plaus_auroc", "plaus_minus_clean_auroc", "plaus_worst_auroc", "plaus_flip_rate",
            "plaus_mean_abs_dp", "plaus_ece", "plaus_brier", "stress_auroc"]
    m = pts.groupby("intervention", sort=False)[cols].mean().round(3)
    L += ["| intervention | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for iv, r in m.iterrows():
        L.append(f"| {iv} | " + " | ".join(f"{r[c]:.3f}" for c in cols) + " |")
    L += ["", "## Paired differences vs baseline (positive = better after sign-correction in the verdict)", "",
          "| intervention | metric | mean diff | seed range | worst-seed CI | verdict |", "|---|---|---|---|---|---|"]
    for iv, g in summ.groupby("intervention", sort=False):
        for _, r in g[g["metric"].isin(C.PRIMARY)].iterrows():
            L.append(f"| {iv} | {r['metric']} | {r['mean_diff']:+.3f} | [{r['min_diff']:+.3f}, {r['max_diff']:+.3f}] | "
                     f"[{r['ci_lo_min_seed']:+.3f}, {r['ci_hi_max_seed']:+.3f}] | {r['verdict']} |")
    tu = summ[summ["metric"].isin(["plaus_auroc_trained", "plaus_auroc_untrained"])]
    if len(tu):
        L += ["", "## Adaptation vs transfer (plausible-tier AUROC difference)", "",
              "'trained' = conditions of the intervention's own family (adaptation to a known corruption); 'untrained' = all others (transfer).", "",
              "| intervention | trained conditions | untrained conditions |", "|---|---|---|"]
        for iv, g in tu.groupby("intervention", sort=False):
            s = g.set_index("metric")
            L.append(f"| {iv} | {s.loc['plaus_auroc_trained','mean_diff']:+.3f} ({s.loc['plaus_auroc_trained','verdict']}) | "
                     f"{s.loc['plaus_auroc_untrained','mean_diff']:+.3f} ({s.loc['plaus_auroc_untrained','verdict']}) |")
    L += ["", "## Selection", ""]
    if final:
        L += [f"Selection frozen on validation before this evaluation: **{sel['chosen']}**. All other models are reported "
              "descriptively and cannot change the selection.", ""]
    else:
        L += [f"Pre-declared rule: {json.dumps(sel['rule'])}", "", f"**Chosen: {sel['chosen']}**"
              + (" (no intervention qualified: robust training is not adopted)" if sel["chosen"] == "baseline" else ""), "",
              "| intervention | clean AUROC diff | clean ok | plausible AUROC diff | robustness improved |", "|---|---|---|---|---|"]
        for c in sel["candidates"]:
            L.append(f"| {c['intervention']} | {c['clean_auroc_diff']:+.3f} | {c['clean_ok']} | {c['plaus_auroc_diff']:+.3f} | {c['robust_improved']} |")
    nc = sel.get("negative_control")
    if nc and nc in set(summ["intervention"]):
        s = summ[summ["intervention"] == nc]
        imp = s[s["verdict"] == "improved"]["metric"].tolist()
        L += ["", f"### Negative control: {nc}",
              f"{nc} trains only on conditions that were NOT diagnosed as weaknesses, so it should not help. "
              + (f"It was nevertheless 'improved' on: {', '.join(imp)}. **Verdicts of that size are within the false-positive range of this "
                 "comparison; discount equal-sized 'improvements' elsewhere.**" if imp else "It showed no 'improved' verdict, as expected.")]
    ch = sel["chosen"]
    if ch != "baseline":
        sub = summ[summ["intervention"] == ch]
        worse = sub[sub["verdict"] == "worse"]["metric"].tolist()
        better = sub[sub["verdict"] == "improved"]["metric"].tolist()
        L += ["", f"### Trade-offs of {ch}", f"- improved (all seeds, CI excludes 0): {', '.join(better) or 'none'}",
              f"- worse (all seeds, CI excludes 0): {', '.join(worse) or 'none'}",
              "- everything else is inconclusive: the data cannot distinguish it from no change."]
        gap = sub[sub["metric"] == "plaus_minus_clean_auroc"]
        if len(gap) and "plaus_auroc" in better and gap["verdict"].iloc[0] != "improved":
            L += ["", f"> **Attribution caveat:** {ch}'s higher plausible-tier AUROC is NOT shown to be robustness. The degradation gap "
                  f"(plausible minus clean AUROC) changed by {gap['mean_diff'].iloc[0]:+.3f} (seed range [{gap['min_diff'].iloc[0]:+.3f}, "
                  f"{gap['max_diff'].iloc[0]:+.3f}], verdict {gap['verdict'].iloc[0]}): the gain tracks a change in clean AUROC "
                  f"({sub[sub['metric'] == 'clean_auroc']['mean_diff'].iloc[0]:+.3f}), i.e. a better model overall rather than less "
                  "degradation under distortion."]
    L += ["", "## Limitations", "",
          "- Tiers are a-priori; label preservation under distortion is unverified; speckle is a simplified model.",
          "- Training on a distortion family and auditing the same family measures adaptation to a known corruption; read the transfer column.",
          f"- {len(seeds)} training seed(s): " + ("seed-to-seed variance is NOT measured, treat every verdict as provisional." if len(seeds) < 2
                                                  else "verdicts require agreement across all seeds."),
          "- ~30 correlated metrics are compared; 'improved' at 95% is not multiplicity-corrected.",
          "- Calibration under distortion uses top-label ECE (10 equal-width bins), noisy for small n.",
          "- ROI-aware training and test-time augmentation are not implemented (no annotation masks; TTA was a reliability probe only)."]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
