"""Registry: scan a workspace for run manifests, store them, query them. Idempotent; bad manifests are reported, never fatal."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session as DBSession

from liverguard.db.models import Base, Job, ReportRecord, Run
from liverguard.utils import get_logger

log = get_logger(__name__)

MANIFEST_NAMES = {"manifest.json": "train", "audit_manifest.json": "audit", "reliability_manifest.json": "reliability",
                  "explain_manifest.json": "explain", "final_manifest.json": "m6_final", "selection.json": "m6_validation"}


def _headline(kind: str, d: dict[str, Any]) -> dict[str, Any]:
    """Copy headline numbers out of a manifest (no computation)."""
    if kind == "train":
        r = d.get("results", {})
        return {"best_epoch": d.get("training", {}).get("best_epoch"), "val": r.get("val"), "test": r.get("test"),
                "val_youden": r.get("val_at_youden"), "test_at_val_youden": r.get("test_at_val_youden")}
    if kind == "audit":
        return {"clean": d.get("clean"), "worst_case": d.get("worst_case"), "top": d.get("top_disruptive_conditions", [])[:3],
                "threshold": d.get("threshold"), "n_images": d.get("dataset", {}).get("n_images")}
    if kind == "reliability":
        return {"calibration": d.get("calibration"), "selective": d.get("selective"), "outcomes": d.get("outcomes"),
                "n_images": d.get("dataset", {}).get("n_images")}
    if kind == "explain":
        return {"masking": d.get("masking"), "cam_faithfulness": d.get("cam_faithfulness"),
                "randomization": d.get("gradcam_randomization"), "roi_source": d.get("roi_source")}
    return {k: d.get(k) for k in ("chosen", "rule", "candidates", "negative_control") if k in d}


class Registry:
    def __init__(self, db_path: str | Path) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", future=True)
        Base.metadata.create_all(self.engine)

    # ---------------------------------------------------------------- runs
    def scan(self, root: str | Path) -> dict[str, Any]:
        """Index every known manifest under `root`. Returns {'added', 'updated', 'errors': [(path, message)]}."""
        added = updated = 0
        errors: list[tuple[str, str]] = []
        root = Path(root)
        if not root.exists():
            return {"added": 0, "updated": 0, "removed": 0, "errors": [(str(root), "workspace directory does not exist")]}
        files = [p for p in root.rglob("*.json") if p.name in MANIFEST_NAMES]
        with DBSession(self.engine) as s:
            for p in sorted(files):
                try:
                    d = json.loads(p.read_text())
                    kind = MANIFEST_NAMES[p.name]
                    if kind == "train" and "training" not in d:          # a different manifest.json
                        continue
                    if kind == "m6_validation" and "chosen" not in d:
                        continue
                    row = s.scalar(select(Run).where(Run.manifest_path == str(p)))
                    new = row is None
                    row = row or Run(manifest_path=str(p))
                    self._fill(row, kind, d, p)
                    s.add(row)
                    added, updated = added + new, updated + (not new)
                except Exception as exc:                                  # noqa: BLE001 - report, keep scanning
                    errors.append((str(p), f"{type(exc).__name__}: {exc}"))
                    log.warning("could not index %s: %s", p, exc)
            # drop rows whose manifest vanished from this workspace (deleted/moved runs must not linger as ghosts)
            seen = {str(p) for p in files}
            removed = 0
            for row in list(s.scalars(select(Run))):
                if row.manifest_path not in seen and str(row.manifest_path).startswith(str(root)) and not Path(row.manifest_path).exists():
                    s.delete(row)
                    removed += 1
            s.commit()
        return {"added": added, "updated": updated, "removed": removed, "errors": errors}

    @staticmethod
    def _fill(row: Run, kind: str, d: dict[str, Any], p: Path) -> None:
        ck = d.get("checkpoint", {}) if isinstance(d.get("checkpoint"), dict) else {}
        ds = d.get("dataset", {}) if isinstance(d.get("dataset"), dict) else {}
        model = d.get("model", {}) if isinstance(d.get("model"), dict) else {}
        row.kind, row.out_dir = kind, str(p.parent)
        row.name = d.get("run_name") or p.parent.name
        row.created_utc = d.get("created_utc", "")
        row.valid_as_result = bool(d.get("valid_as_result", kind in ("m6_validation", "m6_final") and False))
        row.watermark = d.get("watermark") or ("" if row.valid_as_result else "NOT A VALID RESULT (see run manifest)")
        row.arch = model.get("arch") or ck.get("arch", "")
        row.split = d.get("split", "")
        row.split_id = ds.get("split_id") or ck.get("split_id", "")
        cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
        row.seed = next((v for v in (d.get("seed"), cfg.get("seed"), ck.get("training_seed")) if v is not None), None)
        if kind == "train":
            row.checkpoint_path = d.get("artifacts", {}).get("best.pt", "")
            row.config_json = json.dumps(d.get("config", {}), default=str)
        else:
            row.checkpoint_path = ck.get("path", "")
            row.checkpoint_sha256 = ck.get("sha256", "")
            row.config_json = json.dumps(d.get("config", d.get("settings", {})), default=str)
        row.metrics_json = json.dumps(_headline(kind, d), default=str)
        row.indexed_at = datetime.utcnow()

    def runs(self, kind: str | None = None) -> list[Run]:
        with DBSession(self.engine, expire_on_commit=False) as s:
            q = select(Run).order_by(Run.created_utc.desc(), Run.id.desc())
            if kind:
                q = q.where(Run.kind == kind)
            return list(s.scalars(q))

    def get_run(self, run_id: int) -> Run | None:
        with DBSession(self.engine, expire_on_commit=False) as s:
            return s.get(Run, run_id)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.runs():
            out[r.kind] = out.get(r.kind, 0) + 1
        return out

    # ---------------------------------------------------------------- jobs
    def start_job(self, kind: str, label: str, argv: list[str], *, cwd: str | Path, log_dir: str | Path, out_dir: str = "") -> Job:
        """Run `argv` in the background; stdout/stderr go to a log file. The process survives Streamlit reruns."""
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        with DBSession(self.engine, expire_on_commit=False) as s:
            job = Job(kind=kind, label=label, command=" ".join(argv), out_dir=out_dir, log_path="")
            s.add(job)
            s.flush()
            job.log_path = str(Path(log_dir) / f"job_{job.id}.log")
            fh = open(job.log_path, "w")                                  # noqa: SIM115 - handed to the child process
            rc_file = job.log_path[:-4] + ".rc"                           # exit code survives server restarts / reaping
            try:
                # A small Python supervisor writes the exit code after the child exits.
                # This avoids relying on a Unix shell and works on Windows and POSIX.
                runner = (
                    "import json, pathlib, subprocess, sys\n"
                    "argv = json.loads(sys.argv[1])\n"
                    "rc_path = pathlib.Path(sys.argv[2])\n"
                    "try:\n"
                    "    result = subprocess.run(argv, check=False)\n"
                    "    code = result.returncode\n"
                    "except BaseException:\n"
                    "    code = 1\n"
                    "tmp = rc_path.with_suffix(rc_path.suffix + '.tmp')\n"
                    "tmp.write_text(str(code), encoding='utf-8')\n"
                    "tmp.replace(rc_path)\n"
                    "sys.exit(code)\n"
                )
                popen_options: dict[str, Any] = {}
                if os.name == "nt":
                    popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    popen_options["start_new_session"] = True
                proc = subprocess.Popen(
                    [sys.executable, "-u", "-c", runner, json.dumps(argv), rc_file],
                    cwd=str(cwd), stdout=fh, stderr=subprocess.STDOUT,
                    env={**os.environ, "PYTHONUNBUFFERED": "1"}, **popen_options,
                )
            except OSError as exc:
                job.status, job.returncode, job.ended = "failed", -1, datetime.utcnow()
                fh.write(f"could not start: {exc}\n")
            else:
                job.pid = proc.pid
            finally:
                fh.close()
            s.commit()
            return job

    def refresh_job(self, job_id: int) -> Job:
        """Update a running job from its exit-code file or process liveness."""
        with DBSession(self.engine, expire_on_commit=False) as s:
            job = s.get(Job, job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status == "running":
                rc_file = Path(job.log_path[:-4] + ".rc")
                if rc_file.exists():
                    try:
                        job.returncode = int(rc_file.read_text().strip())
                    except ValueError:
                        job.returncode = -1
                    job.status = "succeeded" if job.returncode == 0 else "failed"
                    job.ended = datetime.utcnow()
                    s.commit()
                elif job.pid and not _alive(job.pid):                       # died without writing an exit code (killed)
                    job.status, job.returncode, job.ended = "failed", -9, datetime.utcnow()
                    s.commit()
            return job

    def cancel_job(self, job_id: int) -> Job:
        with DBSession(self.engine, expire_on_commit=False) as s:
            job = s.get(Job, job_id)
            if job and job.status == "running" and job.pid:
                if Path(job.log_path[:-4] + ".rc").exists():           # already finished: record the real outcome, do not call it cancelled
                    return self.refresh_job(job_id)
                try:
                    if os.name == "nt":
                        # /T terminates the supervisor and its child process tree.
                        subprocess.run(
                            ["taskkill", "/PID", str(job.pid), "/T", "/F"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            check=False, timeout=10,
                        )
                    else:
                        os.killpg(job.pid, 15)
                except (ProcessLookupError, subprocess.TimeoutExpired):
                    pass
                except OSError as exc:
                    log.warning("could not terminate job %s (pid %s): %s", job.id, job.pid, exc)
                job.status, job.ended = "cancelled", datetime.utcnow()
                s.commit()
            return job

    def jobs(self, limit: int = 50) -> list[Job]:
        with DBSession(self.engine, expire_on_commit=False) as s:
            ids = [j.id for j in s.scalars(select(Job).order_by(Job.id.desc()).limit(limit))]
        return [self.refresh_job(i) for i in ids]

    # ---------------------------------------------------------------- reports
    def add_report(self, title: str, path: str | Path, run_ids: list[int], contains_non_results: bool) -> ReportRecord:
        with DBSession(self.engine, expire_on_commit=False) as s:
            rec = s.scalar(select(ReportRecord).where(ReportRecord.path == str(path))) or ReportRecord(path=str(path), title=title)
            rec.title, rec.run_ids = title, ",".join(map(str, run_ids))
            rec.contains_non_results, rec.size_bytes = contains_non_results, float(Path(path).stat().st_size)
            rec.created = datetime.utcnow()
            s.add(rec)
            s.commit()
            return rec

    def reports(self) -> list[ReportRecord]:
        with DBSession(self.engine, expire_on_commit=False) as s:
            return list(s.scalars(select(ReportRecord).order_by(ReportRecord.created.desc())))


def _alive(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is not a safe process-existence probe on Windows.
        # Query the process handle instead; access denied means it exists.
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return ctypes.get_last_error() == 5  # ERROR_ACCESS_DENIED => process exists
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return True
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)

    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                                                  # a zombie child is dead for our purposes on Linux
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except OSError:
        return True
