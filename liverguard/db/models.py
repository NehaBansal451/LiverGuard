"""Relational schema. The manifests written by the ML modules are the source of truth; these rows index them."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    """One finished experiment: training run, robustness audit, reliability analysis, explainability analysis, M6 comparison."""
    __tablename__ = "runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)          # train|audit|reliability|explain|m6_validation|m6_final
    name: Mapped[str] = mapped_column(String(200))
    manifest_path: Mapped[str] = mapped_column(String(1000), unique=True)
    out_dir: Mapped[str] = mapped_column(String(1000))
    created_utc: Mapped[str] = mapped_column(String(40), default="")
    valid_as_result: Mapped[bool] = mapped_column(Boolean, default=False)
    watermark: Mapped[str] = mapped_column(String(300), default="")
    arch: Mapped[str] = mapped_column(String(40), default="")
    split: Mapped[str] = mapped_column(String(10), default="")          # analysed split (audit/reliability/explain)
    split_id: Mapped[str] = mapped_column(String(40), default="")
    seed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    checkpoint_path: Mapped[str] = mapped_column(String(1000), default="")
    checkpoint_sha256: Mapped[str] = mapped_column(String(80), default="")
    config_json: Mapped[str] = mapped_column(Text, default="{}")
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")      # headline numbers copied from the manifest, never computed here
    indexed_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class Job(Base):
    """A background command started from the UI (training, audit, ...). Log and outputs are files."""
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(32))
    label: Mapped[str] = mapped_column(String(200))
    command: Mapped[str] = mapped_column(Text)
    out_dir: Mapped[str] = mapped_column(String(1000), default="")
    log_path: Mapped[str] = mapped_column(String(1000))
    pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="running")  # running|succeeded|failed|cancelled
    returncode: Mapped[int | None] = mapped_column(Integer, nullable=True)
    started: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    ended: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ReportRecord(Base):
    __tablename__ = "reports"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title: Mapped[str] = mapped_column(String(300))
    path: Mapped[str] = mapped_column(String(1000), unique=True)
    run_ids: Mapped[str] = mapped_column(String(500), default="")
    contains_non_results: Mapped[bool] = mapped_column(Boolean, default=True)
    created: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    size_bytes: Mapped[float] = mapped_column(Float, default=0)
