"""Plotly figures built from the CSV/JSON files the ML modules write. Pure functions: DataFrame in, Figure out; no numbers are invented here."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve

from liverguard.app.theme import BAD, DASHES, GOOD, INK, LAVENDER, MARKERS, MUTED, NAVY, SERIES


def _series_style(i: int) -> dict:
    return dict(color=SERIES[i % len(SERIES)], dash=DASHES[i % len(DASHES)])


def _wm(fig: go.Figure, watermark: str | None) -> go.Figure:
    if watermark:
        fig.add_annotation(text="NOT A RESULT", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, textangle=-20,
                           font=dict(size=46, color="rgba(180,68,47,0.16)"))
    return fig


def training_curves(h: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "AUROC"))
    for i, (col, name) in enumerate([("train_loss", "train"), ("val_loss", "validation")]):
        fig.add_scatter(x=h["epoch"], y=h[col], name=name, mode="lines+markers", line=_series_style(i), marker=dict(symbol=MARKERS[i]), row=1, col=1,
                        legendgroup=name)
    for i, (col, name) in enumerate([("train_auroc", "train"), ("val_auroc", "validation")]):
        fig.add_scatter(x=h["epoch"], y=h[col], name=name, mode="lines+markers", line=_series_style(i), marker=dict(symbol=MARKERS[i]), row=1, col=2,
                        legendgroup=name, showlegend=False)
    if "phase" in h and (h["phase"] == "finetune").any():
        first = int(h.loc[h["phase"] == "finetune", "epoch"].min())
        for c in (1, 2):
            fig.add_vline(x=first - 0.5, line_dash="dot", line_color=MUTED, row=1, col=c)
    fig.update_xaxes(title_text="epoch", dtick=1)
    fig.update_layout(height=340, title="Learning curves (dotted line: backbone unfrozen)")
    return _wm(fig, watermark)


def confusion_matrix_fig(cm: list[list[int]], labels=("Hemangioma", "HCC"), title: str = "Confusion matrix") -> go.Figure:
    z = np.asarray(cm)
    fig = go.Figure(go.Heatmap(z=z, x=[f"pred {l}" for l in labels], y=[f"true {l}" for l in labels], colorscale=[[0, "#FFFFFF"], [1, NAVY]],
                               text=z, texttemplate="%{text}", showscale=False, hovertemplate="%{y}, %{x}: %{z}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=320, title=title)
    return fig


def roc_fig(y: np.ndarray, p: np.ndarray, auroc: float | None = None, title: str = "ROC") -> go.Figure:
    fpr, tpr, _ = roc_curve(y, p)
    fig = go.Figure()
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dot"), name="chance")
    fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=SERIES[0], width=2.5, shape="hv"), name="model" + (f" (AUROC {auroc:.3f})" if auroc is not None else ""))
    fig.update_layout(height=320, title=title, xaxis_title="1 - specificity", yaxis_title="sensitivity")
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def degradation_curves(summary: pd.DataFrame, metric: str, conditions: list[str], plausible_max: int = 3, title: str | None = None,
                       watermark: str | None = None) -> go.Figure:
    labels = {"d_auroc": "change in AUROC vs clean", "pfr": "prediction flip rate", "conf_drift": "confidence drift (signed)",
              "d_accuracy": "change in accuracy", "d_f1": "change in F1", "auroc": "AUROC", "mean_abs_dp": "mean |change in p(HCC)|"}
    fig = go.Figure()
    for i, c in enumerate(conditions):
        d = summary[summary["condition"] == c].sort_values("level")
        lo, hi = f"{metric}_lo", f"{metric}_hi"
        err = dict(type="data", symmetric=False, array=d[hi] - d[metric], arrayminus=d[metric] - d[lo], thickness=1) if lo in d and d[lo].notna().all() else None
        fig.add_scatter(x=d["level"], y=d[metric], name=c, mode="lines+markers", line=_series_style(i), marker=dict(symbol=MARKERS[i % 5], size=8),
                        error_y=err, hovertemplate=f"{c}<br>level %{{x}}<br>{metric} %{{y:.3f}}<extra></extra>")
    mx = int(summary["level"].max())
    fig.add_vrect(x0=plausible_max + 0.5, x1=mx + 0.5, fillcolor=LAVENDER, opacity=0.18, line_width=0,
                  annotation_text="stress levels (a-priori tier)", annotation_position="top left")
    fig.update_xaxes(title="severity level", dtick=1)
    fig.update_yaxes(title=labels.get(metric, metric))
    fig.update_layout(height=420, title=title or f"{labels.get(metric, metric)} by severity")
    return _wm(fig, watermark)


def condition_heatmap(summary: pd.DataFrame, value: str, title: str, diverging: bool = False, watermark: str | None = None) -> go.Figure:
    piv = summary.pivot(index="condition", columns="level", values=value)
    scale = ([[0, "#C25E0A"], [0.5, "#F4F1EC"], [1, "#2F6FD0"]] if diverging else [[0, "#F4F7FC"], [1, "#1F4E99"]])
    kw = dict(zmid=0) if diverging else {}
    fig = go.Figure(go.Heatmap(z=piv.values, x=[f"L{c}" for c in piv.columns], y=piv.index, colorscale=scale, text=np.round(piv.values, 2),
                               texttemplate="%{text}", colorbar=dict(thickness=10), **kw,
                               hovertemplate="%{y} %{x}: %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=60 + 34 * len(piv), title=title)
    return _wm(fig, watermark)


def ranking_bar(cond: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    d = cond.sort_values("mean_d_auroc_all")
    fig = go.Figure(go.Bar(x=d["mean_d_auroc_all"], y=d["condition"], orientation="h", marker_color=SERIES[0],
                           error_x=dict(type="data", symmetric=False, array=d["mean_d_auroc_all_hi"] - d["mean_d_auroc_all"],
                                        arrayminus=d["mean_d_auroc_all"] - d["mean_d_auroc_all_lo"]) if d["mean_d_auroc_all_lo"].notna().all() else None,
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="mean change in AUROC over levels (more negative = more disruptive)")
    fig.update_layout(height=60 + 30 * len(d), title="Most disruptive distortions")
    return _wm(fig, watermark)


def reliability_diagram(bins: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=[0.5, 1], y=[0.5, 1], mode="lines", line=dict(color=MUTED, dash="dot"), name="perfect calibration")
    for i, (name, g) in enumerate(bins.groupby("probabilities", sort=False)):
        g = g[g["n"] > 0]
        fig.add_scatter(x=g["confidence"], y=g["accuracy"], mode="lines+markers", name=name, line=_series_style(i), marker=dict(symbol=MARKERS[i], size=np.clip(g["n"] ** 0.5 * 3 + 5, 6, 22)),
                        customdata=g["n"], hovertemplate=f"{name}<br>confidence %{{x:.2f}}<br>accuracy %{{y:.2f}}<br>n=%{{customdata}}<extra></extra>")
    fig.update_xaxes(title="confidence (max class probability)", range=[0.48, 1.02])
    fig.update_yaxes(title="accuracy", range=[-0.02, 1.02])
    fig.update_layout(height=380, title="Reliability diagram (marker size ~ bin count)")
    return _wm(fig, watermark)


def confidence_histogram(per_image: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (name, g) in enumerate([("correct", per_image[per_image["correct"]]), ("wrong", per_image[~per_image["correct"]])]):
        fig.add_histogram(x=g["confidence"], name=f"{name} (n={len(g)})", xbins=dict(start=0.5, end=1.0, size=0.05), marker_color=[SERIES[0], SERIES[1]][i], opacity=0.8)
    fig.update_layout(barmode="overlay", height=320, title="Confidence distribution", xaxis_title="confidence", yaxis_title="images")
    return _wm(fig, watermark)


def risk_coverage_fig(rc: pd.DataFrame, summary: pd.DataFrame | None = None, watermark: str | None = None) -> go.Figure:
    fig = go.Figure()
    for i, (name, g) in enumerate(rc.groupby("score", sort=False)):
        fig.add_scatter(x=g["coverage"], y=g["risk"], name=name, mode="lines", line=dict(**_series_style(i), width=2, shape="hv"))
    if summary is not None and len(summary):
        r = float(summary["aurc_random"].iloc[0])
        fig.add_scatter(x=[0, 1], y=[r, r], mode="lines", name="random abstention", line=dict(color=MUTED, dash="dot"))
    fig.update_layout(height=380, title="Risk-coverage (abstain on the most uncertain)", xaxis_title="coverage (share of images answered)", yaxis_title="error rate on answered images")
    return _wm(fig, watermark)


def masking_bar(m: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    d = m.copy()
    d["control"] = d["variant"].str.startswith("control")
    fig = go.Figure(go.Bar(x=d["variant"], y=d["auroc"], marker_color=[SERIES[1] if c else SERIES[0] for c in d["control"]],
                           error_y=dict(type="data", symmetric=False, array=d["auroc_hi"] - d["auroc"], arrayminus=d["auroc"] - d["auroc_lo"]) if d["auroc_lo"].notna().all() else None,
                           hovertemplate="%{x}: AUROC %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, 1.05], title="AUROC on masked images")
    fig.update_layout(height=360, title="Region masking (orange = matched-area random control)")
    return _wm(fig, watermark)


def paired_forest(p: pd.DataFrame, metric: str, watermark: str | None = None) -> go.Figure:
    d = p[p["metric"] == metric].sort_values("mean_diff")
    colors = {"improved": GOOD, "worse": BAD, "inconclusive": MUTED}
    fig = go.Figure()
    for v, g in d.groupby("verdict"):
        fig.add_scatter(x=g["mean_diff"], y=g["intervention"], mode="markers", name=v, marker=dict(color=colors.get(v, NAVY), size=11, symbol="diamond"),
                        error_x=dict(type="data", symmetric=False, array=g["max_diff"] - g["mean_diff"], arrayminus=g["mean_diff"] - g["min_diff"], thickness=1.5))
    fig.add_vline(x=0, line_color=INK)
    fig.update_layout(height=60 + 40 * len(d), title=f"{metric}: difference vs baseline (bars = min-max over seeds)", xaxis_title="difference (model - baseline)")
    return _wm(fig, watermark)


def tradeoff_scatter(p: pd.DataFrame, watermark: str | None = None) -> go.Figure:
    a = p[p["metric"] == "clean_auroc"].set_index("intervention")
    b = p[p["metric"] == "plaus_auroc"].set_index("intervention")
    names = [n for n in a.index if n in b.index]
    fig = go.Figure(go.Scatter(x=a.loc[names, "mean_diff"], y=b.loc[names, "mean_diff"], mode="markers+text", text=names, textposition="top center",
                               marker=dict(size=11, color=SERIES[0]), hovertemplate="%{text}<br>clean %{x:+.3f}<br>plausible %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color=MUTED); fig.add_vline(x=0, line_color=MUTED)
    fig.update_layout(height=380, title="Trade-off: clean vs plausible-distortion AUROC (diagonal = better model, not more robust)",
                      xaxis_title="change in clean AUROC", yaxis_title="change in plausible-tier AUROC")
    return _wm(fig, watermark)


def sweep_fig(df: pd.DataFrame, threshold: float, title: str, watermark: str | None = None) -> go.Figure:
    """One image: p(HCC) at each severity level of each condition; dotted line = operating threshold."""
    fig = go.Figure()
    for i, (c, g) in enumerate(df.groupby("condition", sort=False)):
        fig.add_scatter(x=g["level"], y=g["p"], name=c, mode="lines+markers", line=_series_style(i), marker=dict(symbol=MARKERS[i % 5], size=8))
    fig.add_hline(y=threshold, line_dash="dot", line_color=MUTED, annotation_text=f"threshold {threshold:.2f}")
    fig.update_xaxes(title="severity level (0 = original)", dtick=1)
    fig.update_yaxes(title="p(HCC)", range=[-0.02, 1.02])
    fig.update_layout(height=380, title=title)
    return _wm(fig, watermark)


def metric_bars(rows: list[dict], metrics: list[str], title: str, watermark: str | None = None) -> go.Figure:
    """rows: [{'model': name, metric: value, ...}] grouped bars, one group per metric."""
    fig = go.Figure()
    for i, r in enumerate(rows):
        fig.add_bar(name=r["model"], x=metrics, y=[r.get(m) for m in metrics], marker_color=SERIES[i % len(SERIES)], marker_pattern_shape=["", "/", ".", "x"][i % 4])
    fig.update_layout(barmode="group", height=380, title=title)
    return _wm(fig, watermark)


def occlusion_heat(occ: np.ndarray, floor: float = 0.1, title: str = "Occlusion sensitivity") -> go.Figure:
    """Signed drop in target-class probability when a patch is hidden (positive = evidence FOR the class). Colour range has a floor so noise is not stretched."""
    lim = max(float(np.abs(occ).max()), floor)
    fig = go.Figure(go.Heatmap(z=occ, colorscale=[[0, "#C25E0A"], [0.5, "#F4F1EC"], [1, "#2F6FD0"]], zmin=-lim, zmax=lim, colorbar=dict(thickness=10, title="Δp"),
                               hovertemplate="x %{x}, y %{y}: Δp %{z:.3f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed", scaleanchor="x")
    fig.update_layout(height=360, title=f"{title} (max |Δp| {np.abs(occ).max():.2f}; colour ±{lim:.2f})", margin=dict(l=30, r=10, t=50, b=30))
    return fig


def roi_bars(df: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure(go.Bar(x=df["variant"], y=df["p_hcc"], marker_color=[SERIES[1] if v.startswith("control") else SERIES[0] for v in df["variant"]],
                           error_y=dict(type="data", symmetric=False, array=df["p_max"] - df["p_hcc"], arrayminus=df["p_hcc"] - df["p_min"]),
                           hovertemplate="%{x}: p(HCC) %{y:.3f}<extra></extra>"))
    fig.add_hline(y=threshold, line_dash="dot", line_color=MUTED, annotation_text="threshold")
    fig.update_yaxes(range=[0, 1.02], title="p(HCC)")
    fig.update_layout(height=340, title="Prediction on region-masked versions (orange = matched-area random controls; bars = min-max)")
    return fig
