"""LiverGuard shared design system with Light and Dark themes."""
from __future__ import annotations

import html
from contextlib import contextmanager

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

NAVY, TEAL, LAVENDER = "#14213D", "#3E8E8B", "#B8B3D8"
WARN, BAD, GOOD = "#B7791F", "#B4442F", "#2F7D5B"
SERIES = ["#2F6FD0", "#E07B1F", "#1B9E9E", "#7A5AA6", "#6B7280"]
MARKERS = ["circle", "square", "diamond", "triangle-up", "x"]
DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]
DISCLAIMER = ("LiverGuard is a research and model-auditing platform. It is not a clinical diagnostic system; "
              "nothing shown here is clinically validated.")

BG, SURFACE, INK, MUTED, LINE = "#F7F5F2", "#FFFFFF", "#14213D", "#5B6475", "#E4E0DA"


def _colors(mode: str) -> dict[str, str]:
    if mode == "Dark":
        return dict(BG="#0B1220", SURFACE="#131F32", INK="#E8EEF7", MUTED="#A7B4C7",
                    NAVY="#101B2D", TEAL="#54B8B0", LINE="#26364B",
                    WARN="#E7B968", BAD="#F08B83", GOOD="#67C69A")
    return dict(BG="#F7F5F2", SURFACE="#FFFFFF", INK="#14213D", MUTED="#5B6475",
                NAVY="#14213D", TEAL="#3E8E8B", LINE="#E4E0DA",
                WARN="#B7791F", BAD="#B4442F", GOOD="#2F7D5B")


def _css(c: dict[str, str]) -> str:
    dark = c["BG"] == "#0B1220"
    badge_ok = "#173B31" if dark else "#E3F1EA"
    badge_warn = "#49391F" if dark else "#F7EBD5"
    badge_bad = "#492927" if dark else "#F5E1DC"
    badge_info = "#26344B" if dark else "#E8E6F3"
    banner_warn = "#30291D" if dark else "#FBF4E4"
    banner_bad = "#352221" if dark else "#F9EAE6"
    info_text = "#C5C0F0" if dark else "#4B4580"
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');
:root {{ color-scheme: {'dark' if dark else 'light'}; }}
html, body, [class*="css"] {{ font-family: Inter, system-ui, sans-serif; color:{c['INK']}; }}
.stApp, [data-testid="stAppViewContainer"] {{ background:{c['BG']}; color:{c['INK']}; }}
[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:{c['NAVY']}; }}
[data-testid="stSidebar"] * {{ color:#E9E7F2; }}
[data-testid="stSidebar"] code {{ background:transparent!important; color:#C9D4E5!important; font-size:.7rem; word-break:break-all; }}
[data-testid="stSidebar"] .stButton>button {{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.35); }}
[data-testid="stSidebar"] .stButton>button p {{ color:#fff!important; }}
[data-testid="stSidebar"] a[aria-current="page"], [data-testid="stSidebar"] [data-testid="stSidebarNavLink"][aria-current="page"] {{ background:rgba(255,255,255,.10); border-radius:10px; }}
h1,h2,h3 {{ font-family:'Source Serif 4',Georgia,serif; color:{c['INK']}; letter-spacing:-.01em; }}
.lg-header {{ padding:4px 0 14px; border-bottom:1px solid {c['LINE']}; margin-bottom:18px; }}
.lg-header h1 {{ margin:0; font-size:1.9rem; color:{c['INK']}; }}
.lg-header p {{ margin:4px 0 0; color:{c['MUTED']}; font-size:.95rem; }}
.lg-card,.lg-kpi {{ background:{c['SURFACE']}; border:1px solid {c['LINE']}; border-radius:14px; }}
.lg-card {{ padding:16px 18px; margin-bottom:14px; }}
.lg-card h4 {{ margin:0 0 6px; color:{c['INK']}; }}
.lg-kpi {{ padding:14px 16px; height:100%; }}
.lg-kpi .l {{ color:{c['MUTED']}; font-size:.75rem; text-transform:uppercase; letter-spacing:.06em; }}
.lg-kpi .v {{ color:{c['INK']}; font-size:1.45rem; font-weight:600; line-height:1.25; }}
.lg-kpi .n {{ color:{c['MUTED']}; font-size:.8rem; }}
.lg-badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:.72rem; font-weight:600; margin-right:6px; }}
.lg-ok {{ background:{badge_ok}; color:{c['GOOD']}; }} .lg-warn {{ background:{badge_warn}; color:{c['WARN']}; }}
.lg-bad {{ background:{badge_bad}; color:{c['BAD']}; }} .lg-info {{ background:{badge_info}; color:{info_text}; }}
.lg-banner {{ border-left:4px solid {c['WARN']}; background:{banner_warn}; color:{c['INK']}; padding:10px 14px; border-radius:8px; margin-bottom:14px; }}
.lg-banner.bad {{ border-color:{c['BAD']}; background:{banner_bad}; }}
.lg-empty {{ border:1px dashed {c['LINE']}; border-radius:14px; padding:26px; text-align:center; color:{c['MUTED']}; background:{c['SURFACE']}; }}
.lg-brand {{ font-family:'Source Serif 4',Georgia,serif; font-size:1.5rem; font-weight:600; }}
.lg-brand-sub {{ font-size:.75rem; opacity:.75; margin-bottom:10px; }}
.lg-disc {{ font-size:.72rem; opacity:.75; margin-top:14px; border-top:1px solid rgba(255,255,255,.2); padding-top:8px; }}
.lg-foot {{ color:{c['MUTED']}; font-size:.78rem; margin-top:28px; border-top:1px solid {c['LINE']}; padding-top:10px; }}
div[data-testid="stMetric"] {{ background:{c['SURFACE']}; border:1px solid {c['LINE']}; border-radius:12px; padding:10px 14px; }}
[data-testid="stDataFrame"],[data-testid="stTable"] {{ background:{c['SURFACE']}; }}
.stButton>button {{ border-radius:10px; border:1px solid {c['NAVY']}; }}
.stButton>button[kind="primary"] {{ background:{c['NAVY']}; color:white; }}
</style>"""


def apply_theme(mode: str | None = None) -> None:
    global BG, SURFACE, INK, MUTED, LINE, NAVY, TEAL, WARN, BAD, GOOD
    mode = mode or st.session_state.get("lg_theme", "Light")
    if mode not in ("Light", "Dark"):
        mode = "Light"
    c = _colors(mode)
    BG, SURFACE, INK, MUTED, LINE = (c[k] for k in ("BG", "SURFACE", "INK", "MUTED", "LINE"))
    NAVY, TEAL, WARN, BAD, GOOD = (c[k] for k in ("NAVY", "TEAL", "WARN", "BAD", "GOOD"))
    st.markdown(_css(c), unsafe_allow_html=True)
    grid = "#26364B" if mode == "Dark" else "#EEEAE4"
    pio.templates["liverguard"] = go.layout.Template(layout=go.Layout(
        font=dict(family="Inter, system-ui, sans-serif", color=INK, size=13),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, colorway=SERIES,
        margin=dict(l=50, r=20, t=50, b=45),
        xaxis=dict(gridcolor=grid, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK)),
        yaxis=dict(gridcolor=grid, zerolinecolor=LINE, linecolor=LINE, tickfont=dict(color=INK)),
        legend=dict(bgcolor="rgba(255,255,255,0)", orientation="h", y=-0.2),
        title=dict(font=dict(size=15, color=INK), x=0.01)))
    pio.templates.default = "liverguard"


def page_header(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="lg-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def kpi(col, label: str, value: str, note: str = "") -> None:
    col.markdown(f'<div class="lg-kpi"><div class="l">{html.escape(label)}</div><div class="v">{html.escape(value)}</div>'
                 f'<div class="n">{html.escape(note)}</div></div>', unsafe_allow_html=True)


def badge(text: str, tone: str = "info") -> str:
    return f'<span class="lg-badge lg-{tone}">{html.escape(text)}</span>'


def banner(text: str, tone: str = "warn") -> None:
    st.markdown(f'<div class="lg-banner {"bad" if tone == "bad" else ""}">{html.escape(text)}</div>', unsafe_allow_html=True)


def watermark_banner(watermark: str | None) -> None:
    if watermark:
        banner(f"{watermark}. These numbers come from a pipeline check, not from a valid experiment; do not quote them as findings.", "bad")


def empty_state(message: str, hint: str = "") -> None:
    st.markdown(f'<div class="lg-empty"><b>{html.escape(message)}</b><br><span>{html.escape(hint)}</span></div>', unsafe_allow_html=True)


@contextmanager
def card(title: str | None = None):
    st.markdown(f'<div class="lg-card">{f"<h4>{html.escape(title)}</h4>" if title else ""}', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def limitations(items: dict[str, str] | list[str], title: str = "Limitations of this method") -> None:
    with st.expander(title):
        for k, v in (items.items() if isinstance(items, dict) else enumerate(items)):
            st.markdown(f"- **{k}**: {v}" if isinstance(items, dict) else f"- {v}")


def footer() -> None:
    st.markdown(f'<div class="lg-foot">{html.escape(DISCLAIMER)}</div>', unsafe_allow_html=True)


def fmt(x, nd: int = 3) -> str:
    try:
        if x is None or x != x:
            return "n/a"
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)
