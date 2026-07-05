"""Theme tokens, CSS injection, and Plotly styling for the dashboard.

Two paired themes are supported: a Nordic Steel light mode and a Graphite dark
mode. Custom-rendered surfaces (KPI cards, charts) read explicit palette values
so they render correctly in either mode regardless of the Streamlit base theme.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

PALETTES: dict[str, dict[str, str]] = {
    "light": {
        "page": "#F7F8FA",
        "surface": "#FFFFFF",
        "panel": "#FDFDFE",
        "text": "#1C2530",
        "text_secondary": "#64748B",
        "text_muted": "#94A3B8",
        "border": "#E5E9EF",
        "border_strong": "#D5DBE3",
        "accent": "#3E5C76",
        "accent_soft": "#EAF0F5",
        "on_accent": "#FFFFFF",
        "success": "#4B7F6B",
        "warning": "#B4785A",
        "danger": "#A6534B",
    },
    "dark": {
        "page": "#17181A",
        "surface": "#1F2124",
        "panel": "#24262A",
        "text": "#E6E7E9",
        "text_secondary": "#9AA0A6",
        "text_muted": "#6B7076",
        "border": "#2C2F33",
        "border_strong": "#3A3E42",
        "accent": "#6FA292",
        "accent_soft": "#232B29",
        "on_accent": "#17181A",
        "success": "#6FA292",
        "warning": "#C79A6B",
        "danger": "#C4726A",
    },
}


def get_mode() -> str:
    """Return the active theme mode from session state (default light)."""
    return st.session_state.get("theme_mode", "light")


def palette() -> dict[str, str]:
    """Return the active palette."""
    return PALETTES[get_mode()]


def inject_css() -> None:
    """Inject global CSS for spacing, typography, and dark-mode surfaces."""
    p = palette()
    dark = get_mode() == "dark"
    dark_rules = (
        f"""
        .stApp {{ background: {p['page']}; color: {p['text']}; }}
        section[data-testid="stSidebar"] {{ background: {p['panel']}; }}
        section[data-testid="stSidebar"] * {{ color: {p['text']}; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        h1, h2, h3, h4, p, span, label, li {{ color: {p['text']}; }}
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {p['text']}; }}
        [data-testid="stDataFrame"] {{ background: {p['surface']}; }}
        """
        if dark
        else ""
    )
    st.markdown(
        f"""
        <style>
        .block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px; }}
        #MainMenu, footer {{ visibility: hidden; }}
        h1 {{ font-weight: 500; font-size: 1.7rem; letter-spacing: -0.01em; }}
        h2, h3 {{ font-weight: 500; letter-spacing: -0.01em; }}
        .app-eyebrow {{ font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em;
            color: {p['text_muted']}; margin-bottom: 0.15rem; }}
        .app-sub {{ color: {p['text_secondary']}; font-size: 0.95rem; margin-top: -0.4rem; }}
        .kpi-card {{ background: {p['surface']}; border: 1px solid {p['border']};
            border-radius: 12px; padding: 1rem 1.15rem; }}
        .kpi-label {{ font-size: 0.8rem; color: {p['text_secondary']}; margin-bottom: 0.35rem; }}
        .kpi-value {{ font-size: 1.6rem; font-weight: 500; color: {p['text']}; line-height: 1.1; }}
        .kpi-note {{ font-size: 0.78rem; color: {p['text_muted']}; margin-top: 0.25rem; }}
        .section-title {{ font-size: 1.05rem; font-weight: 500; color: {p['text']};
            margin: 0.4rem 0 0.2rem; }}
        .pill {{ display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 0.78rem; border: 1px solid {p['border_strong']}; color: {p['text_secondary']}; }}
        .divider {{ height: 1px; background: {p['border']}; margin: 1.4rem 0; border: none; }}
        {dark_rules}
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_fig(fig: Any) -> Any:
    """Apply the active palette to a Plotly figure (flat, muted, transparent)."""
    p = palette()
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": p["text_secondary"], "size": 13},
        title={"font": {"color": p["text"], "size": 16}},
        colorway=[p["accent"], p["text_muted"], p["success"], p["warning"]],
        margin={"l": 10, "r": 10, "t": 48, "b": 10},
        legend={"bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": p["surface"], "font_size": 12},
    )
    fig.update_xaxes(
        gridcolor=p["border"], zerolinecolor=p["border"], rangemode="tozero"
    )
    fig.update_yaxes(
        gridcolor=p["border"], zerolinecolor=p["border"], rangemode="tozero"
    )
    return fig


def mode_toggle() -> None:
    """Render the light/dark toggle in the sidebar."""
    current = get_mode()
    is_dark = st.toggle("Dark mode", value=current == "dark", key="theme_toggle")
    new_mode = "dark" if is_dark else "light"
    if new_mode != current:
        st.session_state["theme_mode"] = new_mode
        st.rerun()
    st.session_state.setdefault("theme_mode", new_mode)
