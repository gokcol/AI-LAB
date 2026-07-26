"""Small shared UI polish helpers for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ailab-ink: #172033;
            --ailab-muted: #607086;
            --ailab-border: #d8e3ef;
            --ailab-panel: rgba(255, 255, 255, 0.92);
            --ailab-blue: #2563eb;
            --ailab-green: #15916f;
            --ailab-amber: #c47a18;
            --ailab-coral: #e05f5f;
        }

        .stApp {
            background:
                linear-gradient(180deg, #fbfdff 0%, #f6fbf7 46%, #fffaf3 100%);
            color: var(--ailab-ink);
        }

        h1, h2, h3 {
            letter-spacing: 0;
        }

        h1 {
            font-weight: 780;
            color: #111827;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f4f8ff 48%, #f7fbf4 100%);
            border-right: 1px solid var(--ailab-border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] * {
            color: var(--ailab-ink);
        }

        [data-testid="stSidebar"] .stCaptionContainer,
        [data-testid="stSidebar"] p {
            color: var(--ailab-muted);
        }

        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label {
            color: #475569 !important;
            font-weight: 650;
        }

        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button,
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label div {
            background: #ffffff !important;
            border-color: #cbd5e1 !important;
            color: #1f2937 !important;
            font-weight: 720;
        }

        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button *,
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] label div * {
            color: inherit !important;
        }

        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-pressed="true"],
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[aria-checked="true"],
        [data-testid="stSidebar"] [data-testid="stSegmentedControl"] button[data-selected="true"] {
            background: #2563eb !important;
            border-color: #2563eb !important;
            color: #ffffff !important;
            box-shadow: 0 8px 20px rgba(37, 99, 235, 0.18);
        }

        [data-testid="stSidebar"] a {
            border-radius: 8px;
            padding: 0.18rem 0.3rem;
            color: #243044 !important;
        }

        [data-testid="stSidebar"] a:hover {
            background: rgba(37, 99, 235, 0.08);
            color: #1d4ed8 !important;
        }

        div[data-testid="stMetric"] {
            background: var(--ailab-panel);
            border: 1px solid var(--ailab-border);
            border-radius: 8px;
            padding: 0.85rem 1rem;
            box-shadow: 0 8px 24px rgba(21, 34, 53, 0.06);
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: var(--ailab-border);
            box-shadow: 0 10px 30px rgba(21, 34, 53, 0.05);
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 8px;
            border-color: #c9d6e6;
            font-weight: 650;
        }

        .stButton > button[kind="primary"] {
            background: #1f6feb;
            border-color: #1f6feb;
        }

        .ailab-hero {
            margin: 0.2rem 0 1.3rem;
            padding: 1.25rem 1.35rem;
            border: 1px solid var(--ailab-border);
            border-radius: 8px;
            background:
                linear-gradient(135deg, rgba(255,255,255,0.98), rgba(236,246,255,0.94) 56%, rgba(255,247,232,0.92));
            box-shadow: 0 16px 38px rgba(36, 48, 68, 0.07);
        }

        .ailab-hero h2 {
            margin: 0 0 0.35rem;
            font-size: 1.55rem;
            line-height: 1.2;
        }

        .ailab-hero p {
            margin: 0;
            color: var(--ailab-muted);
            font-size: 1rem;
            line-height: 1.55;
        }

        .ailab-callout {
            padding: 0.95rem 1rem;
            border-left: 4px solid var(--ailab-green);
            background: rgba(232, 248, 241, 0.75);
            border-radius: 8px;
            color: #173b31;
        }

        /* ---------------- tabs: bigger touch targets, clear active state ------- */
        button[data-baseweb="tab"] {
            font-size: 0.96rem;
            font-weight: 620;
            padding: 0.55rem 0.95rem;
            border-radius: 8px 8px 0 0;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: rgba(37, 99, 235, 0.07);
            color: #1d4ed8 !important;
        }
        div[data-baseweb="tab-highlight"] { background-color: #2563eb; height: 3px; }
        /* keep the strip swipeable instead of squashing labels on narrow screens */
        div[data-baseweb="tab-list"] {
            gap: 0.15rem;
            overflow-x: auto;
            scrollbar-width: none;
            -webkit-overflow-scrolling: touch;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        button[data-baseweb="tab"] { flex: 0 0 auto; }

        /* ---------------- mobile ------------------------------------------------ */
        @media (max-width: 640px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            button[data-baseweb="tab"] { font-size: 0.9rem; padding: 0.5rem 0.7rem; }
            .ailab-hero { padding: 0.9rem 1rem; }
            .ailab-hero h2 { font-size: 1.25rem; }
            /* metric rows get cramped at 4-across on a phone */
            div[data-testid="stMetricValue"] { font-size: 1.15rem; }
            div[data-testid="stMetricLabel"] { font-size: 0.72rem; }
        }

        /* feedback honeypot — invisible to humans, present in the DOM for bots */
        .st-key-fb_hp { display: none !important; }

        .ailab-step {
            padding: 0.8rem 0.9rem;
            border: 1px solid var(--ailab-border);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.78);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="ailab-hero">
            <h2>{title}</h2>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
