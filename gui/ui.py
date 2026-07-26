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

        /* ---------------- tabs: pill navigation -------------------------------- */
        div[data-baseweb="tab-list"] {
            gap: 0.45rem;
            border-bottom: none !important;
            padding-bottom: 0.35rem;
            overflow-x: auto;
            scrollbar-width: none;
            -webkit-overflow-scrolling: touch;
        }
        div[data-baseweb="tab-list"]::-webkit-scrollbar { display: none; }
        /* the default underline/baseline get in the way of pills */
        div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] { display: none !important; }

        button[data-baseweb="tab"] {
            flex: 0 0 auto;
            background: #FFFFFF;
            border: 1.5px solid #DDE5EF;
            border-radius: 11px;
            padding: 0.55rem 1.05rem !important;
            font-size: 0.95rem;
            font-weight: 650;
            color: #46566B !important;
            box-shadow: 0 1px 2px rgba(23, 32, 51, 0.04);
            transition: transform .12s ease, box-shadow .12s ease,
                        border-color .12s ease, background .12s ease;
        }
        button[data-baseweb="tab"] * { color: inherit !important; }
        button[data-baseweb="tab"]:hover {
            border-color: #9CBBE8;
            background: #F5F9FF;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(37, 99, 235, 0.10);
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            background: linear-gradient(135deg, #2F7BEA 0%, #1D4ED8 100%);
            border-color: #1D4ED8;
            color: #FFFFFF !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.30);
            transform: translateY(-1px);
        }
        button[data-baseweb="tab"][aria-selected="true"] * { color: #FFFFFF !important; }

        /* ---------------- stat cards ------------------------------------------- */
        .ailab-stats {
            display: flex; flex-wrap: wrap; gap: 0.7rem; margin: 0.5rem 0 1.1rem;
        }
        .ailab-stat {
            flex: 1 1 150px;
            position: relative;
            overflow: hidden;
            border-radius: 14px;
            padding: 0.85rem 1rem 0.9rem;
            color: #fff;
            box-shadow: 0 8px 20px rgba(23, 32, 51, 0.13);
            transition: transform .16s ease, box-shadow .16s ease;
        }
        .ailab-stat:hover { transform: translateY(-3px); box-shadow: 0 14px 28px rgba(23,32,51,.20); }
        .ailab-stat::after {                 /* soft glow blob */
            content: ""; position: absolute; right: -26px; top: -30px;
            width: 92px; height: 92px; border-radius: 50%;
            background: rgba(255,255,255,0.16);
        }
        .ailab-stat .ic   { font-size: 1.12rem; opacity: .95; }
        .ailab-stat .num  { font-size: 2rem; font-weight: 800; line-height: 1.05; margin-top: .1rem;
                            letter-spacing: -0.5px; text-shadow: 0 1px 2px rgba(0,0,0,.12); }
        .ailab-stat .lab  { font-size: .78rem; font-weight: 600; opacity: .95;
                            letter-spacing: .02em; text-transform: uppercase; }
        .s-blue  { background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%); }
        .s-green { background: linear-gradient(135deg, #22C08A 0%, #0E8A5F 100%); }
        .s-amber { background: linear-gradient(135deg, #F0A73A 0%, #C2761A 100%); }
        .s-plum  { background: linear-gradient(135deg, #A970D8 0%, #7040A8 100%); }

        /* ---------------- feedback card ---------------------------------------- */
        .ailab-fb-head {
            border: 1px solid #CFE0F5;
            border-left: 5px solid #2563eb;
            border-radius: 12px;
            padding: 0.95rem 1.15rem;
            margin: 0.2rem 0 0.9rem;
            background: linear-gradient(135deg, #F4F9FF 0%, #EFF6F1 100%);
            box-shadow: 0 8px 22px rgba(36, 48, 68, 0.06);
        }
        .ailab-fb-head h4 { margin: 0 0 0.3rem; font-size: 1.12rem; color: #14243A; }
        .ailab-fb-head p  { margin: 0; color: #56697F; font-size: 0.94rem; line-height: 1.5; }
        .ailab-chip {
            display: inline-block;
            background: #FFFFFF;
            border: 1px solid #D8E4F2;
            border-radius: 999px;
            padding: 0.12rem 0.62rem;
            margin: 0.4rem 0.3rem 0 0;
            font-size: 0.76rem;
            font-weight: 600;
            color: #3C5A7E;
        }
        /* the form itself */
        .st-key-fb_form {
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 1.05rem 1.15rem 0.6rem !important;
            background: #FFFFFF;
            box-shadow: 0 6px 20px rgba(23, 32, 51, 0.05);
        }
        .st-key-fb_form textarea {
            border-radius: 10px !important;
            font-size: 0.97rem !important;
            line-height: 1.5 !important;
        }
        .st-key-fb_form textarea:focus {
            border-color: #2563eb !important;
            box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.14) !important;
        }
        .st-key-fb_form input { border-radius: 9px !important; }
        .st-key-fb_form button[kind="primaryFormSubmit"],
        .st-key-fb_form button[kind="primary"] {
            background: linear-gradient(135deg, #2F7BEA 0%, #1D4ED8 100%) !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 680 !important;
            padding: 0.55rem 1.4rem !important;
            box-shadow: 0 6px 16px rgba(37, 99, 235, 0.28) !important;
            transition: transform .12s ease, box-shadow .12s ease;
        }
        .st-key-fb_form button[kind="primaryFormSubmit"]:hover,
        .st-key-fb_form button[kind="primary"]:hover {
            transform: translateY(-1px);
            box-shadow: 0 9px 22px rgba(37, 99, 235, 0.36) !important;
        }

        /* ---------------- mobile ------------------------------------------------ */
        @media (max-width: 640px) {
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
            button[data-baseweb="tab"] {
                font-size: 0.88rem;
                padding: 0.5rem 0.8rem !important;
                border-radius: 10px;
            }
            .ailab-hero { padding: 0.9rem 1rem; }
            .ailab-hero h2 { font-size: 1.25rem; }
            .ailab-fb-head { padding: 0.8rem 0.9rem; }
            .ailab-stat { flex: 1 1 44%; padding: 0.7rem 0.8rem; }
            .ailab-stat .num { font-size: 1.6rem; }
            .ailab-stat .lab { font-size: .7rem; }
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
