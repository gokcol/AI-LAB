"""Read aggregate counts only. The web process never reads access logs."""
from datetime import date, datetime, timedelta, timezone
from html import escape
import json
import math
import os
from pathlib import Path

START = date(2026, 10, 1)


def validate(raw):
    if raw.get("version") != 1 or raw.get("start") != START.isoformat():
        raise ValueError("Unknown statistics format")
    updated = datetime.fromisoformat(raw["updated"])
    if updated.tzinfo is None:
        raise ValueError("Statistics need a timezone")
    end = date.fromisoformat(raw["through"])
    rows = raw["daily"]
    if not START <= end < updated.astimezone(timezone.utc).date() or len(rows) != (end-START).days+1:
        raise ValueError("Statistics must cover completed UTC days")
    clean = []
    for offset, row in enumerate(rows):
        count = row["visits"]
        if row["day"] != (START+timedelta(days=offset)).isoformat() or not (count is None or type(count) is int and count >= 0):
            raise ValueError("Invalid daily statistics")
        clean.append({"day": row["day"], "visits": count})
    return {"version": 1, "start": START.isoformat(), "through": end.isoformat(),
            "updated": updated.astimezone(timezone.utc).isoformat(), "daily": clean}


def summarize(data):
    data = validate(data)
    known = [row for row in data["daily"] if row["visits"] is not None]
    total = sum(row["visits"] for row in known)
    weeks = {}
    for row in data["daily"]:
        day = date.fromisoformat(row["day"])
        monday = day-timedelta(days=day.weekday())
        week = weeks.setdefault(monday.isoformat(), {"Week starting": monday.isoformat(), "Recorded visits": 0,
                    "Recorded days": 0, "Days in window": 0})
        week["Days in window"] += 1
        if row["visits"] is not None:
            week["Recorded visits"] += row["visits"]
            week["Recorded days"] += 1
    for week in weeks.values():
        week["Visits"] = week["Recorded visits"] if week["Recorded days"] == week["Days in window"] else None
        week["Coverage"] = "Missing history" if week["Visits"] is None else ("Partial week" if week["Days in window"] < 7 else "Full week")
    return {"total": total if known else None, "daily_average": total/len(known) if known else None,
            "known_days": len(known), "missing_days": len(data["daily"])-len(known), "weeks": list(weeks.values())}


def weekly_chart(weeks):
    """Inline SVG inherits the app palette, including browser-default theme changes."""
    width = max(960, len(weeks) * 20 + 80)
    left, right, top, bottom = 70, width - 20, 32, 272
    peak = max((w["Visits"] or 0 for w in weeks), default=0)
    # Four integer grid intervals, with headroom and a useful scale for all-zero data.
    magnitude = 10 ** max(0, math.floor(math.log10(max(peak / 4, 1))))
    step = max(1, math.ceil(peak / 4 / magnitude)) * magnitude
    ceiling = step * 4
    slot = (right - left) / max(len(weeks), 1)
    bar_width = min(100, slot * .65)
    svg = [f'<div style="overflow-x:auto"><svg xmlns="http://www.w3.org/2000/svg" '
           f'viewBox="0 0 {width} 334" role="group" aria-label="Weekly visits over time" '
           'style="display:block;width:100%;min-width:640px;background:transparent;'
           'font-family:inherit;font-size:13px;fill:var(--ailab-ink,#172033)">']
    svg.append(f'<text x="{left}" y="18">Visits</text>')
    for tick in range(5):
        y = bottom - tick / 4 * (bottom - top)
        svg.append(f'<line x1="{left}" y1="{y}" x2="{right}" y2="{y}" '
                   'stroke="var(--ailab-border,#d8e3ef)"/>')
        svg.append(f'<text x="{left - 10}" y="{y + 4}" text-anchor="end">{tick * step:,}</text>')
    label_every = max(1, math.ceil(len(weeks) / 7))
    for index, week in enumerate(weeks):
        x = left + (index + .5) * slot
        count = week["Visits"]
        value = f'{count:,} visits' if count is not None else 'Unavailable'
        detail = (f'Week starting {week["Week starting"]}: {value}. {week["Coverage"]}; '
                  f'{week["Recorded days"]} of {week["Days in window"]} days in this window recorded.')
        if count is None:
            detail += f' Recorded subtotal: {week["Recorded visits"]:,} visits.'
        label = escape(detail, quote=True)
        svg.append(f'<g role="img" tabindex="0" aria-label="{label}" '
                   f'data-week="{week["Week starting"]}"><title>{label}</title>')
        if count is None:
            svg.append(f'<text x="{x}" y="{bottom - 10}" text-anchor="middle">—</text>')
        elif count == 0:
            svg.append(f'<text x="{x}" y="{bottom - 10}" text-anchor="middle">0</text>')
        else:
            height = count / ceiling * (bottom - top)
            svg.append(f'<rect x="{x - bar_width / 2}" y="{bottom - height}" '
                       f'width="{bar_width}" height="{height}" rx="3" fill="var(--ailab-blue,#2563eb)"/>')
            if len(weeks) <= 12:
                svg.append(f'<text x="{x}" y="{bottom - height - 8}" text-anchor="middle">{count:,}</text>')
        svg.append('</g>')
        if index % label_every == 0:
            svg.append(f'<text x="{x}" y="{bottom + 24}" text-anchor="middle">{week["Week starting"]}</text>')
    svg.append(f'<text x="{(left + right) / 2}" y="326" text-anchor="middle">Week starting (Monday, UTC)</text></svg></div>')
    return ''.join(svg)


def render():
    import streamlit as st

    @st.cache_data(ttl=300, show_spinner=False)
    def load(path):
        text = Path(path).read_text(encoding="utf-8")
        if len(text) > 1_000_000:
            raise ValueError("Statistics file too large")
        return validate(json.loads(text))

    st.markdown("### 📊 AI Lab in numbers")
    st.caption("Visits mean successful direct page loads, including reloads. Known bots, probes and configured internal traffic are excluded. These are not unique people; navigation within an open app tab may not create a new page load.")
    try:
        data = load(os.environ.get("AILAB_PUBLIC_STATS_FILE", "/opt/ai-lab/shared/public-stats.json"))
        summary = summarize(data)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        st.info("Visit statistics are not available yet. Missing data is not counted as zero.")
        return
    left, right = st.columns(2)
    label = "Recorded visits since 1 Oct 2026" if summary["missing_days"] else "Total visits since 1 Oct 2026"
    left.metric(label, f"{summary['total']:,}" if summary["total"] is not None else "Unavailable")
    average = summary["daily_average"]
    right.metric("Daily average visits", f"{average:,.1f}" if average is not None else "Unavailable")
    st.caption(f"Through {data['through']} (completed UTC days). Average uses {summary['known_days']} recorded days. Updated {data['updated'][:16].replace('T', ' ')} UTC.")
    if summary["missing_days"]:
        st.warning(f"{summary['missing_days']} days have unavailable history. The total includes recorded visits only; missing days are excluded from the average and incomplete weekly bars.")
    if datetime.now(timezone.utc)-datetime.fromisoformat(data["updated"]) > timedelta(days=2):
        st.warning("Statistics have not refreshed in more than two days; the figures above are the last available snapshot.")
    st.markdown("#### Weekly visits over time")
    st.markdown(weekly_chart(summary["weeks"]), unsafe_allow_html=True)
    st.caption("Weeks run Monday–Sunday. The first and latest weeks may cover fewer than seven days. Hover over a bar for its count and coverage. A dash means unavailable history, not zero.")

