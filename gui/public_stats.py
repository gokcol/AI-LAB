"""Read aggregate counts only. The web process never reads access logs."""
from datetime import date, datetime, timedelta, timezone
import json
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


def render():
    import pandas as pd
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
    frame = pd.DataFrame(summary["weeks"])
    st.bar_chart(frame.set_index("Week starting")[["Visits"]])
    st.caption("Weeks run Monday–Sunday. The first and latest weeks may cover fewer than seven days; compare their coverage in the table.")
    with st.expander("Weekly counts and coverage"):
        st.dataframe(frame[["Week starting", "Recorded visits", "Recorded days", "Days in window", "Coverage"]], hide_index=True)

