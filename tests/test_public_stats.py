"""Public aggregates must be private, honest about gaps, and durable across rotation."""
from datetime import datetime, timezone
import json
import pytest
from streamlit.testing.v1 import AppTest
from gui.public_stats import summarize, validate
from tools.collect_public_stats import build_snapshot

NOW = datetime(2026, 10, 9, 1, tzinfo=timezone.utc)


def row(day="2026-10-01", **kw):
    return dict(day=day, ip="192.0.2.1", path="/math_probability?private=query",
                bot=False, probe=False, method="GET", status="200", **kw)


def test_counts_filter_noise_and_publish_only_dates_and_numbers():
    base = row()
    excluded = [dict(base, **patch) for patch in (
        {"bot": True}, {"probe": True}, {"status": "429"}, {"status": "404"},
        {"path": "/.env"}, {"path": "/_stcore/stream", "status": "101"},
        {"method": "POST"}, {"ip": "127.0.0.1"}, {"day": "2026-10-09"})]
    data = build_snapshot([base, dict(base, status="304")]+excluded,
        {"days": {"2026-10-01", "2026-10-02"}}, NOW, own_ips={"127.0.0.1"})
    assert data["daily"][0]["visits"] == 2
    assert data["daily"][1]["visits"] == 0
    assert data["daily"][2]["visits"] is None
    text = json.dumps(data)
    assert not any(secret in text for secret in ("192.0.2.1", "private=query", "path", "status", "probe"))
    summary = summarize(data)
    assert summary["total"] == 2 and summary["daily_average"] == 1
    assert summary["known_days"] == 2 and summary["missing_days"] == 6
    assert summary["weeks"][0]["Visits"] is None


def test_week_boundaries_and_average_use_completed_recorded_days():
    days = {f"2026-10-{i:02}" for i in range(1, 9)}
    data = build_snapshot([row(day=d) for d in days], {"days": days}, NOW)
    summary = summarize(data)
    assert summary["total"] == 8 and summary["daily_average"] == 1
    assert [(w["Week starting"], w["Visits"], w["Days in window"]) for w in summary["weeks"]] == [
        ("2026-09-28", 4, 4), ("2026-10-05", 4, 4)]
    assert all(w["Coverage"] == "Partial week" for w in summary["weeks"])


def test_rotating_logs_cannot_erase_published_daily_totals():
    old = build_snapshot([row(), row()], {"days": {"2026-10-01"}}, NOW)
    fresh = build_snapshot([row("2026-10-08")], {"days": {"2026-10-08"}}, NOW, old)
    assert fresh["daily"][0]["visits"] == 2
    partial = build_snapshot([row()], {"days": {"2026-10-01"}}, NOW, fresh)
    assert partial["daily"][0]["visits"] == 2
    assert partial["daily"][-1]["visits"] == 1
    with pytest.raises(ValueError, match="retaining"):
        build_snapshot([], {"errors": ["read failure"]}, NOW, old)
    assert build_snapshot([], {"unparsed": 1}, NOW, old)["daily"] == old["daily"]


def test_no_history_is_unavailable_and_invalid_counts_are_rejected():
    data = build_snapshot([], {}, NOW)
    assert summarize(data)["total"] is None
    assert summarize(data)["daily_average"] is None
    for bad in (-1, True, "12"):
        data["daily"][0]["visits"] = bad
        with pytest.raises(ValueError):
            validate(data)


def test_public_statistics_render_snapshot_and_missing_file(tmp_path, monkeypatch):
    file = tmp_path/"public-stats.json"
    monkeypatch.setenv("AILAB_PUBLIC_STATS_FILE", str(file))
    file.write_text(json.dumps(build_snapshot([row()], {"days": {"2026-10-01"}}, NOW)))
    at = AppTest.from_string('from gui.public_stats import render\nrender()').run()
    assert not at.exception
    assert [m.value for m in at.metric] == ["1", "1.0"]
    assert any("unavailable history" in w.value for w in at.warning)
    monkeypatch.setenv("AILAB_PUBLIC_STATS_FILE", str(tmp_path/"absent.json"))
    at.run()
    assert not at.exception and not at.metric
    assert any("not available yet" in i.value for i in at.info)
