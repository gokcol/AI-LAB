"""A deployed update must announce itself and retain published history."""

import json
from pathlib import Path

import pytest

from tools.check_release_notes import validate


def test_public_notes_are_valid():
    validate(Path(__file__).resolve().parents[1] / "gui/content/releases.json")


def test_deployment_requires_new_entry_and_preserves_history(tmp_path):
    old = [{"version": "1.02", "date": "2026-09-20", "title": "Learning tools", "changes": ["Saved progress"]}]
    previous, current = tmp_path / "old.json", tmp_path / "new.json"
    previous.write_text(json.dumps(old))
    current.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="Prepend"):
        validate(current, previous)
    new = [{"version": "1.03", "date": "2026-09-20", "title": "What's new", "changes": ["Release history"]}] + old
    current.write_text(json.dumps(new))
    assert validate(current, previous) == new
    new[1]["title"] = "Rewritten history"
    current.write_text(json.dumps(new))
    with pytest.raises(ValueError, match="preserve"):
        validate(current, previous)


def test_deployment_rejects_reused_version(tmp_path):
    entry = {"version": "1.03", "date": "2026-09-20", "title": "Update", "changes": ["A fix"]}
    current = tmp_path / "new.json"
    current.write_text(json.dumps([entry, entry]))
    with pytest.raises(ValueError, match="unique"):
        validate(current)


def test_numbered_revision_keeps_version_and_preserves_history(tmp_path):
    old = [{"version": "1.14", "date": "2026-10-15", "title": "Foundations", "changes": ["Four guides"]}]
    previous, current = tmp_path/"old.json", tmp_path/"new.json"
    previous.write_text(json.dumps(old))
    update = dict(old[0], revision=2, title="More guides")
    current.write_text(json.dumps([update]+old))
    assert validate(current, previous)[0]["version"] == "1.14"
    current.write_text(json.dumps([update]))
    with pytest.raises(ValueError, match="preserve"):
        validate(current, previous)
    for bad in (0, True, "2"):
        current.write_text(json.dumps([dict(update, revision=bad)]+old))
        with pytest.raises(ValueError, match="revision"):
            validate(current)


def test_september_history_is_one_v113_section_without_lost_changes():
    from gui.release_history import grouped
    entries = json.loads((Path(__file__).resolve().parents[1]/"gui/content/releases.json").read_text())
    groups = grouped(entries)
    september = [g for g in groups if g["date"] == "2026-09-20"]
    assert len(september) == 1 and september[0]["version"] == "1.13"
    expected = {c for e in entries if e["date"] == "2026-09-20" for c in e["changes"]}
    assert set(september[0]["changes"]) == expected
    assert groups[0]["version"] == "1.14"


def test_explicit_redaction_preserves_other_history_and_requires_new_release(tmp_path):
    def entry(version):
        return {"version": version, "date": "2026-09-20", "title": "Update", "changes": ["A fix"]}

    old = [entry("1.03"), entry("1.02"), entry("1.01")]
    previous, current = tmp_path / "old.json", tmp_path / "new.json"
    previous.write_text(json.dumps(old))
    new = [entry("1.04"), old[0], old[2]]
    current.write_text(json.dumps(new))
    with pytest.raises(ValueError, match="preserve"):
        validate(current, previous)
    assert validate(current, previous, redact_versions=["1.02"]) == new
    with pytest.raises(ValueError, match="exist"):
        validate(current, previous, redact_versions=["1.09"])
    new[-1]["title"] = "Unrequested change"
    current.write_text(json.dumps(new))
    with pytest.raises(ValueError, match="preserve"):
        validate(current, previous, redact_versions=["1.02"])
    current.write_text(json.dumps(old[:1] + old[2:]))
    with pytest.raises(ValueError, match="Prepend"):
        validate(current, previous, redact_versions=["1.02"])


def test_release_date_accepts_today_in_another_timezone_but_not_future(tmp_path, monkeypatch):
    from tools import check_release_notes as notes
    import datetime as dt
    monkeypatch.setattr(notes, "latest_calendar_date", lambda: dt.date(2026, 10, 16))
    path = tmp_path / "notes.json"
    entry = {"version": "1.14", "date": "2026-10-16", "title": "Theme fixes", "changes": ["Readable plots"]}
    path.write_text(json.dumps([entry]))
    assert notes.validate(path)
    entry["date"] = "2026-10-17"
    path.write_text(json.dumps([entry]))
    with pytest.raises(ValueError, match="future"):
        notes.validate(path)
