"""Validated learner data. Persisted in this browser; processed in the app session.

No learner records are written to the server. JSON export is the portable backup.
"""
from __future__ import annotations

import copy
import json
from uuid import uuid4
from datetime import datetime, timezone

MAX_BYTES = 500_000
STAGES = ("Not started", "Explored", "Explained", "Rebuilt")


def empty():
    return {"version": 1, "topics": {}, "runs": [], "practice": {}, "last_topic": None}


def validate(raw):
    if isinstance(raw, (str, bytes)):
        if len(raw) > MAX_BYTES:
            raise ValueError("Learning file exceeds 500 KB.")
        try:
            raw = json.loads(raw)
        except (ValueError, UnicodeError) as exc:
            raise ValueError("Choose an AI Lab JSON export.") from exc
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Unsupported learning file version.")
    out = empty()
    from gui.guided_content import valid_practice_key, PRACTICE_KEYS
    practice = raw.get("practice", {})
    if not isinstance(practice, dict) or len(practice) > len(PRACTICE_KEYS):
        raise ValueError("Invalid guided practice records.")
    for key, record in practice.items():
        if not valid_practice_key(key):
            continue
        if not isinstance(record, dict):
            raise ValueError("Invalid guided practice record.")
        attempts, hints = record.get("attempts", 0), record.get("hints", 0)
        if type(attempts) is not int or not 0 <= attempts <= 10000 or type(hints) is not int or not 0 <= hints <= 2:
            raise ValueError("Invalid practice counts.")
        if any(type(record.get(k, False)) is not bool for k in ("solved", "independent", "revealed")):
            raise ValueError("Invalid practice result.")
        out["practice"][key] = {"attempts": attempts, "hints": hints,
            "solved": record.get("solved", False) or record.get("independent", False),
            "independent": record.get("independent", False), "revealed": record.get("revealed", False),
            "updated": str(record.get("updated", ""))[:40]}
    from gui.curriculum import CATALOG
    topics = raw.get("topics", {})
    if not isinstance(topics, dict) or len(topics) > 100:
        raise ValueError("Invalid topic records.")
    for key, entry in topics.items():
        if key not in CATALOG or not isinstance(entry, dict):
            continue
        stage = entry.get("stage", "Not started")
        if stage not in STAGES:
            raise ValueError("Unknown learning stage.")
        out["topics"][key] = {
            "stage": stage, "bookmark": bool(entry.get("bookmark", False)),
            "note": str(entry.get("note", ""))[:4000],
            "updated": str(entry.get("updated", ""))[:40],
        }
    if raw.get("last_topic") in CATALOG:
        out["last_topic"] = raw["last_topic"]
    runs = raw.get("runs", [])
    if not isinstance(runs, list) or len(runs) > 100:
        raise ValueError("Keep at most 100 notebook entries.")
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("Invalid notebook entry.")
        out["runs"].append({key: str(run.get(key, ""))[:4000] for key in
                            ("id", "title", "lesson", "seed", "settings", "prediction", "result", "explanation", "created", "prediction_saved_at", "completed_at")})
    if len(json.dumps(out).encode()) > MAX_BYTES:
        raise ValueError("Learning file exceeds 500 KB.")
    return out


def record_practice(data, key, action, *, correct=False):
    """Hints and reveals persist, so reloading cannot turn an assisted solve into an independent one."""
    from gui.guided_content import valid_practice_key
    if not valid_practice_key(key) or action not in ("check", "hint", "reveal"):
        raise ValueError("Unknown practice action.")
    updated = validate(data)
    record = updated["practice"].setdefault(key, {"attempts": 0, "hints": 0,
        "solved": False, "independent": False, "revealed": False})
    if action == "hint":
        record["hints"] = min(2, record["hints"] + 1)
    elif action == "reveal":
        record["revealed"] = True
    else:
        record["attempts"] = min(10000, record["attempts"] + 1)
        record["solved"] |= bool(correct)
        record["independent"] |= bool(correct) and not record["hints"] and not record["revealed"]
    record["updated"] = datetime.now(timezone.utc).isoformat()
    return validate(updated)


def merge_practice(first, second):
    """Progress and help exposure are cumulative across browser tabs and imports."""
    merged = copy.deepcopy(first)
    for key, other in second.items():
        current = merged.setdefault(key, {})
        for field in ("solved", "independent", "revealed"):
            current[field] = bool(current.get(field) or other.get(field))
        for field in ("attempts", "hints"):
            current[field] = max(current.get(field, 0), other.get(field, 0))
        current["updated"] = max(current.get("updated", ""), other.get("updated", ""))
    return merged


def start_run(data, *, title, lesson, seed, settings, prediction):
    """Commit the original prediction; completing a run cannot rewrite it."""
    if not title.strip() or not prediction.strip():
        raise ValueError("Enter a title and prediction before running.")
    updated = copy.deepcopy(data)
    now = datetime.now(timezone.utc).isoformat()
    run = dict(id=uuid4().hex, title=title, lesson=lesson, seed=str(seed),
               settings=settings, prediction=prediction, result="", explanation="",
               created=now, prediction_saved_at=now, completed_at="")
    updated["runs"].append(run)
    return validate(updated), run["id"]


def finish_run(data, run_id, *, result, explanation):
    if not result.strip() or not explanation.strip():
        raise ValueError("Record both the result and your explanation.")
    updated = copy.deepcopy(data)
    run = next((r for r in updated["runs"] if r["id"] == run_id), None)
    if run is None:
        raise ValueError("Notebook entry not found.")
    run.update(result=result, explanation=explanation,
               completed_at=datetime.now(timezone.utc).isoformat())
    return validate(updated)


def state():
    import streamlit as st
    if "learning_data" not in st.session_state:
        st.session_state.learning_data = empty()
    return st.session_state.learning_data


def update_topic(key, *, stage, bookmark, note):
    data = copy.deepcopy(state())
    data["topics"][key] = {"stage": stage, "bookmark": bookmark, "note": note,
                            "updated": datetime.now(timezone.utc).isoformat()}
    data["last_topic"] = key
    replace(data)


def replace(data):
    import streamlit as st
    st.session_state.learning_data = validate(data)
    st.session_state.learning_revision = st.session_state.get("learning_revision", 0) + 1


def sync_browser():
    """Mount once at the END of app.py so this rerun's edits are persisted immediately."""
    import streamlit as st
    from gui.guided_content import PRACTICE_KEYS
    result = _storage_component()(
        key="learning_browser_storage", data={"ready": st.session_state.get("learning_loaded", False),
          "revision": st.session_state.get("learning_revision", 0), "value": state(),
          "practice_limit": len(PRACTICE_KEYS)},
        on_loaded_change=lambda: None, on_error_change=lambda: None, on_merged_change=lambda: None,
    )
    if not st.session_state.get("learning_loaded") and result.loaded is not None:
        try:
            browser_data = validate(result.loaded)
            # An early click must not disappear while storage initializes.
            if not st.session_state.get("learning_revision"):
                st.session_state.learning_data = browser_data
        except ValueError:
            st.warning("Saved learning data could not be read. Import a backup in My learning.")
        st.session_state.learning_loaded = True
        st.rerun()
    if result.merged and result.merged.get("revision") == st.session_state.get("learning_revision", 0):
        try:
            merged = validate(result.merged["value"])
            if merged != state():
                st.session_state.learning_data = merged
                st.rerun()
        except (ValueError, KeyError):
            pass
    if result.error:
        st.caption("Browser saving is unavailable. Keep this session open and export your learning file.")


_component = None


def _storage_component():
    global _component
    if _component is None:
        import streamlit.components.v2 as components
        _component = components.component("ailab_learning", js=r'''
export default function({data, setStateValue}) {
  const key = "ailab.learning.v1";
  try {
    if (!data.ready) {
      const raw = localStorage.getItem(key);
      const value = raw && raw.length <= 500000 ? JSON.parse(raw) :
        {version:1, topics:{}, runs:[], last_topic:null};
      setStateValue("loaded", value);
    } else {
      // A revision of zero is hydration, not an edit. Other tabs cannot erase newer work.
      if (data.revision > 0) {
        const raw = localStorage.getItem(key);
        let saved = raw && raw.length <= 500000 ? JSON.parse(raw) : null;
        const value = structuredClone(data.value);
        if (saved && saved.version === 1 && saved.topics && Array.isArray(saved.runs)) {
          for (const [topic, record] of Object.entries(saved.topics)) {
            if (!value.topics[topic] || String(record.updated || "") > String(value.topics[topic].updated || ""))
              value.topics[topic] = record;
          }
          const runs = new Map(saved.runs.map(run => [run.id, run]));
          const stamp = run => String(run.completed_at || run.prediction_saved_at || run.created || "");
          for (const run of value.runs) {
            const previous = runs.get(run.id);
            if (!previous || stamp(run) >= stamp(previous)) runs.set(run.id, run);
          }
          value.runs = Array.from(runs.values());
          value.practice = value.practice || {};
          for (const [key, other] of Object.entries(saved.practice || {})) {
            const current = value.practice[key] || {};
            for (const field of ["solved", "independent", "revealed"])
              current[field] = Boolean(current[field] || other[field]);
            for (const field of ["attempts", "hints"])
              current[field] = Math.max(current[field] || 0, other[field] || 0);
            current.updated = String(current.updated || "") > String(other.updated || "")
              ? current.updated : other.updated;
            value.practice[key] = current;
          }
        }
        const serialized = JSON.stringify(value);
        if (serialized.length > 500000 || value.runs.length > 100 || Object.keys(value.practice || {}).length > data.practice_limit) {
          setStateValue("error", true);
          return;
        }
        if (serialized !== raw) localStorage.setItem(key, serialized);
        if (serialized !== JSON.stringify(data.value))
          setStateValue("merged", {revision:data.revision, value});
      }
    }
  } catch (e) {
    setStateValue("error", true);
    if (!data.ready) setStateValue("loaded", {version:1, topics:{}, runs:[], last_topic:null});
  }
}
''')
    return _component
