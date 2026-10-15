"""Numerical answers, help-aware persistence, and track-specific learning flows."""
import json
from pathlib import Path
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.elements.widgets.button import ButtonMixin
from gui import guided_content as content, progress

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("variant", range(5))
def test_exercise_numbers_against_independent_calculations(variant):
    a = 2+variant
    dot = content.exercises("math_vectors", variant)
    assert dot[0].answer == np.dot([a, -1], [3, 4])
    assert dot[2].answer == np.dot([a, -a], [3, 6])-np.dot([a, -a], [3, 4])
    assert dot[7].answer == np.dot([a, 1], [3, 4])+2
    grad = content.exercises("math_calculus", variant)
    loss = lambda w: (a*w-1)**2
    numerical = (loss(1+1e-5)-loss(1-1e-5))/2e-5
    assert grad[4].answer == pytest.approx(numerical)
    assert (a+1-.25*grad[1].answer) == grad[2].answer
    reg = content.exercises("ml_regression", variant)
    assert reg[2].answer == np.mean(np.square([-a, a+1]))
    assert reg[3].answer == np.mean([a, a+2, a+4])
    clf = content.exercises("ml_classification", variant)
    actual = np.array([1]*a+[0]+[1, 1])
    predicted = np.array([1]*(a+1)+[0, 0])
    assert clf[2].answer == np.mean(actual[predicted == 1])
    assert clf[3].answer == np.mean(predicted[actual == 1])
    for topic in content.LESSONS:
        questions = content.exercises(topic, variant)
        assert len(questions) == 8
        for q in questions:
            assert q.check(q.answer)
            if q.options:
                assert all(not q.check(other) for other in q.options if other != q.answer)
            else:
                assert q.check(str(round(q.answer, 3)))
                assert not q.check(float("nan"))
                assert not q.check(float("inf"))
                assert not q.check(q.answer+1)
                assert not q.check("not a number")


def test_help_reveal_retry_export_and_import_preserve_evidence():
    key = content.practice_key("math_vectors", 0, 0)
    data = progress.empty()
    data = progress.record_practice(data, key, "check", correct=False)
    assert not data["practice"][key]["solved"]
    data = progress.record_practice(data, key, "hint")
    restored = progress.validate(json.dumps(data))
    data = progress.record_practice(restored, key, "check", correct=True)
    assert data["practice"][key]["solved"] and not data["practice"][key]["independent"]
    fresh = content.practice_key("math_vectors", 0, 1)
    data = progress.record_practice(data, fresh, "check", correct=True)
    assert content.practice_summary("math_vectors", data["practice"]) == {"solved": 1, "independent": 1, "total": 8}
    revealed = content.practice_key("math_calculus", 0, 0)
    data = progress.record_practice(data, revealed, "reveal")
    assert not data["practice"][revealed]["solved"]
    data = progress.record_practice(data, revealed, "check", correct=True)
    assert not data["practice"][revealed]["independent"]
    assert not data["topics"]  # practice never changes self-assessed checkpoints
    assert progress.validate({"version":1, "topics":{}, "runs":[]})["practice"] == {}
    assert progress.validate(json.dumps(data)) == data


def test_merge_keeps_help_exposure_and_independent_success():
    key = content.practice_key("ml_regression", 0, 0)
    hints = progress.record_practice(progress.empty(), key, "hint")["practice"]
    solved = progress.record_practice(progress.empty(), key, "check", correct=True)["practice"]
    merged = progress.merge_practice(hints, solved)
    assert merged[key]["hints"] == 1
    assert merged[key]["independent"] and merged[key]["solved"]
    assert progress.merge_practice(solved, hints) == merged
    assert not hints[key]["solved"]  # source snapshots are not mutated


@pytest.mark.parametrize("record", [{"attempts":-1}, {"hints":3}, {"independent":"yes"}, []])
def test_reject_invalid_imported_practice(record):
    raw = progress.empty()
    raw["practice"]["math_vectors:0:0"] = record
    with pytest.raises(ValueError):
        progress.validate(raw)


def button(at, label):
    return next(b for b in at.button if b.label == label)


def test_checked_exercise_does_not_reveal_on_wrong_answer(monkeypatch):
    monkeypatch.setattr(ButtonMixin, "_page_link", lambda self, *a, **k: None)
    at = AppTest.from_string('from gui.guided import render_exercises\nrender_exercises("math_vectors")').run()
    at.text_input[0].set_value("10")
    button(at, "Check answer").click().run()
    assert not at.exception
    assert any("negative sign" in w.value for w in at.warning)
    assert not at.success
    key = "math_vectors:0:0"
    assert not at.session_state["learning_data"]["practice"][key]["solved"]
    button(at, "Show next hint").click().run()
    at.text_input[0].set_value("2")
    button(at, "Check answer").click().run()
    assert not at.exception
    saved = at.session_state["learning_data"]["practice"][key]
    assert saved["solved"] and not saved["independent"]
    at.selectbox(key="guided_lesson_math_vectors_variant").set_value(1).run()
    at.text_input[0].set_value("5")
    button(at, "Check answer").click().run()
    assert at.session_state["learning_data"]["practice"]["math_vectors:0:1"]["independent"]


@pytest.mark.parametrize("topic", list(content.LESSONS))
def test_guided_exploration_requires_prediction_and_can_save(topic, monkeypatch):
    monkeypatch.setattr(ButtonMixin, "_page_link", lambda self, *a, **k: None)
    at = AppTest.from_string(f'from gui.guided import render\nrender("{topic}")', default_timeout=30).run()
    at.segmented_control[0].set_value("Predict & explore").run()
    assert not at.exception and not at.dataframe and not at.metric
    at.text_area[0].set_value("I predict a lower error; I will compare the contributions.")
    button(at, "Record prediction").click().run()
    assert not at.dataframe and not at.metric
    button(at, "Run with these settings").click().run()
    assert not at.exception
    assert at.dataframe or at.metric
    next(x for x in at.text_area if x.label.startswith("Explain the result")).set_value("The calculated contributions explain this result.")
    button(at, "Save result to notebook").click().run()
    assert at.session_state["learning_data"]["runs"][0]["completed_at"]
    # New settings need a new prediction and cannot display the previous result.
    first = at.slider[0]
    first.set_value(first.value+first.step).run()
    assert not at.exception and not at.dataframe and not at.metric


def test_coach_follows_track_across_all_tabs_and_sidebar_changes(monkeypatch):
    monkeypatch.setattr(ButtonMixin, "_page_link", lambda self, *a, **k: None)
    at = AppTest.from_file(str(ROOT/"gui/views/study_coach.py"), default_timeout=30)
    at.session_state["module"] = "Math"
    at.run()
    assert not at.exception
    assert at.selectbox(key="coach_track").value == "Math"
    assert any("Dot products" in m.value for m in at.markdown)
    assert not any(s.label == "Question set" for s in at.segmented_control)
    data = progress.empty()
    for topic in ("math_vectors", "math_calculus"):
        for i in range(8):
            data = progress.record_practice(data, content.practice_key(topic, i, 0), "check", correct=True)
    at.session_state["learning_data"] = data
    at.run()
    assert not at.exception
    # Completing guided exercises advances the recommendation even without a self-rating.
    # Links are stubbed in this isolated harness; the recommended objective is visible.
    assert any("Reason about conditional probability and uncertainty." in m.value for m in at.markdown)
    at.selectbox(key="coach_track").set_value("ML").run()
    assert not at.exception
    assert any("Classification" in m.value for m in at.markdown)
    assert not any("Tier 0" in m.value for m in at.markdown)
    # An explicit sidebar change becomes the coach's starting track.
    at.session_state["module"] = "ANN"
    at.run()
    assert not at.exception
    assert at.selectbox(key="coach_track").value == "ANN"
    assert any(s.label == "Question set" for s in at.segmented_control)
