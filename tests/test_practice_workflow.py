"""Behavioral checks for the two-stage notebook and challenge controls."""
from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest
from streamlit.elements.widgets.button import ButtonMixin

ROOT = Path(__file__).resolve().parents[1]


def button(at, label):
    return next(b for b in at.button if b.label == label)


@pytest.mark.parametrize('page', ['bridge_lab', 'debugging_lab', 'evaluation_lab'])
def test_prediction_precedes_execution_and_result(page, monkeypatch):
    monkeypatch.setattr(ButtonMixin, '_page_link', lambda self, *a, **k: None)
    at = AppTest.from_file(str(ROOT / 'gui/views' / f'{page}.py'), default_timeout=30).run()
    assert not at.exception
    assert not at.dataframe
    at.text_area[0].set_value('My independent prediction')
    button(at, 'Record prediction').click().run()
    assert not at.exception
    assert not at.dataframe  # committing does not run the experiment
    record = at.session_state['learning_data']['runs'][0].copy()
    assert record['prediction_saved_at'] and not record['completed_at']
    button(at, 'Run with these settings').click().run()
    assert not at.exception
    assert at.dataframe or page == 'debugging_lab' and at.code
    next(a for a in at.text_area if a.label.startswith('Explain the result')).set_value('The observed result changed my explanation.')
    button(at, 'Save result to notebook').click().run()
    assert not at.exception
    saved = at.session_state['learning_data']['runs'][0]
    assert saved['completed_at'] and saved['result']
    for field in ('id', 'prediction', 'seed', 'settings', 'prediction_saved_at'):
        assert saved[field] == record[field]
    # A parameter change must hide results and require another prediction.
    at.slider[0].set_value(3 if page == 'bridge_lab' else .1).run()
    assert not at.exception
    assert not at.dataframe
    assert not any(b.label == 'Run with these settings' for b in at.button)


def test_experiment_deep_link_is_allowlisted(monkeypatch):
    monkeypatch.setattr(ButtonMixin, '_page_link', lambda self, *a, **k: None)
    at = AppTest.from_file(str(ROOT / 'gui/views/experiments.py'))
    at.query_params['experiment'] = 'e20'
    at.run()
    assert not at.exception
    assert 'e20_transformer_block' in at.caption[0].value
    at.query_params['experiment'] = '../../etc/passwd'
    at.run()
    assert not at.exception
    assert 'e20_transformer_block' in at.caption[0].value
