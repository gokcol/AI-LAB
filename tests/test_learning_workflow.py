"""Portable learning data must remain bounded, valid and independent per browser."""
import json
from pathlib import Path
import pytest
from gui import curriculum, progress


def test_learning_export_roundtrip_and_unknown_topics():
    data = progress.empty()
    data['topics'] = {'playground': {'stage':'Explained','bookmark':True,'note':'bias shifts the threshold'},
                      'unknown': {'stage':'Explained'}}
    data['last_topic']='playground'
    out = progress.validate(json.dumps(data))
    assert out['topics']['playground']['note']=='bias shifts the threshold'
    assert out['topics']['playground']['bookmark']
    assert 'unknown' not in out['topics']
    assert progress.validate(json.dumps(out))==out


@pytest.mark.parametrize('raw', ['not json', '[]', '{"version":9}', b'\xff',
    {'version':1,'topics':[]}, {'version':1,'topics':{'playground':{'stage':'passed'}}},
    {'version':1,'runs':[{}]*101}, 'x'*500001])
def test_import_rejects_invalid_or_oversized_records(raw):
    with pytest.raises(ValueError):
        progress.validate(raw)


def test_curriculum_links_and_acyclic_prerequisites():
    root=Path(__file__).resolve().parents[1]/'gui'
    def visit(key, ancestors):
        assert key not in ancestors
        topic=curriculum.CATALOG[key]
        assert (root/topic.path).is_file()
        for pre in topic.prerequisites:
            visit(pre, ancestors|{key})
    for key in curriculum.CATALOG:
        visit(key,set())
    assert curriculum.next_topic('playground').key=='two_neurons'
    assert curriculum.next_topic('math_numerical') is None


def test_prediction_commit_completion_and_legacy_import():
    before = progress.empty()
    recorded, run_id = progress.start_run(before, title='Causal block', lesson='attention',
        seed=7, settings='heads=2', prediction='Earlier tokens remain unchanged')
    assert before['runs'] == []
    original = recorded['runs'][0].copy()
    assert original['prediction_saved_at'] and not original['result']
    completed = progress.finish_run(recorded, run_id, result='difference=0', explanation='Causal mask excludes the future')
    assert recorded['runs'][0] == original  # no mutation of an earlier snapshot
    for field in ('id', 'seed', 'settings', 'prediction', 'prediction_saved_at'):
        assert completed['runs'][0][field] == original[field]
    assert completed['runs'][0]['completed_at'] >= original['prediction_saved_at']
    assert progress.validate(json.dumps(completed)) == completed
    legacy = progress.empty()
    legacy['runs'] = [{'id':'old', 'prediction':'Legacy entry', 'result':'Already observed'}]
    restored = progress.validate(legacy)['runs'][0]
    assert restored['prediction_saved_at'] == ''  # never claim legacy entries were preregistered
    assert restored['result'] == 'Already observed'
    with pytest.raises(ValueError):
        progress.start_run(before, title='Test', lesson='attention', seed=1, settings='', prediction=' ')
    with pytest.raises(ValueError):
        progress.finish_run(recorded, run_id, result='', explanation='missing observation')


def test_lesson_bridges_resolve_to_executable_experiments():
    from gui.lab import all_experiments
    available = {experiment.id for experiment in all_experiments()}
    assert all(set(ids) <= available for ids in curriculum.EXPERIMENT_BRIDGES.values())
    assert all(topic.reviewed is None or topic.reviewed <= topic.updated for topic in curriculum.TOPICS)
