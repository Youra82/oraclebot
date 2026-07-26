import json
from datetime import datetime, timedelta, timezone

from oraclebot.utils import training_history


def make_cfg(min_confidence=0.60, model_max_depth=3, barrier_pct=1.0, context_timeframes=None):
    return {
        'min_confidence': min_confidence,
        'model_max_depth': model_max_depth,
        'barrier_pct': barrier_pct,
        'context_timeframes': context_timeframes if context_timeframes is not None else ['1h'],
    }


def write_raw_entry(history_path, cfg, timestamp, val_accuracy=0.7):
    entry = {'timestamp': timestamp.isoformat(), 'val_accuracy': val_accuracy,
              'walk_forward_mean': 0.7, 'walk_forward_worst_case': 0.65}
    entry.update(cfg)
    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def test_append_entry_and_load_history(tmp_path):
    history_path = str(tmp_path / 'history.jsonl')
    training_history.append_entry(history_path, make_cfg(), val_accuracy=0.75,
                                   walk_forward_mean=0.72, walk_forward_worst_case=0.68)
    entries = training_history.load_history(history_path)
    assert len(entries) == 1
    assert entries[0]['min_confidence'] == 0.60
    assert entries[0]['val_accuracy'] == 0.75


def test_load_history_missing_file_returns_empty(tmp_path):
    assert training_history.load_history(str(tmp_path / 'does_not_exist.jsonl')) == []


def test_no_warning_with_fewer_than_min_runs(tmp_path):
    history_path = str(tmp_path / 'history.jsonl')
    now = datetime.now(timezone.utc)
    write_raw_entry(history_path, make_cfg(min_confidence=0.60), now)
    write_raw_entry(history_path, make_cfg(min_confidence=0.70), now)
    assert training_history.check_overfitting_risk(history_path) is None


def test_no_warning_when_repeated_same_params(tmp_path):
    """Reines Retraining auf neuen Daten mit unveraenderten Parametern ist gesundes Verhalten."""
    history_path = str(tmp_path / 'history.jsonl')
    now = datetime.now(timezone.utc)
    for _ in range(5):
        write_raw_entry(history_path, make_cfg(min_confidence=0.60), now)
    assert training_history.check_overfitting_risk(history_path) is None


def test_warning_when_multiple_different_params_within_window(tmp_path):
    history_path = str(tmp_path / 'history.jsonl')
    now = datetime.now(timezone.utc)
    write_raw_entry(history_path, make_cfg(min_confidence=0.60), now)
    write_raw_entry(history_path, make_cfg(min_confidence=0.65), now)
    write_raw_entry(history_path, make_cfg(min_confidence=0.70), now)
    warning = training_history.check_overfitting_risk(history_path)
    assert warning is not None
    assert 'WARNUNG' in warning
    assert '3 Trainingslaeufe' in warning


def test_no_warning_when_differing_runs_are_outside_time_window(tmp_path):
    history_path = str(tmp_path / 'history.jsonl')
    old = datetime.now(timezone.utc) - timedelta(hours=48)
    write_raw_entry(history_path, make_cfg(min_confidence=0.60), old)
    write_raw_entry(history_path, make_cfg(min_confidence=0.65), old)
    write_raw_entry(history_path, make_cfg(min_confidence=0.70), old)
    assert training_history.check_overfitting_risk(history_path, window_hours=24) is None


def test_context_timeframes_list_is_compared_correctly(tmp_path):
    """context_timeframes ist eine Liste -- muss trotzdem korrekt (un)gleich verglichen werden."""
    history_path = str(tmp_path / 'history.jsonl')
    now = datetime.now(timezone.utc)
    write_raw_entry(history_path, make_cfg(context_timeframes=['1h', '4h']), now)
    write_raw_entry(history_path, make_cfg(context_timeframes=['1h', '4h']), now)
    write_raw_entry(history_path, make_cfg(context_timeframes=['1h']), now)
    warning = training_history.check_overfitting_risk(history_path)
    assert warning is not None
