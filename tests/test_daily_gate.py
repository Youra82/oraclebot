import pandas as pd

from oraclebot.utils.daily_gate import check_daily_gate, mark_daily_run_complete


def test_outside_window_before_midnight_skips(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 23:45', tz='UTC'), marker)
    assert not should_run
    assert 'Ausserhalb' in reason


def test_outside_window_after_30min_skips(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:30', tz='UTC'), marker)
    assert not should_run
    assert 'Ausserhalb' in reason


def test_first_tick_in_window_runs(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:00', tz='UTC'), marker)
    assert should_run
    assert reason is None


def test_second_tick_same_day_is_skipped_after_marking(tmp_path):
    """Reproduziert exakt den 2026-07-14-Bug: zwei */15-Cron-Ticks (:00 und :15) liegen beide
    im 00:00-00:29-Fenster -- nur der erste darf tatsaechlich eine Prognose senden."""
    marker = str(tmp_path / 'marker.txt')
    first_tick = pd.Timestamp('2026-07-14 00:00', tz='UTC')
    should_run_1, reason_1 = check_daily_gate(first_tick, marker)
    assert should_run_1
    mark_daily_run_complete(first_tick, marker)

    second_tick = pd.Timestamp('2026-07-14 00:15', tz='UTC')
    should_run_2, reason_2 = check_daily_gate(second_tick, marker)
    assert not should_run_2
    assert 'bereits gesendet' in reason_2


def test_third_tick_same_day_still_skipped(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    mark_daily_run_complete(pd.Timestamp('2026-07-14 00:00', tz='UTC'), marker)
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:29', tz='UTC'), marker)
    assert not should_run
    assert 'bereits gesendet' in reason


def test_next_day_runs_again(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    mark_daily_run_complete(pd.Timestamp('2026-07-14 00:05', tz='UTC'), marker)

    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-15 00:05', tz='UTC'), marker)
    assert should_run
    assert reason is None


def test_no_marker_file_yet_runs(tmp_path):
    marker = str(tmp_path / 'does_not_exist_yet.txt')
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:10', tz='UTC'), marker)
    assert should_run
    assert reason is None
