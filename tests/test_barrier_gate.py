import pandas as pd

from oraclebot.utils.barrier_gate import check_barrier_gate, mark_barrier_run_complete


def test_outside_window_skips():
    should_run, reason = check_barrier_gate(pd.Timestamp('2026-07-14 05:45', tz='UTC'), 'unused')
    assert not should_run
    assert 'Ausserhalb' in reason


def test_hour_not_divisible_by_period_skips():
    should_run, reason = check_barrier_gate(pd.Timestamp('2026-07-14 06:05', tz='UTC'), 'unused')
    assert not should_run


def test_first_tick_at_4h_boundary_runs(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    should_run, reason = check_barrier_gate(pd.Timestamp('2026-07-14 08:00', tz='UTC'), marker)
    assert should_run
    assert reason is None


def test_second_tick_same_period_is_skipped_after_marking(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    first_tick = pd.Timestamp('2026-07-14 08:00', tz='UTC')
    should_run_1, _ = check_barrier_gate(first_tick, marker)
    assert should_run_1
    mark_barrier_run_complete(first_tick, marker)

    second_tick = pd.Timestamp('2026-07-14 08:15', tz='UTC')
    should_run_2, reason_2 = check_barrier_gate(second_tick, marker)
    assert not should_run_2
    assert 'bereits verarbeitet' in reason_2


def test_next_4h_period_runs_again(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    mark_barrier_run_complete(pd.Timestamp('2026-07-14 08:05', tz='UTC'), marker)

    should_run, reason = check_barrier_gate(pd.Timestamp('2026-07-14 12:05', tz='UTC'), marker)
    assert should_run
    assert reason is None


def test_no_marker_file_yet_runs(tmp_path):
    marker = str(tmp_path / 'does_not_exist_yet.txt')
    should_run, reason = check_barrier_gate(pd.Timestamp('2026-07-14 00:10', tz='UTC'), marker)
    assert should_run
    assert reason is None
