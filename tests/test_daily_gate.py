import pandas as pd

from oraclebot.utils.daily_gate import check_daily_gate, mark_daily_run_complete, max_expected_staleness

# Testzeitpunkte liegen im Juli (CEST, UTC+2). Die Standard-Zielzeit "02:05" deutscher Zeit
# entspricht dort 00:05 UTC -- das macht die UTC-Zeitstempel unten direkt nachvollziehbar.


def test_outside_window_before_target_skips(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    # 23:45 UTC Vortag = 01:45 CEST -- vor der Standard-Zielzeit 02:05.
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-13 23:45', tz='UTC'), marker)
    assert not should_run
    assert 'Noch nicht so weit' in reason


def test_outside_window_after_30min_skips(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    # 00:40 UTC = 02:40 CEST -- 5 Min nach Fensterende (02:05 + 30min = 02:35).
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:40', tz='UTC'), marker)
    assert not should_run
    assert 'Ausserhalb' in reason


def test_first_tick_in_window_runs(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    # 00:05 UTC = 02:05 CEST -- exakt die Standard-Zielzeit.
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:05', tz='UTC'), marker)
    assert should_run
    assert reason is None


def test_second_tick_same_day_is_skipped_after_marking(tmp_path):
    """Reproduziert exakt den 2026-07-14-Bug: zwei */15-Cron-Ticks liegen beide im Fenster --
    nur der erste darf tatsaechlich eine Prognose senden."""
    marker = str(tmp_path / 'marker.txt')
    first_tick = pd.Timestamp('2026-07-14 00:05', tz='UTC')
    should_run_1, reason_1 = check_daily_gate(first_tick, marker)
    assert should_run_1
    mark_daily_run_complete(first_tick, marker)

    second_tick = pd.Timestamp('2026-07-14 00:20', tz='UTC')
    should_run_2, reason_2 = check_daily_gate(second_tick, marker)
    assert not should_run_2
    assert 'bereits gesendet' in reason_2


def test_third_tick_same_day_still_skipped(tmp_path):
    marker = str(tmp_path / 'marker.txt')
    mark_daily_run_complete(pd.Timestamp('2026-07-14 00:05', tz='UTC'), marker)
    should_run, reason = check_daily_gate(pd.Timestamp('2026-07-14 00:34', tz='UTC'), marker)
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


def test_custom_notification_time_is_respected(tmp_path):
    """Frei waehlbare deutsche Ortszeit, z.B. 08:00 -- kein eingebautes Limit."""
    marker = str(tmp_path / 'marker.txt')
    # 08:00 CEST = 06:00 UTC.
    should_run_early, _ = check_daily_gate(
        pd.Timestamp('2026-07-14 05:00', tz='UTC'), marker, notification_time_local='08:00')
    assert not should_run_early

    should_run, reason = check_daily_gate(
        pd.Timestamp('2026-07-14 06:00', tz='UTC'), marker, notification_time_local='08:00')
    assert should_run
    assert reason is None


def test_very_early_notification_time_is_allowed_no_restriction(tmp_path):
    """Keine erzwungene Untergrenze -- eine sehr frueh liegende Uhrzeit wird nicht blockiert,
    auch wenn sie (aus UTC-Sicht) nahe am eigentlichen Kerzenschluss liegt."""
    marker = str(tmp_path / 'marker.txt')
    should_run, reason = check_daily_gate(
        pd.Timestamp('2026-07-13 22:05', tz='UTC'), marker, notification_time_local='00:05')
    assert should_run
    assert reason is None


def test_max_expected_staleness_scales_with_notification_time():
    """Spaeter am Tag konfigurierte Zeiten erlauben eine entsprechend hoehere Staleness, bevor
    der Cache-Fehler-Alarm ausloest -- sonst wuerde predict_next_candle.py bei z.B. '20:00'
    jeden Tag faelschlich abbrechen."""
    now_utc = pd.Timestamp('2026-07-14 06:00', tz='UTC')
    early = max_expected_staleness(now_utc, '02:05')
    late = max_expected_staleness(now_utc, '20:00')
    assert late > early
    # 20:00 CEST = 18:00 UTC -> 24h Kerze + 18h Abstand + 6h Puffer = 48h.
    assert late == pd.Timedelta(hours=48)
