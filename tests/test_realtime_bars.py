from oraclebot.utils.realtime_bars import BarAggregator, bars_to_df


def test_ticks_within_same_window_update_high_low_close():
    agg = BarAggregator(bar_seconds=5)
    assert agg.add_tick(1000, 100.0) is None  # startet Fenster [1000ms .. 6000ms)
    assert agg.add_tick(1200, 101.0) is None
    assert agg.add_tick(1400, 99.0) is None
    finished = agg.add_tick(6000, 102.0)  # neues Fenster -> vorheriges wird geschlossen
    assert finished["open"] == 100.0
    assert finished["high"] == 101.0
    assert finished["low"] == 99.0
    assert finished["close"] == 99.0  # letzter Tick VOR dem neuen Fenster


def test_bar_boundaries_are_absolute_not_relative_to_first_tick():
    agg = BarAggregator(bar_seconds=5)
    agg.add_tick(3000, 100.0)  # faellt in Fenster [0..5000), nicht [3000..8000)
    finished = agg.add_tick(5000, 105.0)  # 5000 ist bereits das NAECHSTE Fenster
    assert finished is not None
    assert finished["ts"].value // 1_000_000 == 0  # Fenster startete bei 0ms


def test_late_tick_from_older_window_is_dropped():
    agg = BarAggregator(bar_seconds=5)
    agg.add_tick(6000, 100.0)
    result = agg.add_tick(1000, 999.0)  # aelteres Fenster, sollte verworfen werden
    assert result is None
    # Bestaetigen, dass der veraltete Tick den aktuellen Bar NICHT beeinflusst hat.
    finished = agg.add_tick(11000, 100.0)
    assert finished["low"] != 999.0


def test_flush_stale_bars_fills_gaps_with_flat_bars():
    agg = BarAggregator(bar_seconds=5)
    agg.add_tick(1000, 100.0)
    # 17 Sekunden spaeter, ohne neue Ticks dazwischen -> mehrere Fenster ueberfaellig.
    bars = agg.flush_stale_bars(now_ms=18000)
    assert len(bars) >= 2
    for b in bars:
        assert b["open"] == b["high"] == b["low"] == b["close"]


def test_bars_to_df_produces_expected_columns_and_index():
    agg = BarAggregator(bar_seconds=5)
    agg.add_tick(1000, 100.0)
    agg.add_tick(1500, 102.0)
    b1 = agg.add_tick(6000, 103.0)
    b2 = agg.add_tick(11000, 104.0)
    df = bars_to_df([b1, b2])
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert len(df) == 2
    assert df.index.tz is not None


def test_bars_to_df_empty_list_returns_empty_frame():
    df = bars_to_df([])
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close"]
