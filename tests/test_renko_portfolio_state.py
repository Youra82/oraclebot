import numpy as np
import pandas as pd
import pytest

from oraclebot.data.ear_bricks import build_ear_bricks
from oraclebot.strategy.horizontal_breakout_signal import backtest_horizontal_breakout
from oraclebot.strategy.renko_portfolio_state import (detect_exit, detect_fresh_entry,
                                                        update_symbol_bricks)

BASE_PCT, K_ENTROPY, H_WINDOW = 0.01, 0.5, 5
LOOKBACK, RUN = 4, 2


def make_ohlcv(closes, start='2024-01-01', freq='5min'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz='UTC')
    return pd.DataFrame({'open': closes, 'high': closes + 0.05, 'low': closes - 0.05,
                          'close': closes}, index=idx)


def _run_incremental(df, batch_size):
    """Simuliert mehrere Cron-Laeufe: verarbeitet `df` in Haeppchen von `batch_size` Kerzen und
    haengt die dabei entstehenden frischen Bricks zusammen."""
    state = {}
    all_fresh = []
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start + batch_size]
        state, fresh = update_symbol_bricks(state, batch, BASE_PCT, K_ENTROPY, H_WINDOW)
        all_fresh.extend(fresh)
    return state, all_fresh


def test_incremental_batches_match_one_shot_build():
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 12, 150)) * 0.6)
    df = make_ohlcv(closes)

    full_bricks = build_ear_bricks(df, base_pct=BASE_PCT, k_entropy=K_ENTROPY, h_window=H_WINDOW)

    for batch_size in (1, 3, 7, 25):
        _, fresh = _run_incremental(df, batch_size)
        fresh_closes = [b['close'] for b in fresh]
        full_closes = [b['close'] for b in full_bricks]
        assert len(fresh_closes) == len(full_closes), f"batch_size={batch_size}: Brick-Anzahl weicht ab"
        # Gleiche Bricks bis auf Gleitkomma-Rauschen (rolling().mean() summiert je nach
        # Fenstergroesse/-anzahl der Aufrufe in leicht anderer Reihenfolge -- exakte
        # Bit-Gleichheit ist hier nicht das Kriterium, nur dass keine STRUKTURELLE Abweichung
        # entsteht, siehe Moduldoc-Begruendung fuer den Anker-Fix).
        assert np.allclose(fresh_closes, full_closes, rtol=1e-6), f"batch_size={batch_size} weicht ab"


def test_fresh_entry_detection_matches_backtest_entries():
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 12, 150)) * 0.6)
    df = make_ohlcv(closes)
    full_bricks = build_ear_bricks(df, base_pct=BASE_PCT, k_entropy=K_ENTROPY, h_window=H_WINDOW)
    expected_trades = backtest_horizontal_breakout(full_bricks, LOOKBACK, RUN)
    expected_entry_prices = sorted(round(t['entry_price'], 6) for t in expected_trades)

    # Live-Simulation: Kerze fuer Kerze, Portfolio-Arbitrierung ueber ein einzelnes Symbol
    # (kein anderes Symbol konkurriert) -- jedes erkannte Entry-Signal wird sofort "ausgefuehrt"
    # (in_position=True), Exit beim ersten Gegen-Brick, danach wieder auf Entry-Suche.
    state = {}
    in_position = False
    position_direction = None
    found_entries = []
    for i in range(len(df)):
        candle = df.iloc[[i]]
        state, fresh = update_symbol_bricks(state, candle, BASE_PCT, K_ENTROPY, H_WINDOW)
        n_fresh = len(fresh)
        if n_fresh == 0:
            continue
        recent = state['recent_bricks']
        if in_position:
            exit_sig = detect_exit(recent, n_fresh, position_direction)
            if exit_sig is not None:
                in_position = False
                position_direction = None
        else:
            entry_sig = detect_fresh_entry(recent, n_fresh, LOOKBACK, RUN)
            if entry_sig is not None:
                found_entries.append(round(entry_sig['entry_price'], 6))
                in_position = True
                position_direction = entry_sig['direction']

    assert sorted(found_entries) == expected_entry_prices


def test_detect_fresh_entry_ignores_stale_signal_not_from_this_run():
    directions_prefix = ['up', 'down', 'up', 'down']  # horizontal, LOOKBACK=4
    bricks = []
    close = 100.0
    idx = pd.date_range('2024-01-01', periods=10, freq='5min', tz='UTC')
    for ts, d in zip(idx[:4], directions_prefix):
        close = close + 1 if d == 'up' else close - 1
        bricks.append({'ts': ts, 'direction': d, 'close': close})
    for ts, d in zip(idx[4:6], ['up', 'up']):
        close += 1
        bricks.append({'ts': ts, 'direction': d, 'close': close})
    # Der Ausbruch (letzte 2 Bricks) ist bereits eine Runde alt (n_fresh=0 fuer diesen Aufruf).
    assert detect_fresh_entry(bricks, n_fresh=0, horizontal_lookback=4, breakout_run=2) is None


def test_detect_exit_fires_on_first_opposite_fresh_brick():
    recent = [{'ts': '1', 'direction': 'up', 'close': 100.0},
              {'ts': '2', 'direction': 'down', 'close': 99.0},
              {'ts': '3', 'direction': 'down', 'close': 98.0}]
    exit_sig = detect_exit(recent, n_fresh=2, position_direction='long')
    assert exit_sig is not None
    assert exit_sig['exit_price'] == 99.0  # der ERSTE frische Gegen-Brick, nicht der zweite


def test_detect_exit_none_when_fresh_bricks_still_same_direction():
    recent = [{'ts': '1', 'direction': 'up', 'close': 100.0},
              {'ts': '2', 'direction': 'up', 'close': 101.0}]
    assert detect_exit(recent, n_fresh=1, position_direction='long') is None
