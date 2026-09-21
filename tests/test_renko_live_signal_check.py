import numpy as np
import pandas as pd
import pytest

from oraclebot.analysis.renko_live_signal_check import check_brick_chain_consistency
from oraclebot.data.ear_bricks import build_ear_bricks

BASE_PCT, K_ENTROPY, H_WINDOW = 0.01, 0.5, 5


def make_ohlcv(closes, start='2024-01-01', freq='5min'):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz='UTC')
    return pd.DataFrame({'open': closes, 'high': closes + 0.05, 'low': closes - 0.05,
                          'close': closes}, index=idx)


def test_consistent_state_reports_match(monkeypatch):
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 12, 150)) * 0.6)
    df = make_ohlcv(closes)
    bricks = build_ear_bricks(df, base_pct=BASE_PCT, k_entropy=K_ENTROPY, h_window=H_WINDOW)

    live_state = {'recent_bricks': [{'ts': str(b['ts']), 'direction': b['direction'],
                                       'close': b['close']} for b in bricks[-10:]]}

    import oraclebot.analysis.renko_live_signal_check as mod
    monkeypatch.setattr(mod, 'fetch_ohlcv', lambda *a, **k: df)

    result = check_brick_chain_consistency('FAKE/USDT:USDT', live_state, BASE_PCT, K_ENTROPY,
                                            H_WINDOW, '5m', lookback_days=1)
    assert result['match'] is True
    assert result['n_compared'] > 0


def test_diverged_state_reports_mismatch(monkeypatch):
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 12, 150)) * 0.6)
    df = make_ohlcv(closes)
    bricks = build_ear_bricks(df, base_pct=BASE_PCT, k_entropy=K_ENTROPY, h_window=H_WINDOW)

    # Live-Zustand mit vertauschter Richtung auf den letzten Bricks simulieren (kuenstlicher
    # Divergenz-Fall, wie er bei einem kaputten Kontinuitaets-Bug entstehen wuerde).
    corrupted = [{'ts': str(b['ts']), 'direction': ('down' if b['direction'] == 'up' else 'up'),
                  'close': b['close']} for b in bricks[-10:]]

    import oraclebot.analysis.renko_live_signal_check as mod
    monkeypatch.setattr(mod, 'fetch_ohlcv', lambda *a, **k: df)

    result = check_brick_chain_consistency('FAKE/USDT:USDT', {'recent_bricks': corrupted},
                                            BASE_PCT, K_ENTROPY, H_WINDOW, '5m', lookback_days=1)
    assert result['match'] is False
    assert len(result['mismatches']) > 0


def test_no_live_state_returns_none_match():
    result = check_brick_chain_consistency('FAKE/USDT:USDT', {}, BASE_PCT, K_ENTROPY, H_WINDOW, '5m')
    assert result['match'] is None
