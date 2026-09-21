import numpy as np
import pandas as pd
import pytest

from oraclebot.data.ear_bricks import build_ear_bricks, candle_entropy


def make_ohlcv(closes, start='2024-01-01', freq='15min'):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz='UTC')
    closes = np.array(closes, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.01
    lows = np.minimum(opens, closes) - 0.01
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes,
                          'volume': [10.0] * len(closes)}, index=idx)


def test_candle_entropy_is_low_for_directional_candle():
    # Schluss nahe am Hoch -> klar gerichtet -> niedrige Entropie
    h = candle_entropy(np.array([100.0]), np.array([110.0]), np.array([100.0]), np.array([109.5]))
    assert h[0] < 0.3


def test_candle_entropy_is_high_for_indecisive_candle():
    # Schluss mittig -> unentschlossen -> hohe Entropie (nahe 1)
    h = candle_entropy(np.array([100.0]), np.array([110.0]), np.array([100.0]), np.array([105.0]))
    assert h[0] > 0.9


def test_steady_uptrend_produces_only_up_bricks():
    closes = 100 + np.arange(50) * 0.5  # stetiger, gerader Anstieg
    df = make_ohlcv(closes)
    bricks = build_ear_bricks(df, base_pct=0.01, k_entropy=0.0, h_window=5)
    assert len(bricks) > 0
    assert all(b['direction'] == 'up' for b in bricks)
    # Brick-Schluesse steigen monoton
    closes_seq = [b['close'] for b in bricks]
    assert closes_seq == sorted(closes_seq)


def test_reversal_requires_double_brick_size():
    # Anstieg bis klar ueber einen Brick hinaus, dann ein Ruecksetzer, der NUR die einfache
    # Brick-Groesse unterschreitet (nicht die doppelte) -- darf NICHT umkehren.
    closes = [100.0, 102.0, 101.3]  # +2% (1 Brick bei 1%), dann -0.7% vom Hoch (< 2x Brick)
    df = make_ohlcv(closes)
    bricks = build_ear_bricks(df, base_pct=0.01, k_entropy=0.0, h_window=5)
    assert all(b['direction'] == 'up' for b in bricks)


def test_reversal_flips_after_double_brick_size_move():
    closes = [100.0, 102.0, 98.5]  # +2% hoch, dann > 2% (2x Brick) vom letzten Brick-Schluss runter
    df = make_ohlcv(closes)
    bricks = build_ear_bricks(df, base_pct=0.01, k_entropy=0.0, h_window=5)
    directions = [b['direction'] for b in bricks]
    assert 'down' in directions


def test_chaotic_market_produces_larger_bricks_than_calm_market():
    """Hoehere Entropie (k_entropy>0) muss zu GROESSEREN, also WENIGER Bricks fuer dieselbe
    Kursbewegung fuehren als bei k_entropy=0."""
    rng = np.random.default_rng(3)
    # Kerzen mit Schluss mittig in der Range -> hohe Entropie.
    n = 60
    base = 100 + np.cumsum(rng.normal(0.1, 0.3, n))
    idx = pd.date_range('2024-01-01', periods=n, freq='15min', tz='UTC')
    df = pd.DataFrame({
        'open': base, 'close': base + 0.05,
        'high': base + 2.0, 'low': base - 2.0,  # riesige Range, Schluss mittig -> H nahe 1
        'volume': [10.0] * n,
    }, index=idx)

    bricks_calm = build_ear_bricks(df, base_pct=0.005, k_entropy=0.0, h_window=5)
    bricks_chaos = build_ear_bricks(df, base_pct=0.005, k_entropy=2.0, h_window=5)
    assert len(bricks_chaos) <= len(bricks_calm)


def test_too_short_history_returns_empty():
    df = make_ohlcv([100.0])
    assert build_ear_bricks(df) == []


def test_incremental_continuation_with_precomputed_h_roll_matches_fresh_build():
    """Eine Kette, die in einem Rutsch gebaut wird, muss (fuer denselben Endzustand) dieselben
    neuen Bricks liefern wie eine Fortsetzung ab einem persistierten Zwischenstand -- Grundlage
    fuer einen kuenftigen Live-Betrieb (siehe zerobots eigene Live-vs-Backtest-Absicherung,
    [[research_zerobot_live_vs_backtest_2026_08]]). Nutzt `precomputed_H_roll`, damit die
    rollierende Entropie-Glaettung an der Nahtstelle exakt dem durchgaengigen Aufbau entspricht,
    OHNE die Puffer-Kerzen selbst nochmal durch die Brick-Konstruktion laufen zu lassen (das
    wuerde bereits erzeugte Bricks doppelt bauen) -- dasselbe Muster wie in zerobots ear_engine.py."""
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 10, 80)) * 0.5)
    df = make_ohlcv(closes)
    h_window = 5
    split = 40

    H_raw_full = candle_entropy(df['open'].to_numpy(), df['high'].to_numpy(),
                                 df['low'].to_numpy(), df['close'].to_numpy())
    H_roll_full = pd.Series(H_raw_full).rolling(h_window, min_periods=1).mean().to_numpy()

    full = build_ear_bricks(df, base_pct=0.01, k_entropy=0.5, h_window=h_window,
                             precomputed_H_roll=H_roll_full)
    first_half_bricks = [b for b in full if b['candle_idx'] < split]
    second_half_bricks_expected = [b for b in full if b['candle_idx'] >= split]

    if not first_half_bricks:
        pytest.skip("Testdaten erzeugten keine Bricks in der ersten Haelfte")

    last_brick = first_half_bricks[-1]
    # Nur die NEUEN Kerzen (ab split) werden verarbeitet, aber mit dem schon korrekt
    # geglaetteten H_roll-Ausschnitt aus dem durchgaengigen Aufbau.
    continued = build_ear_bricks(df.iloc[split:], base_pct=0.01, k_entropy=0.5, h_window=h_window,
                                  init_close=last_brick['close'], init_direction=last_brick['direction'],
                                  precomputed_H_roll=H_roll_full[split:])
    continued_closes = [round(b['close'], 6) for b in continued]
    expected_closes = [round(b['close'], 6) for b in second_half_bricks_expected]
    assert continued_closes == expected_closes
