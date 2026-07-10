import numpy as np
import pandas as pd
import pytest

from oraclebot.data.features import FEATURE_NAMES, compute_features


def make_ohlcv(n=300, seed=42):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def test_compute_features_returns_expected_columns():
    df = make_ohlcv()
    feats = compute_features(df)
    for col in FEATURE_NAMES:
        assert col in feats.columns
    assert 'regime' in feats.columns


def test_compute_features_drops_warmup_nans():
    df = make_ohlcv(n=100)
    feats = compute_features(df, atr_window=14, ema_window=50, volume_window=20)
    assert not feats[FEATURE_NAMES].isna().any().any()
    assert len(feats) < len(df)


def test_body_and_wicks_sum_to_one_or_less():
    df = make_ohlcv()
    feats = compute_features(df)
    total = feats['body'] + feats['upper_wick'] + feats['lower_wick']
    # body + wicks darf die Gesamtrange nicht ueberschreiten (Rundungstoleranz)
    assert (total <= 1.0 + 1e-6).all()


def test_regime_is_one_of_known_labels():
    from oraclebot.data.features import REGIME_LABELS
    df = make_ohlcv()
    feats = compute_features(df)
    assert set(feats['regime'].unique()).issubset(set(REGIME_LABELS))


def test_dow_encoding_is_bounded():
    df = make_ohlcv()
    feats = compute_features(df)
    assert (feats['dow_sin'].abs() <= 1.0 + 1e-9).all()
    assert (feats['dow_cos'].abs() <= 1.0 + 1e-9).all()


def test_gap_matches_manual_formula():
    idx = pd.date_range('2024-01-01', periods=80, freq='D', tz='UTC')
    close = pd.Series(100.0, index=idx)
    open_ = close.shift(1).fillna(100.0)
    # Klarer Gap-Up am letzten Tag: Open weit ueber dem Vortages-Close
    open_.iloc[-1] = 130.0
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})

    feats = compute_features(df)
    # Gap = (Open_t - Close_t-1) / ATR; Close war konstant 100, Open sprang auf 130 -> Gap deutlich positiv
    assert feats['gap'].iloc[-1] > 1.0


def test_resistance_and_support_distance_are_nonnegative():
    df = make_ohlcv()
    feats = compute_features(df)
    assert (feats['resistance_distance'] >= 0).all()
    assert (feats['support_distance'] >= 0).all()


def test_macd_hist_present_and_finite():
    df = make_ohlcv()
    feats = compute_features(df)
    assert feats['macd_hist'].notna().all()
    assert np.isfinite(feats['macd_hist']).all()


def test_channel_position_and_slope_present_and_finite():
    df = make_ohlcv()
    feats = compute_features(df)
    assert feats['channel_position'].notna().all()
    assert np.isfinite(feats['channel_position']).all()
    assert feats['channel_slope'].notna().all()
    assert np.isfinite(feats['channel_slope']).all()


def test_channel_slope_is_positive_in_strong_uptrend():
    idx = pd.date_range('2024-01-01', periods=120, freq='D', tz='UTC')
    t = np.arange(120)
    # Klarer Aufwaertstrend mit Oszillation, damit Swing-Highs/-Lows entstehen.
    close = 100 + 0.8 * t + 3 * np.sin(2 * np.pi * t / 10)
    open_ = close - 0.3 * np.sin(2 * np.pi * t / 10)
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)

    feats = compute_features(df)
    assert feats['channel_slope'].iloc[-20:].mean() > 0


def test_swing_features_are_causal_no_lookahead():
    """Feature-Werte an Kerze t duerfen sich nicht aendern, wenn zukuenftige Kerzen (> t)
    aus dem DataFrame entfernt werden -- sonst haette die Berechnung an Kerze t Wissen ueber
    noch nicht abgeschlossene Kerzen benutzt (siehe _compute_swings()-Fix vom 2026-07-10:
    Swing-High/Low-Erkennung nutzt centered rolling() und muss ueber `confirmed_at` gegated
    werden, sonst lecken resistance_distance/support_distance/channel_*/structure Zukunftsdaten).
    """
    df = make_ohlcv(n=200)
    cutoff = 150
    feats_full = compute_features(df)
    feats_truncated = compute_features(df.iloc[:cutoff])

    common_idx = feats_full.index.intersection(feats_truncated.index)
    # Die letzten Kerzen vor dem Cutoff sind der kritische Fall: hier trat das Leck auf.
    check_idx = common_idx[common_idx <= df.index[cutoff - 1]][-10:]
    assert len(check_idx) > 0
    for col in ['structure', 'resistance_distance', 'support_distance', 'channel_position', 'channel_slope']:
        pd.testing.assert_series_equal(
            feats_full.loc[check_idx, col], feats_truncated.loc[check_idx, col],
            check_names=False, obj=col)


def test_channel_position_exceeds_one_on_breakout_above_channel():
    idx = pd.date_range('2024-01-01', periods=61, freq='D', tz='UTC')
    t = np.arange(61)
    # 60 Tage Seitwaerts-Kanal zwischen ~95 und ~105, dann ein klarer Ausbruch nach oben.
    close = 100 + 5 * np.sin(2 * np.pi * t / 10)
    close[-1] = 140.0  # deutlicher Ausbruch ueber alle bisherigen Swing-Highs
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)

    feats = compute_features(df)
    assert feats['channel_position'].iloc[-1] > 1.0
