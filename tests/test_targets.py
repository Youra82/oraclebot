import numpy as np
import pandas as pd

from oraclebot.data.targets import TARGET_NAMES, compute_targets


def make_ohlcv(n=100, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def make_intraday_from_daily(daily_df: pd.DataFrame, bars_per_day: int = 6) -> pd.DataFrame:
    """Baut ein synthetisches 4h-artiges Intraday-DataFrame, das zu einem Daily-DataFrame passt.

    Deterministischer Pfad pro Tag: Open -> Tagestief (bar 1) -> Tageshoch (bar 3) -> Close.
    Tief und Hoch liegen dadurch garantiert in unterschiedlichen Sub-Kerzen (bar 1 vor bar 3,
    also "Tief zuerst" als Default -- Tests fuer high_first ueberschreiben die Bars gezielt).
    """
    assert bars_per_day >= 6, "Pfad Open->Low->High->Close braucht mindestens 6 Sub-Kerzen"
    rows = []
    idx = []
    for day_start, row in daily_df.iterrows():
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        waypoints = [o, l, l, h, h, c] + [c] * (bars_per_day - 6)
        for j in range(bars_per_day):
            idx.append(day_start + pd.Timedelta(hours=4 * j))
            seg_open = waypoints[j]
            seg_close = waypoints[j + 1] if j + 1 < len(waypoints) else c
            rows.append({
                'open': seg_open, 'close': seg_close,
                'high': max(seg_open, seg_close), 'low': min(seg_open, seg_close),
                'volume': row['volume'] / bars_per_day,
            })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def test_compute_targets_returns_expected_columns():
    df = make_ohlcv()
    intraday = make_intraday_from_daily(df)
    targets = compute_targets(df, intraday)
    for col in TARGET_NAMES:
        assert col in targets.columns


def test_target_categories_within_valid_ranges():
    df = make_ohlcv()
    intraday = make_intraday_from_daily(df)
    targets = compute_targets(df, intraday)
    assert targets['trend'].isin([0, 1, 2]).all()
    assert targets['range'].isin([0, 1, 2, 3]).all()
    assert targets['close_position'].isin([0, 1, 2]).all()
    assert targets['upper_wick'].isin([0, 1, 2]).all()
    assert targets['lower_wick'].isin([0, 1, 2]).all()
    assert targets['gap_yn'].isin([0, 1]).all()
    assert targets['inside_outside_day'].isin([0, 1, 2]).all()
    assert targets['high_first'].isin([0, 1]).all()


def test_extreme_bullish_candle_is_labeled_bullish():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    close = pd.Series([100.0] * 19 + [130.0], index=idx)  # letzter Tag: +30% Sprung
    open_ = close.shift(1).fillna(100.0)
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.1
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.1
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    intraday = make_intraday_from_daily(df)

    targets = compute_targets(df, intraday)
    assert targets['trend'].iloc[-1] == 2  # bullish


def test_close_position_matches_manual_formula():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    df = pd.DataFrame({
        'open': [100.0] * 20,
        'high': [110.0] * 20,
        'low': [90.0] * 20,
        'close': [108.0] * 20,  # CP = (108-90)/(110-90) = 0.9 -> upper_third
        'volume': [500.0] * 20,
    }, index=idx)
    intraday = make_intraday_from_daily(df)
    targets = compute_targets(df, intraday)
    assert targets['close_position'].iloc[-1] == 2


def test_gap_detected_on_large_jump():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    close = pd.Series(100.0, index=idx)
    open_ = close.shift(1).fillna(100.0)
    open_.iloc[-1] = 150.0  # grosser Gap-Up am letzten Tag
    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.5
    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.5
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    intraday = make_intraday_from_daily(df)

    targets = compute_targets(df, intraday)
    assert targets['gap_yn'].iloc[-1] == 1


def test_inside_day_detected():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    high = pd.Series(110.0, index=idx)
    low = pd.Series(90.0, index=idx)
    high.iloc[-1] = 105.0  # letzter Tag: enger im Vortagesrange -> inside day
    low.iloc[-1] = 95.0
    open_ = pd.Series(100.0, index=idx)
    close = pd.Series(100.0, index=idx)
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    intraday = make_intraday_from_daily(df)

    targets = compute_targets(df, intraday)
    assert targets['inside_outside_day'].iloc[-1] == 1


def test_outside_day_detected():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    high = pd.Series(105.0, index=idx)
    low = pd.Series(95.0, index=idx)
    high.iloc[-1] = 115.0  # letzter Tag: weiter als Vortag in beide Richtungen -> outside day
    low.iloc[-1] = 85.0
    open_ = pd.Series(100.0, index=idx)
    close = pd.Series(100.0, index=idx)
    volume = pd.Series(500.0, index=idx)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume})
    intraday = make_intraday_from_daily(df)

    targets = compute_targets(df, intraday)
    assert targets['inside_outside_day'].iloc[-1] == 2


def test_high_first_detected_from_intraday_order():
    idx = pd.date_range('2024-01-01', periods=20, freq='D', tz='UTC')
    df = pd.DataFrame({
        'open': [100.0] * 20, 'high': [110.0] * 20, 'low': [90.0] * 20,
        'close': [100.0] * 20, 'volume': [500.0] * 20,
    }, index=idx)

    # Intraday-Kerzen fuer den letzten Tag manuell so bauen, dass das Hoch VOR dem Tief kommt.
    intraday = make_intraday_from_daily(df)
    last_day = idx[-1]
    day_bars = [ts for ts in intraday.index if last_day <= ts < last_day + pd.Timedelta(days=1)]
    intraday.loc[day_bars[0], ['high', 'low']] = [110.0, 100.0]
    intraday.loc[day_bars[1], ['high', 'low']] = [100.0, 90.0]
    for ts in day_bars[2:]:
        intraday.loc[ts, ['high', 'low']] = [100.0, 100.0]

    targets = compute_targets(df, intraday)
    assert targets['high_first'].iloc[-1] == 1
