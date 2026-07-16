import numpy as np
import pandas as pd
import pytest

from oraclebot.analysis.hourly_volatility import (compute_hourly_path_profile,
                                                    compute_hourly_volatility_profile,
                                                    load_path_profile, load_profile,
                                                    save_path_profile, save_profile)


def make_hourly_df(n_days=30, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n_days * 24, freq='h', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, len(idx)))
    open_ = close + rng.normal(0, 0.3, len(idx))
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, len(idx)))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, len(idx)))
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close}, index=idx)


def test_profile_has_all_24_hours():
    df = make_hourly_df()
    profile = compute_hourly_volatility_profile(df)
    assert list(profile.index) == list(range(24))


def test_profile_means_are_nonnegative():
    df = make_hourly_df()
    profile = compute_hourly_volatility_profile(df)
    assert (profile['mean'].dropna() >= 0).all()


def test_profile_count_matches_number_of_days():
    df = make_hourly_df(n_days=30)
    profile = compute_hourly_volatility_profile(df)
    # Jede volle Stunde kommt an (fast) allen 30 Tagen vor.
    assert (profile['count'] <= 30).all()
    assert profile['count'].max() == 30


def test_save_and_load_profile_roundtrip(tmp_path):
    df = make_hourly_df()
    profile = compute_hourly_volatility_profile(df)
    path = str(tmp_path / 'profile.json')
    save_profile(profile, path)
    loaded = load_profile(path)

    assert set(loaded.keys()).issubset(set(range(24)))
    for hour, values in loaded.items():
        assert values['mean'] == pytest.approx(profile.loc[hour, 'mean'])
        assert values['count'] == profile.loc[hour, 'count']


def test_zero_range_days_are_excluded_not_crashing():
    """Tage mit High==Low (Range 0) duerfen keine Division durch Null verursachen."""
    idx = pd.date_range('2024-01-01', periods=48, freq='h', tz='UTC')
    df = pd.DataFrame({'open': 100.0, 'high': 100.0, 'low': 100.0, 'close': 100.0}, index=idx)
    profile = compute_hourly_volatility_profile(df)
    assert profile['mean'].isna().all()


def make_trending_hourly_df(n_days=40, seed=0):
    """Konstruiert Tage mit einem klaren monotonen Stunden-Pfad: gerade Tage (Index) steigen
    gleichmaessig (bullisch), ungerade Tage fallen gleichmaessig (bearish) -- damit
    compute_hourly_path_profile() eine erwartbare, klar unterscheidbare Form liefern muss."""
    rng = np.random.default_rng(seed)
    rows = []
    idx_start = pd.Timestamp('2024-01-01', tz='UTC')
    for day in range(n_days):
        day_start = idx_start + pd.Timedelta(days=day)
        bullish = day % 2 == 0
        base = 100.0
        for hour in range(24):
            frac = hour / 23
            level = base + (frac if bullish else -frac) * 10 + rng.normal(0, 0.05)
            rows.append({
                'ts': day_start + pd.Timedelta(hours=hour),
                'open': level, 'high': level + 0.2, 'low': level - 0.2, 'close': level,
            })
    df = pd.DataFrame(rows).set_index('ts')
    return df


def test_path_profile_separates_bullish_and_bearish_direction():
    df = make_trending_hourly_df()
    path = compute_hourly_path_profile(df)
    assert set(path.keys()) == {0, 1}
    # Bullische Tage: Position innerhalb der Tagesspanne soll ueber den Tag steigen.
    assert path[1][0]['mean'] < path[1][23]['mean']
    # Baerische Tage: soll fallen.
    assert path[0][0]['mean'] > path[0][23]['mean']


def test_path_profile_fractions_within_bounds():
    df = make_trending_hourly_df()
    path = compute_hourly_path_profile(df)
    for hours in path.values():
        for values in hours.values():
            assert -0.01 <= values['mean'] <= 1.01


def test_save_and_load_path_profile_roundtrip(tmp_path):
    df = make_trending_hourly_df()
    path = compute_hourly_path_profile(df)
    out_path = str(tmp_path / 'path_profile.json')
    save_path_profile(path, out_path)
    loaded = load_path_profile(out_path)

    assert set(loaded.keys()) == set(path.keys())
    for trend, hours in path.items():
        for hour, values in hours.items():
            assert loaded[trend][hour]['mean'] == pytest.approx(values['mean'])
            assert loaded[trend][hour]['count'] == values['count']
