import numpy as np
import pandas as pd
import pytest

from oraclebot.analysis.hourly_volatility import (compute_hourly_volatility_profile,
                                                    load_profile, save_profile)


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
