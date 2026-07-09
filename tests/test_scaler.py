import os

import numpy as np
import pandas as pd

from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.data.scaler import FeatureScaler


def make_feature_df():
    rng = np.random.default_rng(3)
    idx = pd.date_range('2024-01-01', periods=300, freq='D', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, 300))
    open_ = close + rng.normal(0, 0.3, 300)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, 300))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, 300))
    volume = rng.uniform(100, 1000, 300)
    df = pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)
    return compute_features(df)


def test_fit_transform_produces_standardized_output():
    feats = make_feature_df()
    scaler = FeatureScaler().fit(feats)
    scaled = scaler.transform(feats)

    assert scaled.dtype == np.float32
    assert scaled.shape == (len(feats), len(FEATURE_NAMES))
    # Standardisiert: Mittelwert nahe 0, Std nahe 1 je Spalte
    assert np.allclose(scaled.mean(axis=0), 0, atol=1e-1)
    assert np.allclose(scaled.std(axis=0), 1, atol=1e-1)


def test_transform_before_fit_raises():
    scaler = FeatureScaler()
    feats = make_feature_df()
    try:
        scaler.transform(feats)
        assert False, "sollte RuntimeError werfen"
    except RuntimeError:
        pass


def test_save_load_roundtrip(tmp_path):
    feats = make_feature_df()
    scaler = FeatureScaler().fit(feats)
    scaled_before = scaler.transform(feats)

    path = os.path.join(tmp_path, 'scaler.pkl')
    scaler.save(path)
    loaded = FeatureScaler.load(path)
    scaled_after = loaded.transform(feats)

    assert np.allclose(scaled_before, scaled_after)
