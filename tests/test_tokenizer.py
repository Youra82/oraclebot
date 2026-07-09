import os

import numpy as np
import pandas as pd

from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.data.tokenizer import MarketTokenizer


def make_ohlcv(n=400, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range('2024-01-01', periods=n, freq='D', tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def make_feature_df():
    return compute_features(make_ohlcv())


def test_fit_transform_returns_valid_token_ids():
    feats = make_feature_df()
    tokenizer = MarketTokenizer(n_tokens=16, random_state=1)
    tokens = tokenizer.fit_transform(feats)
    assert len(tokens) == len(feats)
    assert tokens.min() >= 0
    assert tokens.max() < 16


def test_transform_is_deterministic_after_fit():
    feats = make_feature_df()
    tokenizer = MarketTokenizer(n_tokens=16, random_state=1)
    tokenizer.fit(feats)
    tokens_a = tokenizer.transform(feats)
    tokens_b = tokenizer.transform(feats)
    assert (tokens_a == tokens_b).all()


def test_describe_token_returns_interpretable_fields():
    feats = make_feature_df()
    tokenizer = MarketTokenizer(n_tokens=8, random_state=1)
    tokenizer.fit(feats)
    desc = tokenizer.describe_token(0)
    assert desc['token_id'] == 0
    assert 'Trend' in desc['label']
    assert set(desc['centroid'].keys()) == set(FEATURE_NAMES)


def test_vocabulary_summary_counts_match_number_of_rows():
    feats = make_feature_df()
    tokenizer = MarketTokenizer(n_tokens=10, random_state=1)
    tokenizer.fit(feats)
    summary = tokenizer.vocabulary_summary()
    assert summary['count'].sum() == len(feats)
    assert summary['token_id'].nunique() == len(summary)


def test_save_and_load_roundtrip(tmp_path):
    feats = make_feature_df()
    tokenizer = MarketTokenizer(n_tokens=12, random_state=3)
    tokens_before = tokenizer.fit_transform(feats)

    path = os.path.join(tmp_path, 'tokenizer.pkl')
    tokenizer.save(path)
    loaded = MarketTokenizer.load(path)
    tokens_after = loaded.transform(feats)

    assert (tokens_before == tokens_after).all()
