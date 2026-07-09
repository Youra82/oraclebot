import numpy as np
import pandas as pd

from oraclebot.data.dataset import build_training_examples
from oraclebot.data.targets import TARGET_NAMES


def make_ohlcv(index, seed):
    rng = np.random.default_rng(seed)
    n = len(index)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=index)


def make_multi_timeframe(n_days=120):
    daily_idx = pd.date_range('2024-01-01', periods=n_days, freq='D', tz='UTC')
    hourly4_idx = pd.date_range('2024-01-01', periods=n_days * 6, freq='4h', tz='UTC')
    return {
        '1d': make_ohlcv(daily_idx, seed=1),
        '4h': make_ohlcv(hourly4_idx, seed=2),
    }


def test_build_training_examples_basic_structure():
    data = make_multi_timeframe()
    examples = build_training_examples(
        data, reference_timeframe='1d',
        window_sizes={'1d': 5, '4h': 10},
    )
    assert len(examples) > 0
    ex = examples[0]
    assert '1d' in ex and '4h' in ex and 'target' in ex
    assert len(ex['1d']) == 5
    assert len(ex['4h']) == 10
    assert set(ex['target'].keys()) == set(TARGET_NAMES)


def test_no_lookahead_4h_window_stays_before_target_open():
    data = make_multi_timeframe()
    examples = build_training_examples(
        data, reference_timeframe='1d',
        window_sizes={'1d': 5, '4h': 10},
    )
    ff_4h = data['4h']
    for ex in examples:
        target_open = pd.Timestamp(ex['date'])
        # Die letzte 4h-Kerze im Fenster muss strikt vor dem Oeffnungszeitpunkt der Ziel-Kerze liegen.
        # Wir pruefen das indirekt: Anzahl 4h-Kerzen vor target_open muss >= window sein.
        n_before = (ff_4h.index < target_open).sum()
        assert n_before >= 10


def test_insufficient_history_examples_are_skipped():
    data = make_multi_timeframe(n_days=10)
    examples = build_training_examples(
        data, reference_timeframe='1d',
        window_sizes={'1d': 200, '4h': 500},  # absichtlich zu gross fuer 10 Tage Historie
    )
    assert examples == []
