import numpy as np
import pandas as pd

from oraclebot.data.barrier_targets import build_barrier_examples, compute_barrier_labels


def make_reference_df(closes, start='2024-01-01', freq='4h'):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz='UTC')
    return pd.DataFrame({'open': closes, 'high': closes, 'low': closes, 'close': closes,
                          'volume': [100.0] * len(closes)}, index=idx)


def make_intraday_df(rows, start, freq='15min'):
    """rows: Liste von (low, high) Tupeln, eine Zeile pro `freq`-Schritt ab `start`."""
    idx = pd.date_range(start, periods=len(rows), freq=freq, tz='UTC')
    lows = [r[0] for r in rows]
    highs = [r[1] for r in rows]
    closes = [(l + h) / 2 for l, h in rows]
    return pd.DataFrame({'open': closes, 'high': highs, 'low': lows, 'close': closes,
                          'volume': [10.0] * len(rows)}, index=idx)


def test_up_first_detected():
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    # Erste Kerze bleibt neutral, zweite erreicht +1% (101), dritte erst -1% (99)
    intraday = make_intraday_df([(99.5, 100.5), (100.5, 101.5), (98.5, 99.5)], start=ts + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 1
    assert labels.loc[ts, 'label'] == 1  # hoch zuerst


def test_down_first_detected():
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    intraday = make_intraday_df([(99.5, 100.5), (98.5, 99.5), (100.5, 101.5)], start=ts + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert labels.loc[ts, 'label'] == 0  # runter zuerst


def test_both_barriers_in_same_bar_prefers_down_first():
    """Konservative Konvention (wie im Rest des Projekts): wenn SL und TP in derselben feineren
    Kerze liegen, gewinnt SL (hier: runter zuerst)."""
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    intraday = make_intraday_df([(98.0, 102.0)], start=ts + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert labels.loc[ts, 'label'] == 0


def test_unresolved_barrier_is_dropped():
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    # Bleibt die ganze Zeit innerhalb von +-1%
    intraday = make_intraday_df([(99.7, 100.3)] * 5, start=ts + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 0


def test_exit_time_is_strictly_after_reference_timestamp():
    ref = make_reference_df([100.0, 100.0])
    ts0 = ref.index[0]
    intraday = make_intraday_df([(99.0, 101.5)] * 10, start=ts0 + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert (labels['exit_time'] > labels.index).all()


def test_build_barrier_examples_returns_feature_and_target_per_row():
    n = 80
    rng = np.random.default_rng(0)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, n))
    idx = pd.date_range('2024-01-01', periods=n, freq='4h', tz='UTC')
    # Echte (nicht entartete) Kerzen -- body/wick-Features brauchen high != low, sonst NaN durch 0/0.
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    ref = pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes,
                         'volume': [100.0] * n}, index=idx)
    intraday = make_intraday_df(
        [(c - 2.0, c + 2.0) for c in closes for _ in range(16)],  # grosszuegige Range -> immer aufloesbar
        start=ref.index[0] + pd.Timedelta(minutes=15), freq='15min')
    examples = build_barrier_examples(ref, intraday, barrier_pct=1.0)
    assert len(examples) > 0
    for ex in examples:
        assert 'target' in ex and ex['target'] in (0, 1)
        assert 'features' in ex and isinstance(ex['features'], list)
        assert 'entry' in ex and 'exit_time' in ex
