import numpy as np
import pandas as pd
import pytest

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
    # 'entry' ist der SCHLUSSkurs der Referenzkerze, real erst ab ts+4h (Kerzenschluss) bekannt/
    # handelbar -- die Barriere-Suche darf daher erst dort beginnen (Bugfix 2026-08-11, vorher
    # ts+15min: griff auf 15m-Baeren aus der Bildungsphase der Referenzkerze selbst zu).
    # Erste Kerze bleibt neutral, zweite erreicht +1% (101), dritte erst -1% (99)
    intraday = make_intraday_df([(99.5, 100.5), (100.5, 101.5), (98.5, 99.5)], start=ts + pd.Timedelta(hours=4))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 1
    assert labels.loc[ts, 'label'] == 1  # hoch zuerst


def test_down_first_detected():
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    intraday = make_intraday_df([(99.5, 100.5), (98.5, 99.5), (100.5, 101.5)], start=ts + pd.Timedelta(hours=4))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert labels.loc[ts, 'label'] == 0  # runter zuerst


def test_both_barriers_in_same_bar_prefers_down_first():
    """Konservative Konvention (wie im Rest des Projekts): wenn SL und TP in derselben feineren
    Kerze liegen, gewinnt SL (hier: runter zuerst)."""
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    intraday = make_intraday_df([(98.0, 102.0)], start=ts + pd.Timedelta(hours=4))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert labels.loc[ts, 'label'] == 0


def test_unresolved_barrier_is_dropped():
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    # Bleibt die ganze Zeit innerhalb von +-1%
    intraday = make_intraday_df([(99.7, 100.3)] * 5, start=ts + pd.Timedelta(hours=4))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 0


def test_barrier_search_ignores_bars_before_the_reference_candle_closes():
    """Regression 2026-08-11: die Barriere-Suche darf NICHT auf 15m-Baeren zugreifen, die noch
    innerhalb der Bildungsphase der Referenzkerze selbst liegen (vor ts+4h) -- sonst wird der
    (erst bei Kerzenschluss bekannte) Schlusskurs gegen Kursbewegung geprueft, die chronologisch
    VOR diesem Schlusskurs liegt. Ein Bar direkt nach Kerzenoeffnung (ts+15min), der die Barriere
    beruehrt, darf das Ergebnis nicht bestimmen -- erst ein Bar ab dem Kerzenschluss (ts+4h) zaehlt."""
    ref = make_reference_df([100.0])
    ts = ref.index[0]
    # Bar bei ts+15min (noch waehrend der Referenzkerze) verletzt bereits +-1% in beide
    # Richtungen -- muss ignoriert werden. Erst der Bar ab ts+4h (Kerzenschluss) darf zaehlen,
    # dort wird zuerst die obere Barriere beruehrt.
    intraday = make_intraday_df(
        [(90.0, 110.0)] + [(99.7, 100.3)] * 15 + [(100.5, 101.5)],
        start=ts + pd.Timedelta(minutes=15))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 1
    assert labels.loc[ts, 'label'] == 1  # hoch zuerst, aus dem Bar ab Kerzenschluss
    assert labels.loc[ts, 'exit_time'] >= ts + pd.Timedelta(hours=4)


def test_exit_time_is_strictly_after_reference_timestamp():
    ref = make_reference_df([100.0, 100.0])
    ts0 = ref.index[0]
    intraday = make_intraday_df([(99.0, 101.5)] * 40, start=ts0 + pd.Timedelta(hours=4))
    labels = compute_barrier_labels(ref, intraday, barrier_pct=1.0)
    assert len(labels) == 2
    assert (labels['exit_time'] > labels.index).all()


def make_synthetic_reference_ohlcv(n, freq='4h', start='2024-01-01', seed=0):
    """Echte (nicht entartete) Kerzen -- body/wick-Features brauchen high != low, sonst NaN durch 0/0."""
    rng = np.random.default_rng(seed)
    closes = 100 + np.cumsum(rng.normal(0, 0.3, n))
    idx = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    highs = np.maximum(opens, closes) + 0.5
    lows = np.minimum(opens, closes) - 0.5
    return pd.DataFrame({'open': opens, 'high': highs, 'low': lows, 'close': closes,
                          'volume': [100.0] * n}, index=idx)


def test_build_barrier_examples_returns_feature_and_target_per_row():
    n = 80
    ref = make_synthetic_reference_ohlcv(n, freq='4h')
    intraday = make_intraday_df(
        [(c - 2.0, c + 2.0) for c in ref['close'] for _ in range(16)],  # grosszuegige Range -> immer aufloesbar
        start=ref.index[0] + pd.Timedelta(minutes=15), freq='15min')
    ohlcv = {'4h': ref, '15m': intraday}
    examples = build_barrier_examples(ohlcv, reference_timeframe='4h', intraday_timeframe='15m', barrier_pct=1.0)
    assert len(examples) > 0
    for ex in examples:
        assert 'target' in ex and ex['target'] in (0, 1)
        assert 'features' in ex and isinstance(ex['features'], list)
        assert 'entry' in ex and 'exit_time' in ex


def test_build_barrier_examples_with_context_timeframes_appends_blocks():
    """Mit context_timeframes muss jeder Feature-Vektor laenger sein (Referenz- + Kontext-Bloecke)
    und darf keine Zukunftsdaten nutzen (No-Lookahead via merge_asof(direction='backward'))."""
    n = 80
    ref = make_synthetic_reference_ohlcv(n, freq='4h')
    intraday = make_intraday_df(
        [(c - 2.0, c + 2.0) for c in ref['close'] for _ in range(16)],
        start=ref.index[0] + pd.Timedelta(minutes=15), freq='15min')
    # 1h-Kontext: groebere Aufloesung, deckt denselben Zeitraum ab.
    hourly = make_synthetic_reference_ohlcv(n * 4, freq='1h', start=ref.index[0], seed=1)

    ohlcv = {'4h': ref, '15m': intraday, '1h': hourly}
    no_context = build_barrier_examples(ohlcv, reference_timeframe='4h', intraday_timeframe='15m', barrier_pct=1.0)
    with_context = build_barrier_examples(ohlcv, reference_timeframe='4h', intraday_timeframe='15m',
                                           context_timeframes=['1h'], barrier_pct=1.0)

    assert len(with_context) > 0
    ref_feature_len = len(no_context[0]['features'])
    for ex in with_context:
        assert len(ex['features']) == ref_feature_len * 2

    with_context_by_ts = {ex['reference_time']: ex for ex in with_context}
    for ex in no_context:
        ctx_ex = with_context_by_ts.get(ex['reference_time'])
        if ctx_ex is not None:
            assert ctx_ex['features'][:ref_feature_len] == ex['features']


def test_context_timeframe_with_too_little_history_raises_clear_error():
    """Regression 2026-07-26: ein Kontext-Timeframe mit zu wenig Rohkerzen fuers Feature-Warmup
    (z.B. nach einem --no-cache-Komplettabruf, bei dem Bitget fuer sehr grobe Zeitebenen wie 1M
    weniger Historie zurueckgibt als angefragt) liess compute_features() eine leere DataFrame
    OHNE DatetimeIndex zurueckgeben -- merge_asof() stuerzte dann tief in pandas-Internals mit
    einer kryptischen 'incompatible merge keys ... datetime64 and int64'-MergeError ab, statt
    klar zu sagen, welcher Timeframe das Problem ist. Muss stattdessen einen verstaendlichen
    RuntimeError werfen."""
    n = 80
    ref = make_synthetic_reference_ohlcv(n, freq='4h')
    intraday = make_intraday_df(
        [(c - 2.0, c + 2.0) for c in ref['close'] for _ in range(16)],
        start=ref.index[0] + pd.Timedelta(minutes=15), freq='15min')
    # Nur 1 Rohkerze fuer den Kontext-Timeframe -- reicht fuer kein Feature-Warmup-Fenster,
    # compute_features() liefert 0 Zeilen (siehe features.py).
    starved_context = make_synthetic_reference_ohlcv(1, freq='30D', start='2022-07-01')

    ohlcv = {'4h': ref, '15m': intraday, '1M': starved_context}
    with pytest.raises(RuntimeError, match='1M'):
        build_barrier_examples(ohlcv, reference_timeframe='4h', intraday_timeframe='15m',
                                context_timeframes=['1M'], barrier_pct=1.0)
