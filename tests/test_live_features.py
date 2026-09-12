import os

import numpy as np
import pandas as pd
import pytest

from oraclebot.data.live_features import FeatureReconstructionError, _ensure_min_candles, build_reference_feature_vector

BARRIER_CFG = {'feature_settings': {'atr_window': 14, 'ema_window': 50, 'volume_window': 20}}


def make_ohlcv(n=300, freq='4h', seed=42, start='2026-01-01'):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq=freq, tz='UTC')
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    open_ = close + rng.normal(0, 0.3, n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.5, n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.5, n))
    volume = rng.uniform(100, 1000, n)
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume}, index=idx)


def _patch_fetch(monkeypatch, df):
    monkeypatch.setattr('oraclebot.data.live_features.fetch_ohlcv_incremental',
                         lambda symbol, timeframe, min_candles, cache_path: df)


def test_explicit_ref_ts_picks_that_candle(monkeypatch):
    df = make_ohlcv(n=200)
    _patch_fetch(monkeypatch, df)
    ref_ts = df.index[150]

    result = build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, '/tmp/artifacts', ref_ts=ref_ts)

    assert result['ref_ts'] == ref_ts
    assert result['entry_price'] == pytest.approx(float(df.loc[ref_ts, 'close']))
    assert len(result['feature_row']) == 21
    assert result['blocks'] == [('4h', ref_ts, result['feature_row'])]


def test_explicit_ref_ts_not_in_index_raises(monkeypatch):
    df = make_ohlcv(n=200)
    _patch_fetch(monkeypatch, df)
    missing_ts = pd.Timestamp('2020-01-01', tz='UTC')

    with pytest.raises(FeatureReconstructionError):
        build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, '/tmp/artifacts', ref_ts=missing_ts)


def test_none_ref_ts_uses_latest_completed_candle(monkeypatch):
    df = make_ohlcv(n=200)
    _patch_fetch(monkeypatch, df)
    # letzte Kerze ist noch nicht abgeschlossen -> muss uebersprungen werden
    now_utc = df.index[-1] + pd.Timedelta(hours=1)

    result = build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, '/tmp/artifacts',
                                             ref_ts=None, now_utc=now_utc)

    assert result['ref_ts'] == df.index[-2]


def test_none_ref_ts_uses_last_candle_if_already_closed(monkeypatch):
    df = make_ohlcv(n=200)
    _patch_fetch(monkeypatch, df)
    now_utc = df.index[-1] + pd.Timedelta(hours=5)  # 4h-Kerze laengst abgeschlossen

    result = build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, '/tmp/artifacts',
                                             ref_ts=None, now_utc=now_utc)

    assert result['ref_ts'] == df.index[-1]


def test_too_little_history_raises(monkeypatch):
    df = make_ohlcv(n=5)  # weit unter ema_window=50 -> compute_features liefert 0 Zeilen
    _patch_fetch(monkeypatch, df)

    with pytest.raises(FeatureReconstructionError):
        build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, '/tmp/artifacts',
                                        ref_ts=df.index[-1])


def test_ensure_min_candles_deletes_too_short_cache(tmp_path):
    cache_path = str(tmp_path / 'cache.pkl')
    make_ohlcv(n=50).to_pickle(cache_path)

    _ensure_min_candles(120, cache_path)

    assert not os.path.exists(cache_path)


def test_ensure_min_candles_keeps_sufficient_cache(tmp_path):
    cache_path = str(tmp_path / 'cache.pkl')
    make_ohlcv(n=200).to_pickle(cache_path)

    _ensure_min_candles(120, cache_path)

    assert os.path.exists(cache_path)


def test_ensure_min_candles_noop_if_missing(tmp_path):
    cache_path = str(tmp_path / 'does_not_exist.pkl')
    _ensure_min_candles(120, cache_path)  # darf nicht crashen
    assert not os.path.exists(cache_path)


def test_retrospective_check_backfills_thin_existing_cache(monkeypatch, tmp_path):
    """Reproduziert den echten Fund (2026-09-12): ein bereits vorhandener, aber zu schlanker
    Live-Cache (z.B. frisch auf einer neuen Maschine angelegt) darf eine retrospektive Pruefung
    fuer einen mehrere Tage alten Trade nicht mit 'kein Kontext gefunden' scheitern lassen --
    min_candles/context_min_candles muessen den Cache bei Bedarf per Neuabruf vertiefen."""
    full_df = make_ohlcv(n=400, freq='4h', seed=7, start='2026-01-01')
    thin_df = full_df.iloc[-30:]  # wie ein frisch anglegter, zu kurzer Live-Cache
    cache_path = str(tmp_path / 'ohlcv_live_BTC_USDT_USDT_4h.pkl')
    thin_df.to_pickle(cache_path)

    def fake_fetch(symbol, timeframe, min_candles, cache_path):
        # Simuliert fetch_ohlcv_incremental: kein Cache -> voller Abruf, sonst unveraendert lassen.
        if not os.path.exists(cache_path):
            return full_df.iloc[-min_candles:] if min_candles < len(full_df) else full_df
        return pd.read_pickle(cache_path)

    monkeypatch.setattr('oraclebot.data.live_features.fetch_ohlcv_incremental', fake_fetch)

    ref_ts = full_df.index[150]  # liegt vor dem schlanken Cache-Fenster

    with pytest.raises(FeatureReconstructionError):
        build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, str(tmp_path),
                                        ref_ts=ref_ts, min_candles=30)

    thin_df.to_pickle(cache_path)  # Cache-Datei wieder herstellen (voriger Aufruf hat sie geloescht)
    # min_candles muss so gross sein, dass nach dem ema_window=50-Warmup-Abzug vom Fenster-Anfang
    # ref_ts (Index 150 von 400) noch im gueltigen Feature-Bereich liegt.
    result = build_reference_feature_vector('BTC/USDT:USDT', '4h', [], BARRIER_CFG, str(tmp_path),
                                             ref_ts=ref_ts, min_candles=350)
    assert result['ref_ts'] == ref_ts


def test_context_timeframe_merges_last_candle_before_ref_ts(monkeypatch):
    # ctx_df (1d) muss lange genug VOR ref_df beginnen, damit ihr ema_window=50-Warmup bei
    # ref_ts bereits abgeschlossen ist (200x4h ref_df deckt nur ~33 Kalendertage ab).
    ref_df = make_ohlcv(n=200, freq='4h', seed=1, start='2026-03-01')
    ctx_df = make_ohlcv(n=200, freq='1d', seed=2, start='2025-10-01')

    def fake_fetch(symbol, timeframe, min_candles, cache_path):
        return ref_df if timeframe == '4h' else ctx_df

    monkeypatch.setattr('oraclebot.data.live_features.fetch_ohlcv_incremental', fake_fetch)
    ref_ts = ref_df.index[150]

    result = build_reference_feature_vector('BTC/USDT:USDT', '4h', ['1d'], BARRIER_CFG, '/tmp/artifacts',
                                             ref_ts=ref_ts)

    assert len(result['blocks']) == 2
    ctx_label, ctx_ts, ctx_values = result['blocks'][1]
    assert ctx_label == '1d'
    assert ctx_ts <= ref_ts
    assert len(ctx_values) == 21
    assert len(result['feature_row']) == 42
