from unittest.mock import patch

import pandas as pd

from oraclebot.utils.data_fetch import fetch_ohlcv_incremental

SYMBOL = 'BTC/USDT:USDT'


def make_cached_df(n=3, start='2026-07-01'):
    idx = pd.date_range(start, periods=n, freq='D', tz='UTC')
    return pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0, 'volume': 1.0}, index=idx)


def test_no_cache_does_full_fetch_without_since(tmp_path):
    cache_path = tmp_path / 'cache.pkl'
    fresh = make_cached_df(n=5)

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=5, cache_path=str(cache_path))

    args, kwargs = mock_fetch.call_args
    assert args == (SYMBOL, '1d')
    assert kwargs.get('limit') == 5
    assert 'since_ms' not in kwargs or kwargs['since_ms'] is None


def test_incremental_refetch_uses_inclusive_since_for_last_cached_candle(tmp_path):
    """Regression: Bitgets `since` ist exklusiv -- since_ms MUSS 1ms VOR der letzten gecachten
    Kerze liegen, sonst wird diese Kerze nie erneut abgefragt und bleibt fuer immer auf ihrem
    urspruenglichen (moeglicherweise unvollstaendigen) Stand eingefroren. Genau das passierte
    am 2026-07-11: prev_close zeigte dauerhaft den Open- statt den finalen Close-Preis."""
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=3)
    cached.to_pickle(cache_path)

    fresh = pd.DataFrame(
        {'open': 1.0, 'high': 2.0, 'low': 1.0, 'close': 1.8, 'volume': 1.0}, index=[cached.index[-1]])

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=3, cache_path=str(cache_path))

    args, kwargs = mock_fetch.call_args
    expected_since_ms = int(cached.index[-1].value // 1_000_000) - 1
    assert kwargs['since_ms'] == expected_since_ms


def test_fresh_data_replaces_stale_last_cached_row(tmp_path):
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=3)
    cached.to_pickle(cache_path)

    # Simuliert: letzte gecachte Kerze war unvollstaendig (close=1.0), der frische Fetch
    # bringt den finalen Close-Wert.
    fresh = pd.DataFrame(
        {'open': 1.0, 'high': 2.0, 'low': 1.0, 'close': 1.8, 'volume': 1.0}, index=[cached.index[-1]])

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        df = fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=3, cache_path=str(cache_path))

    assert df['close'].iloc[-1] == 1.8


def test_empty_fresh_response_falls_back_to_cache_unchanged(tmp_path):
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=3)
    cached.to_pickle(cache_path)

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        df = fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=3, cache_path=str(cache_path))

    pd.testing.assert_frame_equal(df, cached)


def test_cache_file_is_written_after_fetch(tmp_path):
    cache_path = tmp_path / 'cache.pkl'
    fresh = make_cached_df(n=5)

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=5, cache_path=str(cache_path))

    assert cache_path.exists()
    reloaded = pd.read_pickle(cache_path)
    assert len(reloaded) == 5
