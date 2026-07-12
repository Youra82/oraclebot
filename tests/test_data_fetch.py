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
    """Regression: Bitgets `since` ist exklusiv -- since_ms MUSS 1ms VOR der betroffenen
    gecachten Kerze liegen, sonst wird sie nie erneut abgefragt und bleibt fuer immer auf
    ihrem urspruenglichen (moeglicherweise unvollstaendigen) Stand eingefroren. Genau das
    passierte am 2026-07-11: prev_close zeigte dauerhaft den Open- statt den finalen
    Close-Preis. Mit nur 3 gecachten Kerzen (< REFRESH_WINDOW=5) ist die AELTESTE Kerze im
    Cache der massgebliche since-Ankerpunkt, nicht nur die allerletzte."""
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=3)
    cached.to_pickle(cache_path)

    # Mit n=3 < REFRESH_WINDOW=5 liegt since_ms bei der AELTESTEN Kerze -- ein realistischer
    # frischer Fetch bestaetigt daher den kompletten Cache-Bereich neu, nicht nur 1 Kerze.
    fresh = pd.DataFrame(
        {'open': 1.0, 'high': 2.0, 'low': 1.0, 'close': 1.8, 'volume': 1.0}, index=cached.index)

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=3, cache_path=str(cache_path))

    args, kwargs = mock_fetch.call_args
    expected_since_ms = int(cached.index[0].value // 1_000_000) - 1
    assert kwargs['since_ms'] == expected_since_ms


def test_refresh_window_reconfirms_multiple_recent_candles_not_just_the_last(tmp_path):
    """Regression fuer den am 2026-07-13 gefundenen Bug: das alte 'nur die letzte Zeile'-Design
    gab jeder Kerze GENAU EINEN Lauf lang die Chance, korrigiert zu werden -- sobald eine
    neuere Kerze angehaengt wurde, fiel die vorherige aus dem Fenster und blieb fuer immer auf
    ihrem (moeglicherweise noch unfertigen) Stand eingefroren, sichtbar im Chart als auffaellig
    duenne/doji-artige Kerzen kurz vor der Prognose-Box. Bei einem Cache > REFRESH_WINDOW muss
    since_ms auf die FUENFTLETZTE Kerze zeigen, nicht auf die letzte."""
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=10)
    cached.to_pickle(cache_path)

    fresh = pd.DataFrame(
        {'open': 1.0, 'high': 2.0, 'low': 1.0, 'close': 1.8, 'volume': 1.0}, index=cached.index[-5:])

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh
        df = fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=10, cache_path=str(cache_path))

    args, kwargs = mock_fetch.call_args
    expected_since_ms = int(cached.index[-5].value // 1_000_000) - 1
    assert kwargs['since_ms'] == expected_since_ms
    # Die ersten 5 (ausserhalb des Refresh-Fensters) bleiben unveraendert, die letzten 5 wurden
    # neu bestaetigt.
    assert (df['close'].iloc[:5] == 1.0).all()
    assert (df['close'].iloc[-5:] == 1.8).all()


def test_fresh_data_replaces_stale_last_cached_row(tmp_path):
    cache_path = tmp_path / 'cache.pkl'
    cached = make_cached_df(n=3)
    cached.to_pickle(cache_path)

    # Simuliert: die gecachten Kerzen waren unvollstaendig (close=1.0), der frische Fetch
    # bringt die finalen Close-Werte fuer den gesamten Refresh-Bereich.
    fresh = pd.DataFrame(
        {'open': 1.0, 'high': 2.0, 'low': 1.0, 'close': 1.8, 'volume': 1.0}, index=cached.index)

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
