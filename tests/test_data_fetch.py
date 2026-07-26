from unittest.mock import MagicMock, patch

import ccxt
import pandas as pd
import pytest

import oraclebot.utils.data_fetch as data_fetch_mod
from oraclebot.utils.data_fetch import (
    TIMEFRAME_MINUTES, fetch_all_timeframes, fetch_ohlcv_incremental, resample_ohlcv,
)

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


def test_incremental_does_not_retry_when_fetch_ohlcv_already_did_its_best(tmp_path, caplog):
    """Bugfix 2026-07-26: fetch_ohlcv() versucht bereits selbst aktiv, echte Boersen-Datenluecken
    zu ueberspringen (siehe _probe_next_available_ts) -- ein Shortfall danach ist fast immer
    dauerhaft (die fehlenden Kerzen existieren schlicht nicht), nicht transient. Ein identischer
    Backfill-Retry wuerde denselben teuren Fetch (inkl. erneuter Luecken-Suche) nur nutzlos
    wiederholen. fetch_ohlcv() darf bei einem Shortfall daher NUR EINMAL aufgerufen werden."""
    cache_path = tmp_path / 'cache.pkl'
    short_result = make_cached_df(n=8)  # < min_candles=10 -- simuliert eine dauerhafte Luecke

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = short_result
        with caplog.at_level('INFO'):
            df = fetch_ohlcv_incremental(SYMBOL, '1d', min_candles=10, cache_path=str(cache_path))

    assert mock_fetch.call_count == 1  # kein Backfill-Retry
    assert len(df) == 8


def _make_mock_exchange(fetch_responses, now_ms, timeframe_seconds):
    """fetch_responses: Liste von Kerzen-Listen -- jeder exchange.fetch_ohlcv()-Aufruf liefert
    das naechste Element (leere Liste, sobald die Liste erschoepft ist)."""
    exchange = MagicMock()
    exchange.load_markets.return_value = None
    exchange.parse_timeframe.return_value = timeframe_seconds
    exchange.milliseconds.return_value = now_ms
    exchange.rateLimit = 0
    call_iter = iter(fetch_responses)
    exchange.fetch_ohlcv.side_effect = lambda *a, **k: next(call_iter, [])
    return exchange


def _make_gapped_mock_exchange(now_ms, timeframe_seconds, gap_start_ms, gap_end_ms, chunk_limit=200):
    """Simuliert eine Boerse mit einer ECHTEN, dauerhaften Datenluecke in [gap_start_ms,
    gap_end_ms) -- jede Anfrage (Hauptschleife UND Probe) reagiert konsistent auf `since`, statt
    eine feste Antwortfolge abzuspulen. Bildet den 2026-07-26 auf einem VPS beobachteten Fall
    nach: zwei komplett unabhaengige Fetch-Versuche brachen exakt an derselben Stelle ab, weil es
    sich um eine echte Boersen-Datenluecke handelte, nicht um einen einzelnen Netzwerk-Hickser."""
    timeframe_ms = timeframe_seconds * 1000
    exchange = MagicMock()
    exchange.load_markets.return_value = None
    exchange.parse_timeframe.return_value = timeframe_seconds
    exchange.milliseconds.return_value = now_ms
    exchange.rateLimit = 0

    def fake_fetch(symbol, timeframe, since, limit):
        # Kerzen-Zeitstempel MUESSEN auf das Zeitraster gerundet werden (wie bei einer echten
        # Boerse) statt roh vom (moeglicherweise unausgerichteten, z.B. "...+1ms") `since`-Wert
        # uebernommen zu werden -- sonst rueckt `since = chunk[-1][0] + 1` bei jedem Call nur um
        # 1ms statt um eine volle Kerzenlaenge vor, was in der Naehe von "jetzt" Millionen
        # Aufrufe statt ein paar Dutzend braeuchte (im Test gefunden 2026-07-26).
        ts = ((since // timeframe_ms) + 1) * timeframe_ms
        if gap_start_ms <= ts < gap_end_ms:
            return []
        candles = []
        while len(candles) < limit and ts < now_ms:
            if gap_start_ms <= ts < gap_end_ms:
                break  # Chunk endet an der Luecke, nicht dahinter springen
            candles.append([ts, 1.0, 1.0, 1.0, 1.0, 1.0])
            ts += timeframe_ms
        return candles

    exchange.fetch_ohlcv.side_effect = fake_fetch
    return exchange


def test_probe_next_available_ts_finds_the_end_of_a_gap(monkeypatch):
    monkeypatch.setattr(data_fetch_mod.time, 'sleep', lambda s: None)
    timeframe_seconds = 60 * 60  # '1h'
    timeframe_ms = timeframe_seconds * 1000
    now_ms = 10_000 * timeframe_ms
    gap_start = now_ms - 5000 * timeframe_ms
    gap_end = gap_start + 500 * timeframe_ms  # 500h Luecke

    exchange = _make_gapped_mock_exchange(now_ms, timeframe_seconds, gap_start, gap_end)
    found = data_fetch_mod._probe_next_available_ts(exchange, SYMBOL, '1h', gap_start, now_ms, timeframe_ms)

    assert found is not None
    assert gap_end - timeframe_ms <= found <= gap_end + timeframe_ms  # nah an der echten Grenze


def test_probe_next_available_ts_returns_none_if_gap_extends_to_upper_bound(monkeypatch):
    monkeypatch.setattr(data_fetch_mod.time, 'sleep', lambda s: None)
    timeframe_seconds = 60 * 60
    timeframe_ms = timeframe_seconds * 1000
    now_ms = 10_000 * timeframe_ms
    gap_start = now_ms - 5000 * timeframe_ms

    # Luecke reicht bis "jetzt" -- nirgends mehr Daten zu finden.
    exchange = _make_gapped_mock_exchange(now_ms, timeframe_seconds, gap_start, now_ms + timeframe_ms)
    found = data_fetch_mod._probe_next_available_ts(exchange, SYMBOL, '1h', gap_start, now_ms, timeframe_ms)

    assert found is None


def test_fetch_ohlcv_skips_over_a_real_gap_and_continues_afterward(monkeypatch):
    """Bugfix 2026-07-26: eine echte Boersen-Datenluecke (siehe dnabot-Fund: BTC 1h fehlte
    komplett fuer 23 Tage) liess den alten Ganz-Fetch-Retry-Mechanismus deterministisch an
    derselben Stelle scheitern, egal wie oft wiederholt wurde. fetch_ohlcv() muss die Luecke
    stattdessen ueberspringen (per _probe_next_available_ts) und DANACH normal weiterladen."""
    monkeypatch.setattr(data_fetch_mod.time, 'sleep', lambda s: None)
    timeframe_seconds = 60 * 60  # '1h'
    timeframe_ms = timeframe_seconds * 1000
    now_ms = 10_000 * timeframe_ms
    gap_start = now_ms - 5000 * timeframe_ms
    gap_end = gap_start + 500 * timeframe_ms

    exchange = _make_gapped_mock_exchange(now_ms, timeframe_seconds, gap_start, gap_end)
    exchange_ctor = MagicMock(return_value=exchange)
    monkeypatch.setattr(data_fetch_mod.ccxt, 'bitget', exchange_ctor)

    df = data_fetch_mod.fetch_ohlcv(SYMBOL, '1h', limit=6000, since_ms=gap_start)

    assert len(df) > 0
    # Es darf keine Kerze INNERHALB der Luecke geben -- die Historie vor und nach der Luecke
    # muss aber beide vorhanden sein.
    in_gap = df[(df.index >= pd.Timestamp(gap_start, unit='ms', tz='UTC'))
                & (df.index < pd.Timestamp(gap_end, unit='ms', tz='UTC'))]
    assert len(in_gap) == 0
    assert df.index[-1] > pd.Timestamp(gap_end, unit='ms', tz='UTC')


def test_fetch_ohlcv_retries_on_rate_limit_exceeded_without_advancing_since(monkeypatch):
    monkeypatch.setattr(data_fetch_mod.time, 'sleep', lambda s: None)
    timeframe_seconds = 60 * 60
    timeframe_ms = timeframe_seconds * 1000
    now_ms = 10_000 * timeframe_ms
    since_ms = now_ms - 3 * timeframe_ms

    good_candles = [[since_ms + 1 + i * timeframe_ms, 1.0, 1.0, 1.0, 1.0, 1.0] for i in range(3)]
    exchange = MagicMock()
    exchange.load_markets.return_value = None
    exchange.parse_timeframe.return_value = timeframe_seconds
    exchange.milliseconds.return_value = now_ms
    exchange.rateLimit = 0
    # Erster Aufruf: RateLimitExceeded. Zweiter Aufruf (gleiches since, da bei der Exception
    # nicht weitergerueckt wird): liefert die Kerzen.
    exchange.fetch_ohlcv.side_effect = [ccxt.RateLimitExceeded('too many requests'), good_candles, []]
    monkeypatch.setattr(data_fetch_mod.ccxt, 'bitget', MagicMock(return_value=exchange))

    df = data_fetch_mod.fetch_ohlcv(SYMBOL, '1h', limit=10, since_ms=since_ms)

    assert len(df) == 3
    assert exchange.fetch_ohlcv.call_count == 3


def test_fetch_ohlcv_empty_response_at_now_ends_cleanly_without_warning(monkeypatch, caplog):
    monkeypatch.setattr(data_fetch_mod.time, 'sleep', lambda s: None)
    timeframe_seconds = 60 * 60
    timeframe_ms = timeframe_seconds * 1000
    now_ms = 10_000 * timeframe_ms
    since_ms = now_ms - timeframe_ms  # 1 Schritt vor "jetzt" -- normaler Endzustand

    exchange = _make_mock_exchange([[]], now_ms, timeframe_seconds)
    monkeypatch.setattr(data_fetch_mod.ccxt, 'bitget', MagicMock(return_value=exchange))

    with caplog.at_level('WARNING'):
        df = data_fetch_mod.fetch_ohlcv(SYMBOL, '1h', limit=10, since_ms=since_ms)

    assert len(df) == 0
    assert not any('Luecke' in r.message or 'Leere Antwort' in r.message for r in caplog.records)


def make_daily_df(n_days, start='2024-01-01'):
    idx = pd.date_range(start, periods=n_days, freq='D', tz='UTC')
    closes = [100.0 + i * 0.1 for i in range(n_days)]
    return pd.DataFrame({
        'open': closes, 'high': [c + 1 for c in closes], 'low': [c - 1 for c in closes],
        'close': closes, 'volume': [10.0] * n_days,
    }, index=idx)


def test_resample_ohlcv_aggregates_daily_into_monthly():
    """Regression 2026-07-26: '1M' wird nicht mehr direkt von Bitget abgefragt, sondern per
    Resampling aus '1d' abgeleitet -- muss die ueblichen OHLCV-Aggregationsregeln einhalten."""
    daily = make_daily_df(65, start='2024-01-01')  # Jan (31) + Feb (29, Schaltjahr) + 5 Tage Maerz
    monthly = resample_ohlcv(daily, '1M')

    assert list(monthly.index) == [pd.Timestamp('2024-01-01', tz='UTC'),
                                    pd.Timestamp('2024-02-01', tz='UTC'),
                                    pd.Timestamp('2024-03-01', tz='UTC')]
    jan = daily.loc['2024-01-01':'2024-01-31']
    assert monthly.loc['2024-01-01', 'open'] == jan['open'].iloc[0]
    assert monthly.loc['2024-01-01', 'close'] == jan['close'].iloc[-1]
    assert monthly.loc['2024-01-01', 'high'] == jan['high'].max()
    assert monthly.loc['2024-01-01', 'low'] == jan['low'].min()
    assert monthly.loc['2024-01-01', 'volume'] == jan['volume'].sum()


def test_resample_ohlcv_rejects_unmapped_timeframe():
    with pytest.raises(ValueError):
        resample_ohlcv(make_daily_df(10), '15m')


def test_fetch_all_timeframes_derives_1M_from_1d_without_direct_fetch(tmp_path):
    """'1M' darf NIE mehr direkt bei Bitget angefragt werden (siehe resample_ohlcv-Begruendung)
    -- fetch_all_timeframes() muss stattdessen automatisch '1d' mitladen und daraus ableiten."""
    daily = make_daily_df(400, start='2024-01-01')

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = daily
        result = fetch_all_timeframes(SYMBOL, ['1M', '1h'], history_days=400,
                                       cache_dir=str(tmp_path), use_cache=False)

    fetched_timeframes = [call.args[1] for call in mock_fetch.call_args_list]
    assert '1M' not in fetched_timeframes
    assert '1d' in fetched_timeframes  # automatisch mitgeladen fuer die Ableitung
    assert '1h' in fetched_timeframes

    assert '1M' in result
    assert len(result['1M']) > 0
    # '1d' war nicht explizit angefragt -- soll nicht unnoetig im Ergebnis auftauchen.
    assert '1d' not in result


def test_fetch_all_timeframes_keeps_1d_in_result_if_explicitly_requested(tmp_path):
    daily = make_daily_df(400, start='2024-01-01')

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = daily
        result = fetch_all_timeframes(SYMBOL, ['1M', '1d'], history_days=400,
                                       cache_dir=str(tmp_path), use_cache=False)

    assert '1d' in result
    assert '1M' in result


def test_fetch_all_timeframes_use_cache_true_still_attempts_an_incremental_top_up(tmp_path):
    """Bugfix 2026-07-26: use_cache=True las eine vorhandene Cache-Datei bisher 1:1 ein, OHNE
    jemals fetch_ohlcv() aufzurufen -- ein Trainings-/Optimizer-Lauf mit 'n' auf die Cache-Frage
    (run_pipeline.sh/optimize.sh) bekam dadurch denselben (potenziell wochenalten) Datenstand
    fuer immer, live beobachtet als Trainingsdaten, die ueber Wochen bei einem festen Datum
    haengen blieben (kompletter Monat ohne Trades im Backtest). Jetzt MUSS auch bei use_cache=True
    ein inkrementeller Top-up-Versuch (via fetch_ohlcv_incremental -> fetch_ohlcv) stattfinden."""
    limit = max(50, int(400 * 24 * 60 / TIMEFRAME_MINUTES['1d']))
    safe_symbol = SYMBOL.replace('/', '_').replace(':', '_')
    cache_path = tmp_path / f"ohlcv_{safe_symbol}_1d_{limit}.pkl"
    stale = make_daily_df(limit, start='2024-01-01')
    stale.to_pickle(cache_path)

    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        fetch_all_timeframes(SYMBOL, ['1d'], history_days=400, cache_dir=str(tmp_path), use_cache=True)

    mock_fetch.assert_called()  # vorher: bei vorhandenem Cache NIE aufgerufen


def test_fetch_all_timeframes_use_cache_true_extends_a_stale_cache_with_fresh_data(tmp_path):
    limit = max(50, int(400 * 24 * 60 / TIMEFRAME_MINUTES['1d']))
    safe_symbol = SYMBOL.replace('/', '_').replace(':', '_')
    cache_path = tmp_path / f"ohlcv_{safe_symbol}_1d_{limit}.pkl"
    stale = make_daily_df(limit, start='2024-01-01')
    stale.to_pickle(cache_path)
    stale_last_date = stale.index[-1]

    fresh_row = pd.DataFrame({'open': 200.0, 'high': 201.0, 'low': 199.0, 'close': 200.5, 'volume': 20.0},
                              index=[stale_last_date + pd.Timedelta(days=1)])
    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh_row
        result = fetch_all_timeframes(SYMBOL, ['1d'], history_days=400, cache_dir=str(tmp_path), use_cache=True)

    assert result['1d'].index[-1] > stale_last_date


def test_fetch_all_timeframes_use_cache_false_deletes_stale_cache_and_forces_full_fetch(tmp_path):
    limit = max(50, int(400 * 24 * 60 / TIMEFRAME_MINUTES['1d']))
    safe_symbol = SYMBOL.replace('/', '_').replace(':', '_')
    cache_path = tmp_path / f"ohlcv_{safe_symbol}_1d_{limit}.pkl"
    make_daily_df(limit, start='2024-01-01').to_pickle(cache_path)

    fresh_full = make_daily_df(limit, start='2025-01-01')
    with patch('oraclebot.utils.data_fetch.fetch_ohlcv') as mock_fetch:
        mock_fetch.return_value = fresh_full
        result = fetch_all_timeframes(SYMBOL, ['1d'], history_days=400, cache_dir=str(tmp_path), use_cache=False)

    # since_ms darf NICHT gesetzt sein -- ein erzwungener Neuabruf muss bei komplett frischem
    # (durch Loeschen des Caches leerem) Zustand starten, nicht inkrementell ab dem alten Stand.
    _, kwargs = mock_fetch.call_args
    assert kwargs.get('since_ms') is None
    assert result['1d'].index[0] == fresh_full.index[0]
