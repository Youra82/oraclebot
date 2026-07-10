# src/oraclebot/utils/data_fetch.py
# Oeffentlicher OHLCV-Download (keine API-Keys noetig) fuer den Dataset-Bau.
import logging
import os
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000, exchange_id: str = 'bitget') -> pd.DataFrame:
    """Laedt die letzten `limit` Kerzen fuer `symbol`/`timeframe` ueber die oeffentliche ccxt-API.

    Paginiert vorwaerts ab einem berechneten Startzeitpunkt (statt rueckwaerts anhand
    einer angenommenen Chunk-Groesse) -- Bitget liefert pro `since`-Request oft deutlich
    weniger Kerzen als angefragt (~90-100 statt 1000), was bei rueckwaerts-Paginierung
    stillschweigend Luecken in der Historie erzeugen wuerde.
    """
    exchange = getattr(ccxt, exchange_id)({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    exchange.load_markets()

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    since = now_ms - limit * timeframe_ms

    # Bitget-Eigenheiten bei paginierten `since`-Requests:
    # 1) Requests, deren impliziter Zeitraum (limit * timeframe_ms) 90 Tage ueberschreitet,
    #    werden abgelehnt (betrifft v.a. 1w/1M mit ihrer eigenen "utc"-Granularitaets-Variante).
    # 2) Wird ein `limit` > ~200 angefragt, ignoriert Bitget `since` und liefert stattdessen
    #    die letzten ~200 Kerzen VOR dem impliziten Ende (since + limit*timeframe_ms) --
    #    das reisst grosse Luecken, weil die Antwort dann nicht mehr bei `since` beginnt.
    # Ein Cap von 100 pro Call haelt beide Faelle sicher ein.
    max_span_ms = 89 * 24 * 3600 * 1000
    chunk_limit = max(2, min(100, max_span_ms // timeframe_ms))

    all_ohlcv = []
    max_iterations = limit  # Backstop, falls die Boerse pro Call nur 1 Kerze liefert
    for _ in range(max_iterations):
        # Retry statt sofortigem Abbruch bei leerer/fehlgeschlagener Antwort: bei sehr grob
        # aufgeloesten Timeframes (v.a. 1M, wo Bitget pro Call nur chunk_limit=2 Kerzen liefert)
        # braucht ein voller Fetch ~20 sequentielle Requests -- ein einzelner transienter
        # Netzwerk-/Rate-Limit-Hickser wuerde sonst die gesamte Historie vorzeitig abschneiden
        # (beobachtet 2026-07-10: VPS-Lauf brach nach 3 von 43 benoetigten 1M-Kerzen ab, obwohl
        # ein identischer Fetch von einem anderen Rechner aus vollstaendig durchlief).
        chunk = None
        for attempt in range(3):
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, since, chunk_limit)
            except Exception as e:
                logger.warning(f"{symbol} {timeframe}: Fetch-Fehler bei since={since} "
                               f"(Versuch {attempt + 1}/3): {type(e).__name__}: {e}")
                chunk = None
            if chunk:
                break
            if not chunk:
                logger.warning(f"{symbol} {timeframe}: Leere Antwort bei since={since} "
                               f"(Versuch {attempt + 1}/3, bisher {len(all_ohlcv)}/{limit} Kerzen).")
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
        if not chunk:
            logger.warning(f"{symbol} {timeframe}: Fetch abgebrochen nach {len(all_ohlcv)}/{limit} "
                           f"Kerzen (since={since} liefert weiterhin nichts nach 3 Versuchen).")
            break
        if all_ohlcv:
            chunk = [c for c in chunk if c[0] > all_ohlcv[-1][0]]
        if not chunk:
            break
        all_ohlcv.extend(chunk)
        # +1ms statt +timeframe_ms: Bitgets `since` ist exklusiv (timestamp > since),
        # ein voller Timeframe-Schritt trifft exakt die naechste Kerze und ueberspringt sie.
        since = all_ohlcv[-1][0] + 1
        if since >= now_ms or len(all_ohlcv) >= limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if not all_ohlcv:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)
    df = df[~df.index.duplicated(keep='last')]
    if len(df) > limit:
        df = df.iloc[-limit:]

    gaps = df.index.to_series().diff().dropna()
    expected = pd.Timedelta(milliseconds=timeframe_ms)
    unexpected_gaps = gaps[gaps > expected * 1.5]
    if len(unexpected_gaps) > 0:
        logger.warning(f"{symbol} {timeframe}: {len(unexpected_gaps)} unerwartete Luecke(n) in der Historie entdeckt.")

    return df


def fetch_all_timeframes(symbol: str, timeframes: list, history_days: int, cache_dir: str = None,
                          use_cache: bool = True) -> dict:
    """Laedt OHLCV fuer alle Timeframes eines Symbols, mit optionalem Datei-Cache.

    Der volle 6-Timeframe-Fetch (v.a. 15m ueber hunderte Tage) dauert mehrere Minuten -- ein
    Cache erspart das erneute Fetchen bei wiederholten Trainings-/Backtest-/Chart-Laeufen.
    Wird sowohl von train_transformer.py als auch von den Backtest-/Chart-Scripts genutzt,
    damit alle denselben Cache treffen.
    """
    ohlcv_by_timeframe = {}
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    for tf in timeframes:
        limit = max(50, int(history_days * 24 * 60 / TIMEFRAME_MINUTES[tf]))
        cache_path = os.path.join(cache_dir, f"ohlcv_{safe_symbol}_{tf}_{limit}.pkl") if cache_dir else None

        if use_cache and cache_path and os.path.exists(cache_path):
            df = pd.read_pickle(cache_path)
            logger.info(f"{symbol} {tf}: {len(df)} Kerzen aus Cache geladen ({cache_path}).")
        else:
            logger.info(f"Lade {symbol} {tf} ({limit} Kerzen, ~{history_days} Tage)...")
            df = fetch_ohlcv(symbol, tf, limit=limit)
            logger.info(f"  -> {len(df)} Kerzen: {df.index[0]} bis {df.index[-1]}" if len(df) else "  -> keine Daten")
            if cache_path:
                df.to_pickle(cache_path)
        ohlcv_by_timeframe[tf] = df
    return ohlcv_by_timeframe
