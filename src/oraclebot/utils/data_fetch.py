# src/oraclebot/utils/data_fetch.py
# Oeffentlicher OHLCV-Download (keine API-Keys noetig) fuer den Dataset-Bau.
import logging
import os
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000, exchange_id: str = 'bitget',
                 since_ms: int = None) -> pd.DataFrame:
    """Laedt die letzten `limit` Kerzen fuer `symbol`/`timeframe` ueber die oeffentliche ccxt-API.

    Paginiert vorwaerts ab einem berechneten Startzeitpunkt (statt rueckwaerts anhand
    einer angenommenen Chunk-Groesse) -- Bitget liefert pro `since`-Request oft deutlich
    weniger Kerzen als angefragt (~90-100 statt 1000), was bei rueckwaerts-Paginierung
    stillschweigend Luecken in der Historie erzeugen wuerde.

    `since_ms`: optionaler expliziter Startzeitpunkt (ms seit Epoch), z.B. fuer inkrementelle
    Updates ab dem letzten Cache-Stand (siehe fetch_ohlcv_incremental()) -- ueberschreibt die
    sonst aus `limit` berechnete Startzeit.
    """
    exchange = getattr(ccxt, exchange_id)({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    exchange.load_markets()

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    since = since_ms if since_ms is not None else now_ms - limit * timeframe_ms

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
        # Nahe an "jetzt" ist eine leere Antwort der NORMALE Endzustand (die aktuell laufende
        # Kerze existiert schlicht noch nicht als abfragbare since-Grenze) -- kein Fehler, kein
        # Retry noetig. Nur bei since deutlich in der Vergangenheit (dort SOLLTE Historie
        # existieren) ist eine leere Antwort verdaechtig genug fuer Retries + Log-Warnungen.
        near_now = (now_ms - since) < (timeframe_ms * 2)
        attempts = 1 if near_now else 3
        chunk = None
        for attempt in range(attempts):
            try:
                chunk = exchange.fetch_ohlcv(symbol, timeframe, since, chunk_limit)
            except Exception as e:
                if not near_now:
                    logger.warning(f"{symbol} {timeframe}: Fetch-Fehler bei since={since} "
                                   f"(Versuch {attempt + 1}/{attempts}): {type(e).__name__}: {e}")
                chunk = None
            if chunk:
                break
            if not chunk and not near_now:
                logger.warning(f"{symbol} {timeframe}: Leere Antwort bei since={since} "
                               f"(Versuch {attempt + 1}/{attempts}, bisher {len(all_ohlcv)}/{limit} Kerzen).")
            if attempt < attempts - 1:
                time.sleep(1.0 * (attempt + 1))
        if not chunk:
            if not near_now:
                logger.warning(f"{symbol} {timeframe}: Fetch abgebrochen nach {len(all_ohlcv)}/{limit} "
                               f"Kerzen (since={since} liefert weiterhin nichts nach {attempts} Versuchen).")
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


def fetch_ohlcv_incremental(symbol: str, timeframe: str, min_candles: int, cache_path: str) -> pd.DataFrame:
    """Live-Cache fuer predict_next_candle.py: erster Lauf holt die volle benoetigte Historie
    und speichert sie unter `cache_path` (gitignored, siehe artifacts/datasets/ohlcv_*.pkl in
    .gitignore); jeder weitere Lauf haengt nur die Kerzen seit dem letzten Cache-Stand an.

    Bei 1M/1w bedeutet das an den meisten Tagen NULL neue API-Calls (die Kerze hat sich seit
    gestern nicht geaendert) statt einer vollen ~15-20-Request-Paginierung -- genau der Teil,
    der auf einem VPS wiederholt deterministisch fehlschlug (2026-07-10). Die letzte gecachte
    Kerze wird IMMER neu abgefragt (nicht ab danach), falls sie beim letzten Lauf noch nicht
    abgeschlossen war und sich der Wert seitdem noch aendern konnte.
    """
    cached = pd.read_pickle(cache_path) if os.path.exists(cache_path) else pd.DataFrame()

    if len(cached) == 0:
        logger.info(f"{symbol} {timeframe}: kein Live-Cache vorhanden, hole volle Historie ({min_candles} Kerzen)...")
        df = fetch_ohlcv(symbol, timeframe, limit=min_candles)
    else:
        exchange = ccxt.bitget({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
        timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
        # -1ms, NICHT die exakte Kerzen-Zeit: Bitgets `since` ist exklusiv (liefert nur
        # timestamp > since), ein since=<Zeitstempel der letzten Kerze> haette diese Kerze
        # NIE erneut zurueckbekommen -- sie blieb dadurch fuer immer auf dem Stand eingefroren,
        # zu dem sie urspruenglich gecacht wurde (beobachtet 2026-07-11: prev_close zeigte
        # dauerhaft den OPEN-Preis statt des finalen Close, weil die 1d-Kerze direkt nach
        # Tagesbeginn gecacht und danach nie mehr aktualisiert wurde -- Open~Close zu dem
        # sehr fruehen Zeitpunkt, daher unbemerkt plausibel bis der Kurs sich deutlich bewegte).
        since_ms = int(cached.index[-1].value // 1_000_000) - 1
        logger.info(f"{symbol} {timeframe}: Cache-Stand bis {cached.index[-1]} ({len(cached)} Kerzen), "
                    f"hole inkrementell ab since={pd.Timestamp(since_ms, unit='ms', tz='UTC')}...")
        fresh = fetch_ohlcv(symbol, timeframe, limit=max(min_candles, 50), since_ms=since_ms)
        if len(fresh) == 0:
            logger.warning(f"{symbol} {timeframe}: inkrementeller Fetch lieferte NICHTS (since="
                           f"{pd.Timestamp(since_ms, unit='ms', tz='UTC')}), nutze reinen Cache-Stand "
                           f"({cached.index[-1]}) unveraendert weiter.")
            df = cached
        else:
            logger.info(f"{symbol} {timeframe}: frischer Fetch liefert {len(fresh)} Kerze(n), "
                        f"{fresh.index[0]} bis {fresh.index[-1]}.")
            df = pd.concat([cached.iloc[:-1], fresh]) if len(cached) > 1 else fresh
            df = df[~df.index.duplicated(keep='last')].sort_index()
        logger.info(f"{symbol} {timeframe}: {len(fresh) if len(fresh) else 0} neue/aktualisierte Kerze(n) "
                    f"seit Cache-Stand ({len(cached)} -> {len(df)}), letzte Kerze jetzt: {df.index[-1]}.")

    # Cache nicht unbegrenzt wachsen lassen -- genug Puffer fuer kuenftige Fenster-Vergroesserungen.
    if len(df) > min_candles * 3:
        df = df.iloc[-min_candles * 3:]

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_pickle(cache_path)

    if len(df) < min_candles:
        logger.warning(f"{symbol} {timeframe}: Cache hat nur {len(df)}/{min_candles} Kerzen. "
                        f"Hole fehlende Historie zusaetzlich nach...")
        backfill = fetch_ohlcv(symbol, timeframe, limit=min_candles)
        if len(backfill) > len(df):
            df = backfill
            df.to_pickle(cache_path)

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
