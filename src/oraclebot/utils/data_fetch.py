# src/oraclebot/utils/data_fetch.py
# Oeffentlicher OHLCV-Download (keine API-Keys noetig) fuer den Dataset-Bau.
import logging
import os
import time

import ccxt
import pandas as pd

from oraclebot.utils.progress import finish_progress, render_progress

logger = logging.getLogger(__name__)

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def _probe_next_available_ts(exchange, symbol: str, timeframe: str, from_ts: int, upper_bound_ts: int,
                              tf_ms: int) -> int:
    """Sucht den naechsten Zeitpunkt >= from_ts, an dem Bitget wieder Kerzen liefert.

    Portiert aus dnabot/src/dnabot/utils/exchange.py -- Bitget hat fuer manche Symbole/
    Timeframes bestaetigte, mehrwoechige Luecken in der eigenen Historie (dnabot-Fund: BTC 1h
    fehlte komplett vom 2026-04-18 bis 2026-05-10, davor/danach regulaer abrufbar). Kein
    Rate-Limit-Problem, sondern eine echte Datenluecke -- blindes Wiederholen derselben Anfrage
    liefert IMMER wieder nichts, egal wie oft oder von welcher Maschine aus (beobachtet
    2026-07-26 bei oraclebot: zwei komplett unabhaengige Fetch-Versuche brachen exakt am selben
    `since` ab). Erst exponentiell wachsende Schritte vorwaerts tasten, bis irgendwo wieder Daten
    auftauchen, dann per Bisektion zwischen letztem leeren und erstem gefundenen Punkt die genaue
    Grenze eingrenzen. Gibt None zurueck, wenn bis upper_bound_ts nirgends mehr Daten zu finden sind.
    """
    lo = from_ts
    hi = from_ts
    step = tf_ms * 200
    found_hi = None
    while hi < upper_bound_ts:
        hi = min(hi + step, upper_bound_ts)
        try:
            probe = exchange.fetch_ohlcv(symbol, timeframe, hi, 5)
        except Exception:
            probe = None
        time.sleep(0.5)
        if probe:
            found_hi = hi
            break
        lo = hi
        step *= 2
    if found_hi is None:
        return None
    for _ in range(12):
        if found_hi - lo <= tf_ms:
            break
        mid = lo + (found_hi - lo) // 2
        try:
            probe = exchange.fetch_ohlcv(symbol, timeframe, mid, 5)
        except Exception:
            probe = None
        time.sleep(0.5)
        if probe:
            found_hi = mid
        else:
            lo = mid
    return found_hi


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000, exchange_id: str = 'bitget',
                 since_ms: int = None) -> pd.DataFrame:
    """Laedt die letzten `limit` Kerzen fuer `symbol`/`timeframe` ueber die oeffentliche ccxt-API.

    Dieselbe Vorwaerts-Paginierung + Luecken-Umgehung wie in ltbbot/dnabot/probebot (Fleet-
    Standard, 2026-07-26 uebernommen, nachdem oraclebots eigene, komplexere Chunk-/Ganz-Fetch-
    Retry-Logik dieselbe Bitget-Datenluecke wiederholt NICHT umgehen konnte -- siehe
    _probe_next_available_ts()). `fetch_limit=200` pro Call ist der in mehreren Bots dieser Flotte
    produktiv bewaehrte Wert.

    `since_ms`: optionaler expliziter Startzeitpunkt (ms seit Epoch), z.B. fuer inkrementelle
    Updates ab dem letzten Cache-Stand (siehe fetch_ohlcv_incremental()) -- ueberschreibt die
    sonst aus `limit` berechnete Startzeit.
    """
    exchange = getattr(ccxt, exchange_id)({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})
    exchange.load_markets()

    timeframe_ms = exchange.parse_timeframe(timeframe) * 1000
    since = since_ms if since_ms is not None else exchange.milliseconds() - limit * timeframe_ms
    fetch_limit = 200

    all_ohlcv = []
    start_time = time.time()
    while since < exchange.milliseconds():
        try:
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since, fetch_limit)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"{symbol} {timeframe}: Rate-Limit erreicht ({e}), warte 5s...")
            time.sleep(5)
            continue
        except Exception as e:
            logger.error(f"{symbol} {timeframe}: Fehler beim Laden: {e}")
            time.sleep(1)
            break

        if not chunk:
            now_ms = exchange.milliseconds()
            if since >= now_ms - timeframe_ms:
                break  # Gegenwart erreicht -- die aktuell laufende Kerze existiert noch nicht, kein Fehler.
            logger.warning(f"{symbol} {timeframe}: Leere Antwort ab {pd.Timestamp(since, unit='ms', tz='UTC')} "
                           f"-- suche naechsten verfuegbaren Zeitpunkt (bekannte Bitget-Datenluecken)...")
            next_ts = _probe_next_available_ts(exchange, symbol, timeframe, since, now_ms, timeframe_ms)
            if next_ts is None:
                logger.warning(f"{symbol} {timeframe}: keine weiteren Daten bis 'jetzt' gefunden, "
                               f"breche ab ({len(all_ohlcv)}/{limit} Kerzen).")
                break
            logger.info(f"{symbol} {timeframe}: Daten wieder verfuegbar ab "
                        f"{pd.Timestamp(next_ts, unit='ms', tz='UTC')}.")
            since = next_ts
            continue

        all_ohlcv.extend(chunk)
        render_progress(f"{symbol} {timeframe}", len(all_ohlcv), limit, start_time)
        # +1ms statt +timeframe_ms: Bitgets `since` ist exklusiv (timestamp > since),
        # ein voller Timeframe-Schritt trifft exakt die naechste Kerze und ueberspringt sie.
        since = chunk[-1][0] + 1
        time.sleep(exchange.rateLimit / 1000)

    if all_ohlcv:
        finish_progress()

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
        logger.warning(f"{symbol} {timeframe}: {len(unexpected_gaps)} uebersprungene Luecke(n) in der "
                       f"Historie (siehe obige 'Daten wieder verfuegbar ab'-Zeilen fuer Details).")

    return df


def fetch_ohlcv_incremental(symbol: str, timeframe: str, min_candles: int, cache_path: str) -> pd.DataFrame:
    """Live-Cache fuer predict_next_barrier.py: erster Lauf holt die volle benoetigte Historie
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
        # Re-Confirm-Fenster: NICHT nur die letzte Kerze, sondern die letzten REFRESH_WINDOW
        # Kerzen. Grund (gefunden 2026-07-13 anhand sichtbar zu duenner/doji-artiger Kerzen im
        # Chart): jede Kerze bekommt beim alten "nur die letzte Zeile"-Ansatz GENAU EINEN Lauf
        # lang die Chance, korrigiert zu werden -- sobald eine neuere Kerze angehaengt wird,
        # faellt die vorherige aus dem Fenster und bleibt fuer immer eingefroren, selbst wenn sie
        # beim genau diesem einen Fetch zufaellig noch unfertig war (z.B. weil an einem Lauf
        # mehrere neue Kerzen auf einmal auftauchten). Mit einem Fenster von mehreren Kerzen
        # bekommt jede Kerze stattdessen mehrere Laeufe hintereinander die Chance, sich auf
        # ihren finalen Wert einzupendeln, bevor sie aus dem Fenster faellt.
        REFRESH_WINDOW = 5
        refresh_from_idx = max(0, len(cached) - REFRESH_WINDOW)
        since_ms = int(cached.index[refresh_from_idx].value // 1_000_000) - 1
        logger.info(f"{symbol} {timeframe}: Cache-Stand bis {cached.index[-1]} ({len(cached)} Kerzen), "
                    f"hole inkrementell ab since={pd.Timestamp(since_ms, unit='ms', tz='UTC')} "
                    f"(bestaetigt die letzten {len(cached) - refresh_from_idx} gecachten Kerzen erneut)...")
        # Luecke dynamisch abdecken statt fixem limit: ein veralteter/lange nicht gelaufener
        # Cache (z.B. nach Downtime) kann Tage hinter "jetzt" liegen -- bei feinen Timeframes
        # (15m) sind das schnell >800 Kerzen. Ein zu kleines limit wuerde dann still nur einen
        # Teil der Luecke schliessen und predict_next_barrier.py wuerde tagealte Kerzen als
        # "aktuellste" behandeln, ohne dass das auffaellt (gefunden 2026-07-25).
        gap_candles = int((exchange.milliseconds() - since_ms) / timeframe_ms) + 2
        fresh = fetch_ohlcv(symbol, timeframe, limit=max(min_candles, 50, gap_candles), since_ms=since_ms)
        if len(fresh) == 0:
            logger.warning(f"{symbol} {timeframe}: inkrementeller Fetch lieferte NICHTS (since="
                           f"{pd.Timestamp(since_ms, unit='ms', tz='UTC')}), nutze reinen Cache-Stand "
                           f"({cached.index[-1]}) unveraendert weiter.")
            df = cached
        else:
            logger.info(f"{symbol} {timeframe}: frischer Fetch liefert {len(fresh)} Kerze(n), "
                        f"{fresh.index[0]} bis {fresh.index[-1]}.")
            df = pd.concat([cached.iloc[:refresh_from_idx], fresh])
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


RESAMPLE_FREQ = {'1M': 'MS'}  # Kalender-Monatsanfang, passend zu Bitgets eigener '1M'-Konvention


def resample_ohlcv(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
    """Leitet eine groebere Zeitebene (aktuell nur '1M') aus einer feineren, bereits geladenen
    OHLCV-DataFrame (z.B. '1d') per Resampling ab, statt Bitgets eigenen Endpunkt fuer diese
    Zeitebene direkt abzufragen.

    Grund: Bitgets '1M'-Endpunkt lieferte auf einem VPS wiederholt (mehrfach in Folge, auch mit
    Ganz-Fetch-Retries) nur einen Bruchteil (1/50) der angefragten Historie, waehrend '1d' auf
    exakt derselben Maschine zuverlaessig die volle angefragte Menge lieferte -- ein
    maschinenspezifisches Rate-Limit-/Routing-Problem speziell fuer diesen einen Endpunkt
    (2026-07-26). Rechnerisch entspricht das Ergebnis den ueblichen OHLCV-Aggregationsregeln
    (open=erste, high=max, low=min, close=letzte, volume=Summe) -- inhaltlich aequivalent zu
    Bitgets eigener Monatskerze, nur ohne den fehleranfaelligen zusaetzlichen API-Call.
    """
    if target_timeframe not in RESAMPLE_FREQ:
        raise ValueError(f"resample_ohlcv: kein Frequenz-Mapping fuer '{target_timeframe}'.")
    resampled = df.resample(RESAMPLE_FREQ[target_timeframe]).agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum',
    })
    return resampled.dropna(subset=['open'])


def fetch_all_timeframes(symbol: str, timeframes: list, history_days: int, cache_dir: str = None,
                          use_cache: bool = True) -> dict:
    """Laedt OHLCV fuer alle Timeframes eines Symbols, mit optionalem Datei-Cache.

    Der volle Mehr-Timeframe-Fetch (v.a. 15m ueber hunderte Tage) dauert mehrere Minuten -- ein
    Cache erspart das erneute Fetchen bei wiederholten Trainings-Laeufen. Wird von
    train_barrier_model.py, optimize_barrier_model.py und show_results.py genutzt, damit alle
    denselben Cache treffen.

    use_cache=True (Standard): inkrementelles Update ueber fetch_ohlcv_incremental() -- laedt
    einen vorhandenen Cache und haengt nur die Kerzen seit dem letzten Stand an, genau wie bei
    der Live-Inferenz (predict_next_barrier.py). BUGFIX 2026-07-26: vorher wurde eine vorhandene
    Cache-Datei stattdessen 1:1 uebernommen, OHNE je neue Kerzen anzuhaengen -- "n" bei "Gecachte
    OHLCV-Daten frisch abrufen?" (run_pipeline.sh/optimize.sh) bedeutete dadurch stillschweigend
    "beliebig alten Cache fuer immer weiterverwenden" statt "schnell, aber trotzdem aktuell".
    Live beobachtet: Trainingsdaten blieben ueber Wochen auf demselben Stand eingefroren, ein
    kompletter Monat ohne Trades im Backtest, weil schlicht keine neueren Kerzen im Datensatz
    waren.
    use_cache=False: loescht einen vorhandenen Cache und erzwingt dadurch einen kompletten
    Neuabruf (langsamer, aber garantiert luecken-/altlastenfrei).

    '1M' wird NICHT direkt von Bitget abgefragt, sondern per resample_ohlcv() aus '1d'
    abgeleitet (siehe dortige Begruendung) -- '1d' wird dafuer automatisch mitgeladen, auch
    wenn es nicht explizit in `timeframes` angefragt wurde.
    """
    ohlcv_by_timeframe = {}
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    fetch_targets = [tf for tf in timeframes if tf != '1M']
    if '1M' in timeframes and '1d' not in fetch_targets:
        fetch_targets.append('1d')

    for tf in fetch_targets:
        limit = max(50, int(history_days * 24 * 60 / TIMEFRAME_MINUTES[tf]))
        cache_path = os.path.join(cache_dir, f"ohlcv_{safe_symbol}_{tf}_{limit}.pkl") if cache_dir else None

        if not use_cache and cache_path and os.path.exists(cache_path):
            os.remove(cache_path)

        if cache_path:
            df = fetch_ohlcv_incremental(symbol, tf, limit, cache_path)
        else:
            logger.info(f"Lade {symbol} {tf} ({limit} Kerzen, ~{history_days} Tage)...")
            df = fetch_ohlcv(symbol, tf, limit=limit)
            logger.info(f"  -> {len(df)} Kerzen: {df.index[0]} bis {df.index[-1]}" if len(df) else "  -> keine Daten")
        ohlcv_by_timeframe[tf] = df

    if '1M' in timeframes:
        derived = resample_ohlcv(ohlcv_by_timeframe['1d'], '1M')
        logger.info(f"{symbol} 1M: {len(derived)} Kerzen aus '1d' abgeleitet (kein Bitget-Direktabruf) -- "
                    f"{derived.index[0]} bis {derived.index[-1]}" if len(derived) else f"{symbol} 1M: 0 Kerzen abgeleitet.")
        ohlcv_by_timeframe['1M'] = derived
        if '1d' not in timeframes:
            del ohlcv_by_timeframe['1d']

    return ohlcv_by_timeframe
