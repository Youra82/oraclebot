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

FULL_FETCH_RETRY_LIMIT = 2
FULL_FETCH_SHORTFALL_RATIO = 0.5


def fetch_ohlcv(symbol: str, timeframe: str, limit: int = 1000, exchange_id: str = 'bitget',
                 since_ms: int = None, _retry_count: int = 0) -> pd.DataFrame:
    """Laedt die letzten `limit` Kerzen fuer `symbol`/`timeframe` ueber die oeffentliche ccxt-API.

    Paginiert vorwaerts ab einem berechneten Startzeitpunkt (statt rueckwaerts anhand
    einer angenommenen Chunk-Groesse) -- Bitget liefert pro `since`-Request oft deutlich
    weniger Kerzen als angefragt (~90-100 statt 1000), was bei rueckwaerts-Paginierung
    stillschweigend Luecken in der Historie erzeugen wuerde.

    `since_ms`: optionaler expliziter Startzeitpunkt (ms seit Epoch), z.B. fuer inkrementelle
    Updates ab dem letzten Cache-Stand (siehe fetch_ohlcv_incremental()) -- ueberschreibt die
    sonst aus `limit` berechnete Startzeit.

    `_retry_count`: intern (nicht fuer Aufrufer gedacht) -- zaehlt Komplett-Fetch-Wiederholungen,
    siehe die Ganz-Fetch-Retry-Logik am Ende der Funktion.
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
    start_time = time.time()
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
        render_progress(f"{symbol} {timeframe}", len(all_ohlcv), limit, start_time)
        # +1ms statt +timeframe_ms: Bitgets `since` ist exklusiv (timestamp > since),
        # ein voller Timeframe-Schritt trifft exakt die naechste Kerze und ueberspringt sie.
        since = all_ohlcv[-1][0] + 1
        if since >= now_ms or len(all_ohlcv) >= limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    if all_ohlcv:
        finish_progress()

    # Ganz-Fetch-Retry bei drastischem Rueckstand: die per-Chunk-Retries oben (3x, kurzer
    # Backoff) fangen einzelne transiente Ausreisser ab, helfen aber nichts, wenn Bitget fuer
    # eine komplette Session konsequent nur einen Bruchteil der Historie liefert (beobachtet
    # 2026-07-26: 1M brach zweimal in Folge auf demselben VPS bei 1/50 Kerzen ab, waehrend
    # derselbe Request von einem anderen Rechner aus zuverlaessig 50+ lieferte -- sieht nach
    # einem laenger anhaltenden Rate-Limit/Routing-Problem aus, nicht nach einem einzelnen
    # Hickser). Nur wenn `since` noch WEIT von "jetzt" entfernt ist -- sonst ist ein knappes
    # Ergebnis normal (z.B. inkrementelle Nachfrage mit nur wenigen neuen Kerzen seit dem
    # letzten Cache-Stand, kein Fehler).
    caught_up_to_now = (now_ms - since) < (timeframe_ms * 2)
    if (not caught_up_to_now and len(all_ohlcv) < limit * FULL_FETCH_SHORTFALL_RATIO
            and _retry_count < FULL_FETCH_RETRY_LIMIT):
        wait_s = 15 * (_retry_count + 1)
        logger.warning(f"{symbol} {timeframe}: nur {len(all_ohlcv)}/{limit} Kerzen erhalten, "
                        f"`since` noch weit von jetzt entfernt -- vermutlich anhaltendes "
                        f"Rate-Limit/Routing-Problem statt einzelnem Hickser. Kompletter "
                        f"Neuversuch {_retry_count + 1}/{FULL_FETCH_RETRY_LIMIT} in {wait_s}s...")
        time.sleep(wait_s)
        return fetch_ohlcv(symbol, timeframe, limit=limit, exchange_id=exchange_id,
                            since_ms=since_ms, _retry_count=_retry_count + 1)

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
