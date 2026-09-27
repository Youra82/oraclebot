# src/oraclebot/utils/ohlcv_cache.py
# Append-only 1h-Kerzen-Cache ab einem festen Startdatum (`anchor`) fuer den Trend-Pool. Renko-Ketten sind
# pfadabhaengig: Live und Backtest muessen die Kette ab demselben Startpunkt auf denselben Kerzen bauen,
# sonst weichen die Signale ab (Fund 2026-09-26/27, siehe strategy/trend_pool.py). Deshalb:
#   - nur ABGESCHLOSSENE Kerzen werden gespeichert (die laufende nie),
#   - bestehende Kerzen werden nie ueberschrieben (append-only),
#   - jeder Abruffehler wird wiederholt; bricht der Abruf endgueltig ab, wird NICHT stillschweigend mit einer
#     verkuerzten Historie weitergearbeitet (Lehre aus den Pagination-Bugs anderer Bots), sondern ein Fehler
#     geworfen.
import logging
import os
import time

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)
_TF_MS = {'1h': 3600_000}


class OhlcvFetchError(RuntimeError):
    pass


def _exchange():
    return ccxt.bitget({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})


def cache_path(cache_dir: str, coin: str, tf: str = '1h') -> str:
    return os.path.join(cache_dir, f"trend_{tf}_{coin}.pkl")


def _fetch_window(ex, market_id: str, start_ms: int, end_ms: int, tf_ms: int, max_failures: int,
                  empty_retries: int, sleep_on_empty: float) -> list:
    """Alle Kerzen mit Oeffnungszeit in [start_ms, end_ms] (hoechstens 200) ueber Bitgets rohen Endpunkt
    `history-candles` mit festem Start UND Ende.

    Fund 2026-09-27: ccxt.fetch_ohlcv() der auf dem VPS gepinnten ccxt==4.3.5 liefert fuer Startzeitpunkte
    zwischen 2026-07-09 und 2026-07-30 bei Bitget DETERMINISTISCH leere Antworten (Endpunkt-Umschaltung
    innerhalb von ccxt); ein blinder Sprung verlor dadurch 600 Kerzen je Coin, Wiederholen half nicht. Der rohe
    Endpunkt liefert das Fenster vollstaendig (gegen ccxt 4.3.5 in isolierter venv verifiziert). Weil jedes
    Fenster exakt festgelegt ist, bedeutet eine (nach Wiederholungen) leere Antwort wirklich: keine Daten."""
    fails = 0
    for attempt in range(empty_retries + 1):
        try:
            r = ex.publicMixGetV2MixMarketHistoryCandles({
                'symbol': market_id, 'productType': 'USDT-FUTURES', 'granularity': '1H',
                'startTime': str(start_ms), 'endTime': str(end_ms + tf_ms), 'limit': '200'})
        except Exception as e:
            fails += 1
            if fails > max_failures:
                raise OhlcvFetchError(f"{market_id}: Abruf {pd.Timestamp(start_ms, unit='ms', tz='UTC')} "
                                      f"nach {max_failures} Versuchen gescheitert: {e}")
            time.sleep(min(30, 2 * fails))
            continue
        rows = [[int(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5])]
                for c in (r.get('data') or []) if start_ms <= int(c[0]) <= end_ms]
        if rows:
            return sorted(rows)
        if attempt < empty_retries:
            time.sleep(sleep_on_empty * (attempt + 1))
    return []


def update_cache(coin: str, anchor: str, cache_dir: str, tf: str = '1h', now: pd.Timestamp = None,
                 exchange=None, max_failures: int = 20, empty_retries: int = 3, sleep_on_empty: float = 2.0) -> pd.DataFrame:
    """Aktualisiert den Cache fuer `coin` bis zur letzten abgeschlossenen Kerze und gibt ihn zurueck."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, coin, tf)
    tf_ms = _TF_MS[tf]
    now = now or pd.Timestamp.now(tz='UTC')
    last_closed_open = now.floor('h') - pd.Timedelta(milliseconds=tf_ms)   # Oeffnungszeit der letzten fertigen Kerze
    df = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    since = int(pd.Timestamp(anchor, tz='UTC').timestamp() * 1000) if df.empty else int(df.index[-1].timestamp() * 1000) + tf_ms
    end_ms = int(last_closed_open.timestamp() * 1000)
    if since > end_ms:
        return df
    ex = exchange or _exchange()
    market_id = f"{coin}USDT"
    rows, empty_windows = [], 0
    while since <= end_ms:
        w_end = min(since + 199 * tf_ms, end_ms)
        chunk = _fetch_window(ex, market_id, since, w_end, tf_ms, max_failures, empty_retries, sleep_on_empty)
        if not chunk:
            empty_windows += 1
        rows.extend(chunk)
        since = w_end + tf_ms
        time.sleep(0.05)
    if rows:
        new = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        new.index = pd.to_datetime(new.pop('ts'), unit='ms', utc=True)
        new = new[~new.index.duplicated(keep='first')]
        if not df.empty:
            new = new[new.index > df.index[-1]]
        df = pd.concat([df, new]).sort_index() if not df.empty else new.sort_index()
        df = df.astype(float)
        tmp = path + '.tmp'
        df.to_pickle(tmp)
        os.replace(tmp, path)
        gaps = int((df.index.to_series().diff() > pd.Timedelta(milliseconds=tf_ms) * 1.5).sum())
        logger.info(f"{coin} {tf}: +{len(new)} Kerzen, Cache bis {df.index[-1]} ({len(df)} gesamt, leere Fenster {empty_windows}, Luecken im Cache {gaps})")
    if df.empty or df.index[-1] < last_closed_open - pd.Timedelta(hours=3):
        raise OhlcvFetchError(f"{market_id} {tf}: Cache endet bei {df.index[-1] if not df.empty else '-'}, "
                              f"erwartet bis {last_closed_open} -- nicht handeln.")
    return df
