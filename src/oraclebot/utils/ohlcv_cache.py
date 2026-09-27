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


def update_cache(coin: str, anchor: str, cache_dir: str, tf: str = '1h', now: pd.Timestamp = None,
                 exchange=None, max_failures: int = 20) -> pd.DataFrame:
    """Aktualisiert den Cache fuer `coin` bis zur letzten abgeschlossenen Kerze und gibt ihn zurueck."""
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, coin, tf)
    tf_ms = _TF_MS[tf]
    now = now or pd.Timestamp.now(tz='UTC')
    last_closed_open = now.floor('h') - pd.Timedelta(milliseconds=tf_ms)   # Oeffnungszeit der letzten fertigen Kerze
    df = pd.read_pickle(path) if os.path.exists(path) else pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    since = int(pd.Timestamp(anchor, tz='UTC').timestamp() * 1000) if df.empty else int(df.index[-1].timestamp() * 1000) + 1
    end_ms = int(last_closed_open.timestamp() * 1000)
    if since > end_ms:
        return df
    ex = exchange or _exchange()
    symbol = f"{coin}/USDT:USDT"
    rows, fails, empty_hops = [], 0, 0
    while since <= end_ms:
        try:
            chunk = ex.fetch_ohlcv(symbol, tf, since, 200)
            fails = 0
        except Exception as e:  # jeder Fehler wird wiederholt, nie still abgebrochen
            fails += 1
            if fails > max_failures:
                raise OhlcvFetchError(f"{symbol} {tf}: Abruf ab {pd.Timestamp(since, unit='ms', tz='UTC')} "
                                      f"nach {max_failures} Versuchen gescheitert: {e}")
            time.sleep(min(30, 2 * fails))
            continue
        if not chunk:
            # echte Bitget-Datenluecke oder noch nicht gelistet: vorwaerts springen (wie robust_fetch)
            empty_hops += 1
            since += 200 * tf_ms
            continue
        rows.extend(c for c in chunk if c[0] <= end_ms)
        since = chunk[-1][0] + 1
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
        logger.info(f"{coin} {tf}: +{len(new)} Kerzen, Cache bis {df.index[-1]} ({len(df)} gesamt, leere Spruenge {empty_hops})")
    if df.empty or df.index[-1] < last_closed_open - pd.Timedelta(hours=3):
        raise OhlcvFetchError(f"{symbol} {tf}: Cache endet bei {df.index[-1] if not df.empty else '-'}, "
                              f"erwartet bis {last_closed_open} -- nicht handeln.")
    return df
