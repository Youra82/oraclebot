# src/oraclebot/data/live_features.py
# Gemeinsame Feature-Rekonstruktion fuer eine Referenzkerze (Referenz-Timeframe + Kontext-
# Timeframes) ueber den Live-Cache -- extrahiert aus predict_next_barrier.py (2026-09-12), damit
# der taegliche Signal-Vergleich (analysis/live_signal_check.py) exakt denselben Code nutzt wie
# die echte Live-Inferenz. Zwei Code-Pfade fuer dieselbe Feature-Berechnung sind genau das Muster,
# das beim 1d-Cache-Bug (Commit 1c9ebeb) zu einer Live-vs-Offline-Feature-Differenz fuehrte --
# eine geteilte Funktion verhindert, dass das erneut unbemerkt auseinanderlaeuft (siehe auch
# feedback zu ltbbot: neue Signal-/Filter-Logik immer als EINE geteilte Funktion bauen).
import os

import pandas as pd

from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.utils.data_fetch import fetch_ohlcv_incremental, resample_ohlcv

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}
# Genug Kerzen fuers laengste Feature-Warmup (EMA-50/MACD) je Timeframe.
MIN_CANDLES_BY_TF = {'1M': 60, '1w': 60, '1d': 120, '4h': 120, '1h': 120, '15m': 120}


def _drop_incomplete_last_candle(df: pd.DataFrame, timeframe: str, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Das Modell hat im Training nur abgeschlossene Kerzen gesehen (siehe No-Lookahead-Regel in
    barrier_targets.py) -- die zuletzt gecachte Kerze eines Timeframes kann noch laufen."""
    if df.empty:
        return df
    last_open = df.index[-1]
    close_time = last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return df.iloc[:-1] if now_utc < close_time else df


class FeatureReconstructionError(Exception):
    pass


def _ensure_min_candles(min_candles: int, cache_path: str) -> None:
    """fetch_ohlcv_incremental() haengt nur neue Kerzen ans Cache-Ende an -- es backfillt NIE
    rueckwirkend, egal welches min_candles man uebergibt (siehe dessen Docstring). Ein Cache, der
    urspruenglich mit einem kleineren min_candles angelegt wurde (z.B. der schlanke Live-Signal-
    Cache, siehe MIN_CANDLES_BY_TF), reicht deshalb u.U. nicht weit genug zurueck fuer eine
    retrospektive Pruefung (build_signal_comparison() fuer einen mehrere Tage alten Trade).
    Loescht die Cache-Datei, wenn sie zu wenige Kerzen enthaelt, damit der naechste
    fetch_ohlcv_incremental()-Aufruf einen kompletten Neuabruf mit ausreichender Tiefe macht --
    betrifft NUR die Cache-DATEI (identisch zu den Datenpunkten, die Bitget public liefert),
    nicht die Positions-/Trade-Historie selbst."""
    if not os.path.exists(cache_path):
        return
    try:
        cached = pd.read_pickle(cache_path)
    except Exception:
        cached = pd.DataFrame()
    if len(cached) < min_candles:
        os.remove(cache_path)


def build_reference_feature_vector(symbol: str, reference_tf: str, context_tfs: list, barrier_cfg: dict,
                                    artifacts_dir: str, ref_ts: pd.Timestamp = None,
                                    now_utc: pd.Timestamp = None, min_candles: int = 120,
                                    context_min_candles: dict = None) -> dict:
    """Baut den vollen Feature-Vektor (Referenz-Block + je ein Block pro Kontext-Timeframe) fuer
    eine Referenzkerze, exakt wie beim Training (barrier_targets.build_barrier_examples) und bei
    der Live-Inferenz -- ueber denselben inkrementellen Live-Cache (ohlcv_live_*.pkl), den auch
    predict_next_barrier.py bei jedem Cron-Tick pflegt.

    Args:
        ref_ts: die gewuenschte Referenzkerzen-Zeit (muss eine bereits ABGESCHLOSSENE Kerze sein,
            z.B. eine vergangene, tatsaechlich gehandelte Referenzkerze). None = aktuellste
            abgeschlossene Kerze (Live-Inferenz-Modus, wie predict_next_barrier.py es nutzt).
        now_utc: fuer den None-Fall (aktuellste Kerze) noetig, um zu pruefen ob die letzte
            gecachte Kerze schon abgeschlossen ist. Ignoriert, wenn ref_ts explizit gesetzt ist
            (dann ist "abgeschlossen" implizit, da die Kerze in der Vergangenheit liegt).
        min_candles: Mindesttiefe fuer den Referenz-Timeframe-Cache (Standard: 120, wie die
            Live-Inferenz). Fuer eine retrospektive Pruefung eines mehrere Tage alten ref_ts MUSS
            das groesser gesetzt werden, sonst reicht ein schlanker/frischer Cache evtl. nicht
            weit genug zurueck (siehe _ensure_min_candles()).
        context_min_candles: optionale {timeframe: min_candles}-Overrides je Kontext-Timeframe,
            aus demselben Grund wie oben.

    Returns:
        dict mit: ref_ts, entry_price, feature_row (Liste), blocks (Liste von
        (timeframe, timestamp, values) -- fuer Diagnose/Logging, wie FEATURE-DUMP in
        predict_next_barrier.py).

    Raises:
        FeatureReconstructionError bei zu wenig Historie oder fehlendem Kontext.
    """
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')
    context_min_candles = context_min_candles or {}

    cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_{reference_tf}.pkl")
    _ensure_min_candles(min_candles, cache_path)
    df = fetch_ohlcv_incremental(symbol, reference_tf, min_candles=min_candles, cache_path=cache_path)
    if ref_ts is None:
        df = _drop_incomplete_last_candle(df, reference_tf, now_utc)

    feat = compute_features(df, **barrier_cfg['feature_settings'])
    if len(feat) == 0:
        raise FeatureReconstructionError(
            f"compute_features() lieferte keine Zeilen fuer {reference_tf} (zu wenig Historie fuer Warmup).")

    if ref_ts is None:
        ref_ts = feat.index[-1]
    elif ref_ts not in feat.index:
        raise FeatureReconstructionError(
            f"Referenzzeitpunkt {ref_ts} nicht im {reference_tf}-Feature-Index (zu alt fuer den "
            f"Live-Cache oder noch nicht abgeschlossen).")

    entry_price = float(df.loc[ref_ts, 'close'])
    feature_row = feat.loc[ref_ts, FEATURE_NAMES].tolist()
    blocks = [(reference_tf, ref_ts, list(feature_row))]

    feature_kwargs_by_timeframe = barrier_cfg.get('feature_settings_by_timeframe', {})
    for ctx_tf in context_tfs:
        # Siehe predict_next_barrier.py: bei gleichzeitigem '1d' + '1M' in context_tfs MUSS
        # derselbe (groessere) min_candles-Wert fuer beide Aufrufe gelten, sonst kappt der
        # kleinere Aufruf die Cache-Datei und zerstoert den 1M-Backfill.
        if ctx_tf == '1d' and '1M' in context_tfs:
            ctx_min_candles = barrier_cfg.get('history_days', 1000)
        else:
            ctx_min_candles = MIN_CANDLES_BY_TF.get(ctx_tf, 120)
        ctx_min_candles = max(ctx_min_candles, context_min_candles.get(ctx_tf, 0))

        if ctx_tf == '1M':
            d1_cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_1d.pkl")
            d1_min_candles = max(barrier_cfg.get('history_days', 1000), context_min_candles.get('1d', 0))
            _ensure_min_candles(d1_min_candles, d1_cache_path)
            d1_df = fetch_ohlcv_incremental(symbol, '1d', min_candles=d1_min_candles, cache_path=d1_cache_path)
            ctx_df = resample_ohlcv(d1_df, '1M')
        else:
            ctx_cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_{ctx_tf}.pkl")
            _ensure_min_candles(ctx_min_candles, ctx_cache_path)
            ctx_df = fetch_ohlcv_incremental(symbol, ctx_tf, min_candles=ctx_min_candles, cache_path=ctx_cache_path)
        if ref_ts is None:
            ctx_df = _drop_incomplete_last_candle(ctx_df, ctx_tf, now_utc)

        ctx_kwargs = {**barrier_cfg['feature_settings'], **feature_kwargs_by_timeframe.get(ctx_tf, {})}
        ctx_feat = compute_features(ctx_df, **ctx_kwargs)
        ctx_feat = ctx_feat[ctx_feat.index <= ref_ts]
        if len(ctx_feat) == 0:
            raise FeatureReconstructionError(
                f"Kontext-Timeframe {ctx_tf}: keine gueltige Kerze <= Referenzzeitpunkt {ref_ts}.")

        ctx_ts = ctx_feat.index[-1]
        ctx_values = ctx_feat.loc[ctx_ts, FEATURE_NAMES].tolist()
        blocks.append((ctx_tf, ctx_ts, ctx_values))
        feature_row += ctx_values

    return {'ref_ts': ref_ts, 'entry_price': entry_price, 'feature_row': feature_row, 'blocks': blocks}
