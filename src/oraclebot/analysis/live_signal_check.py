# src/oraclebot/analysis/live_signal_check.py
# Taeglicher Live-vs-Modell-Signalvergleich: holt echte geschlossene Positionen von Bitget und
# prueft fuer jede, ob das AKTUELL deployte Modell (dieselbe Feature-Rekonstruktion wie die Live-
# Inferenz, siehe data/live_features.py) zum damaligen Referenzzeitpunkt dieselbe Richtung mit
# ausreichender Konfidenz vorhergesagt haette. Ausgangspunkt: eine manuelle Recherche (2026-09-12)
# fand ueber eine Git-Historie-Rekonstruktion echte Live-vs-Offline-Abweichungen, von denen ein
# Teil auf den bereits gefixten 1d-Cache-Bug (Commit 1c9ebeb) zurueckging -- dieses Modul
# automatisiert genau diesen Check fuer die Zukunft, damit sowas nicht erst bei einer manuellen
# Nachfrage auffaellt.
#
# WICHTIG: anders als die Git-Historie-Rekonstruktion prueft dieses Modul NUR gegen das aktuell
# deployte Modell/Config -- sinnvoll fuer den taeglichen Cron (Modell aendert sich nicht binnen
# 24h), aber bei einem Retrain am selben Tag koennen die aeltesten Trades des Fensters faelschlich
# als Mismatch erscheinen, obwohl sie unter dem VORHERIGEN Modell korrekt waren.
import json
import logging
import os

import pandas as pd

from oraclebot.data.live_features import (FeatureReconstructionError, MIN_CANDLES_BY_TF, TIMEFRAME_MINUTES,
                                           build_reference_feature_vector)
from oraclebot.strategy.barrier_signal import compute_barrier_signal
from oraclebot.utils.config import CONFIGS_DIR, config_filename

logger = logging.getLogger(__name__)


def _load_backtest_reference(symbol: str, reference_timeframe: str) -> dict:
    """Liest NUR den '_meta'-Block der Strategie-Config (load_barrier_config() entfernt ihn --
    siehe config.py), um eine Backtest-Erwartung fuer den taeglichen Bericht zu haben: 'wie gut
    war das Modell beim letzten Training auf dem Out-of-Sample-Split?'. Bewusst NICHT der volle
    Anti-Martingale-Backtest aus show_results.py (der braucht den lokalen Trainings-Datensatz-
    Cache, der auf dem VPS i.d.R. NICHT vorhanden ist -- siehe README 'laufen NICHT auf dem VPS')
    -- dieser _meta-Wert ist dagegen git-getrackt und damit ueberall verfuegbar, wo auch das
    Modell selbst liegt.

    HINWEIS: 'confirmation_oos_accuracy' ist die rohe Klassifikations-Genauigkeit des Modells auf
    dem OOS-Split, NICHT die tatsaechliche Handels-Winrate nach min_confidence-Filterung (die liegt
    typischerweise hoeher, siehe README-Vergleichstabelle: 67.5% Accuracy vs. 76.6% Winrate beim
    aktuellen Modell) -- als Richtwert fuer Modell-Drift trotzdem aussagekraeftig, nur nicht 1:1
    mit der Live-Winrate vergleichbar.

    Returns: {} falls keine Config-Datei oder kein '_meta'-Block existiert.
    """
    config_path = os.path.join(CONFIGS_DIR, config_filename(symbol, reference_timeframe))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    return cfg.get('_meta', {})


def _reference_period_start(ctime: pd.Timestamp, reference_tf: str) -> pd.Timestamp:
    """Die Referenzkerze, die live_trade.py fuer eine Position genutzt hat: die zuletzt
    ABGESCHLOSSENE Kerze zum Ausfuehrungszeitpunkt -- also eine volle Periode VOR dem
    Zeitfenster, in dem die Order tatsaechlich platziert wurde (siehe barrier_gate.py: Ausfuehrung
    in den ersten 30 Minuten nach einer Periodengrenze, aber die Kerze DIESER Periode ist zu dem
    Zeitpunkt noch nicht abgeschlossen)."""
    period_minutes = TIMEFRAME_MINUTES[reference_tf]
    return ctime.floor(f"{period_minutes}min") - pd.Timedelta(minutes=period_minutes)


def fetch_recent_live_trades(exchange, symbol: str, reference_tf: str, since_ts: pd.Timestamp,
                              limit: int = 100) -> list:
    """Holt geschlossene Live-Positionen, deren SCHLIESSUNG seit `since_ts` liegt, und leitet je
    Trade die Referenzkerze ab.

    Bugfix 2026-09-20 (Live-Beobachtung): urspruenglich wurde gegen `ctime` (Eroeffnungszeit)
    gefiltert, nicht gegen `utime` (Schliesszeit). Bei diesem Barriere-Modell bleiben Positionen
    haeufig laenger als 24h offen, bevor SL/TP greift -- eine Position, die vor >24h eroeffnet
    wurde, aber INNERHALB der letzten 24h schloss, fiel dadurch systematisch durchs 24h-Fenster
    des taeglichen Checks. Symptom live beobachtet: 7 echte Trade-Eroeffnungen ueber eine Woche
    (aus dem "kein Stacking"-Verhalten zweifelsfrei mind. 6 zugehoerige Schliessungen), aber JEDER
    einzelne taegliche Bericht meldete "keine geschlossenen Live-Trades". Fuer die Referenzkerzen-
    Ableitung (welche Kerze die Entry-Entscheidung ausloeste) bleibt `ctime` weiterhin richtig --
    nur der since_ts-Filter selbst betraf die Eroeffnungszeit, nicht die fuer diesen Bericht
    eigentlich relevante Schliesszeit.

    Returns: Liste von Dicts {ctime, utime, ref_ts, direction, pnl, net_profit, open_price,
    close_price}, chronologisch nach Schliesszeit sortiert.
    """
    positions = exchange.fetch_closed_positions(symbol, limit=limit)
    trades = []
    for p in positions:
        info = p.get('info', {})
        try:
            ctime = pd.Timestamp(int(info['ctime']), unit='ms', tz='UTC')
            utime = pd.Timestamp(int(info['utime']), unit='ms', tz='UTC')
        except (KeyError, ValueError, TypeError):
            continue
        if utime < since_ts:
            continue
        trades.append({
            'ctime': ctime,
            'utime': utime,
            'ref_ts': _reference_period_start(ctime, reference_tf),
            'direction': 'long' if info.get('holdSide') == 'long' else 'short',
            'pnl': float(info.get('pnl', 0.0)),
            'net_profit': float(info.get('netProfit', 0.0)),
            'open_price': float(info.get('openAvgPrice', 0.0)),
            'close_price': float(info.get('closeAvgPrice', 0.0)),
        })
    trades.sort(key=lambda t: t['utime'])
    return trades


def build_signal_comparison(barrier_cfg: dict, predictor, exchange, artifacts_dir: str,
                             since_ts: pd.Timestamp, now_utc: pd.Timestamp = None) -> dict:
    """Baut den Vergleichsbericht: fuer jeden Live-Trade seit `since_ts`, was haette das aktuell
    deployte Modell zum damaligen Referenzzeitpunkt gesagt?

    Returns: dict mit 'since', 'generated_at', 'n_trades', 'n_comparable', 'n_match',
    'n_mismatch', 'n_live_wins', 'n_live_losses', 'live_win_rate', 'backtest_oos_accuracy',
    'backtest_walk_forward_mean' (beide None falls keine Strategie-Config-Meta vorhanden), 'trades'
    (Liste von Dicts, inkl. Fehlerfall 'error' statt Modell-Feldern).
    """
    symbol = barrier_cfg['symbol']
    reference_tf = barrier_cfg['reference_timeframe']
    context_tfs = barrier_cfg.get('context_timeframes', [])
    min_confidence = barrier_cfg.get('min_confidence', 0.60)
    barrier_pct = barrier_cfg.get('barrier_pct', 1.0)
    now_utc = now_utc or pd.Timestamp.now(tz='UTC')

    live_trades = fetch_recent_live_trades(exchange, symbol, reference_tf, since_ts)

    # Der Live-Cache (ohlcv_live_*.pkl) waechst nur vorwaerts (siehe live_features._ensure_min_candles) --
    # fuer einen Trade von vor mehreren Tagen muss der Cache entsprechend weit zurueckreichen. Ohne
    # diese explizite Mindesttiefe wuerde ein frischer/schlanker Cache (z.B. auf einer neu
    # eingerichteten Maschine) faelschlich "kein Kontext gefunden" fuer aeltere Trades melden.
    # Puffer je Zeitebene aus MIN_CANDLES_BY_TF (nicht ein fixer Wert): ein pauschaler Puffer, der
    # fuer 15m/1h/4h/1d passt (ema_window=50 dort), waere fuer 1M/1w um ein Vielfaches zu gross --
    # deren eigenes Warmup ist deutlich kuerzer (feature_settings_by_timeframe: ema_window=6/12).
    # Ein zu grosser Puffer bei einem KALTEN Cache-Neuaufbau (z.B. frische Maschine) kann bei sehr
    # groben Zeitebenen sogar eine Bitget-API-Grenze reissen (>90 Tage Zeitfenster pro Anfrage bei
    # '1w', gefunden 2026-09-12) -- deshalb bewusst zurueckhaltend dimensioniert.
    #
    # Bugfix 2026-09-20: die Spanne MUSS von der aeltesten tatsaechlich benoetigten `ref_ts`
    # ausgehen, nicht von `since_ts` -- seit fetch_recent_live_trades() nach Schliesszeit
    # (utime) statt Eroeffnungszeit (ctime) filtert (siehe dortiger Bugfix), kann eine lange offen
    # gebliebene Position eine `ref_ts` weit VOR `since_ts` haben (Live-Fund: eine Position blieb
    # >30h offen). Mit der alten, nur an since_ts orientierten Spanne waere der Cache fuer genau
    # die Trades zu duenn geblieben, die dieser Bugfix erst sichtbar macht.
    span_minutes = max((now_utc - min(t['ref_ts'] for t in live_trades)).total_seconds() / 60, 0) if live_trades else 0
    ref_min_candles = int(span_minutes / TIMEFRAME_MINUTES[reference_tf]) + MIN_CANDLES_BY_TF.get(reference_tf, 120)
    context_min_candles = {
        tf: int(span_minutes / TIMEFRAME_MINUTES[tf]) + MIN_CANDLES_BY_TF.get(tf, 120) for tf in context_tfs
    }

    rows = []
    for t in live_trades:
        row = {
            'ctime': t['ctime'].isoformat(), 'ref_ts': t['ref_ts'].isoformat(),
            'live_direction': t['direction'], 'pnl': t['pnl'],
        }
        try:
            feat_result = build_reference_feature_vector(symbol, reference_tf, context_tfs, barrier_cfg,
                                                           artifacts_dir, ref_ts=t['ref_ts'], now_utc=now_utc,
                                                           min_candles=ref_min_candles,
                                                           context_min_candles=context_min_candles)
        except FeatureReconstructionError as e:
            row['error'] = str(e)
            rows.append(row)
            continue

        cls, conf = predictor.predict_one(feat_result['feature_row'])
        signal = compute_barrier_signal(cls, conf, feat_result['entry_price'],
                                         min_confidence=min_confidence, barrier_pct=barrier_pct)
        row['model_direction'] = signal['direction']
        row['model_confidence'] = conf
        row['match'] = (signal['direction'] == t['direction'])
        rows.append(row)

    comparable = [r for r in rows if 'error' not in r]
    matches = sum(1 for r in comparable if r['match'])
    live_wins = sum(1 for t in live_trades if t['pnl'] > 0)
    live_losses = sum(1 for t in live_trades if t['pnl'] <= 0)

    backtest_meta = _load_backtest_reference(symbol, reference_tf)

    return {
        'since': since_ts.isoformat(),
        'generated_at': now_utc.isoformat(),
        'n_trades': len(live_trades),
        'n_comparable': len(comparable),
        'n_match': matches,
        'n_mismatch': len(comparable) - matches,
        'n_live_wins': live_wins,
        'n_live_losses': live_losses,
        'live_win_rate': (live_wins / len(live_trades)) if live_trades else None,
        'backtest_oos_accuracy': backtest_meta.get('confirmation_oos_accuracy'),
        'backtest_walk_forward_mean': backtest_meta.get('walk_forward_mean'),
        'trades': rows,
    }


def format_telegram_report(report: dict, symbol: str) -> str:
    n = report['n_trades']
    if n == 0:
        return f"oraclebot Signal-Check: keine geschlossenen Live-Trades seit {report['since'][:16]}."

    lines = [
        f"oraclebot Signal-Check ({symbol})",
        f"Zeitraum: seit {report['since'][:16]}",
        f"Live-Trades: {n} | Winrate: {report['live_win_rate']:.1%}" if report['live_win_rate'] is not None
        else f"Live-Trades: {n}",
        f"Signal-Abgleich mit aktuellem Modell: {report['n_match']}/{report['n_comparable']} Match",
    ]
    oos_acc = report.get('backtest_oos_accuracy')
    if oos_acc is not None:
        lines.append(f"Modell-OOS-Genauigkeit (letztes Training): {oos_acc:.1%} "
                      f"(Klassifikation, nicht 1:1 die Handels-Winrate)")
    if report['n_mismatch'] > 0:
        lines.append(f"⚠ {report['n_mismatch']} Mismatch -- Live-Richtung weicht vom aktuellen "
                      f"Modell fuer diese Referenzkerze ab (siehe Log fuer Details).")
    skipped = n - report['n_comparable']
    if skipped > 0:
        lines.append(f"({skipped} Trade(s) ohne Feature-Vergleich -- siehe Log.)")
    return "\n".join(lines)


def append_report_to_log(report: dict, log_path: str) -> None:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(report) + '\n')
    logger.info(f"Signal-Vergleichsbericht angehaengt: {log_path}")
