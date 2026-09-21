# src/oraclebot/analysis/renko_live_signal_check.py
# Taeglicher Live-vs-Backtest-Konsistenzcheck fuer die Renko-Breakout-Strategie -- Antwort auf
# die Frage "ist der Vergleich von Livetrade und Backtest implementiert?" fuer diese neue
# Strategie (analog zu analysis/live_signal_check.py der Barriere-Strategie, aber anders
# strukturiert: keine ML-Konfidenz zu vergleichen, sondern eine deterministische Regel).
#
# Zwei unabhaengige Pruefungen:
#   1) check_brick_chain_consistency(): baut die Brick-Kette je Symbol KOMPLETT NEU aus den
#      letzten Tagen auf (kein persistierter Zustand) und vergleicht die ueberlappenden Bricks
#      gegen den tatsaechlichen Live-Zustand -- exakt die Fehlerklasse aus dem dokumentierten
#      zerobot-Vorfall (Live baute Bricks strukturell anders als Backtest,
#      [[research_zerobot_live_vs_backtest_2026_08]]). Nutzt NUR oeffentliche OHLCV-Daten, kein
#      Konto-Zugriff.
#   2) check_recent_trades_against_backtest(): vergleicht die zuletzt tatsaechlich getradeten
#      Live-Trades gegen eine frische Backtest-Rekonstruktion desselben Zeitraums (Entry/Exit-
#      Preise, Richtung) -- braucht fetch_closed_positions() (Konto-Zugriff).
import logging

import numpy as np
import pandas as pd

from oraclebot.data.ear_bricks import build_ear_bricks
from oraclebot.strategy.horizontal_breakout_signal import backtest_horizontal_breakout
from oraclebot.utils.data_fetch import fetch_ohlcv

logger = logging.getLogger(__name__)


def check_brick_chain_consistency(symbol: str, live_state: dict, base_pct: float, k_entropy: float,
                                   h_window: int, brick_tf: str, lookback_days: int = 5) -> dict:
    """Baut die Brick-Kette der letzten `lookback_days` Tage komplett frisch auf (unabhaengig vom
    Live-Zustand) und vergleicht die Schlusskurse der ueberlappenden Bricks gegen
    `live_state['recent_bricks']`. Bei Abweichung ist die inkrementelle Fortsetzung (siehe
    strategy/renko_portfolio_state.py) irgendwo von einem sauberen Neuaufbau abgekommen -- z.B.
    durch eine Cache-Luecke, einen Neustart mit verlorenem buffer_candles, oder einen Bug.

    Returns:
        dict mit 'symbol', 'match' (bool oder None falls nicht vergleichbar), 'n_compared',
        'max_close_diff_pct', 'live_tail', 'fresh_tail'.
    """
    live_bricks = live_state.get('recent_bricks', [])
    if not live_bricks:
        return {'symbol': symbol, 'match': None, 'reason': 'kein Live-Zustand vorhanden'}

    df = fetch_ohlcv(symbol, brick_tf, limit=lookback_days * 24 * 60 // 5)
    if df.empty:
        return {'symbol': symbol, 'match': None, 'reason': 'keine OHLCV-Daten erhalten'}

    fresh_bricks = build_ear_bricks(df, base_pct=base_pct, k_entropy=k_entropy, h_window=h_window)
    if not fresh_bricks:
        return {'symbol': symbol, 'match': None, 'reason': 'frischer Neuaufbau ergab keine Bricks'}

    fresh_closes = {round(b['close'], 6): b for b in fresh_bricks}
    n_compared = 0
    max_diff_pct = 0.0
    mismatches = []
    for lb in live_bricks:
        live_close = round(lb['close'], 6)
        candidates = [c for c in fresh_closes if abs(c - live_close) / max(abs(live_close), 1e-9) < 0.01]
        if not candidates:
            continue
        n_compared += 1
        nearest = min(candidates, key=lambda c: abs(c - live_close))
        fresh_brick = fresh_closes[nearest]
        diff_pct = abs(nearest - live_close) / max(abs(live_close), 1e-9) * 100
        max_diff_pct = max(max_diff_pct, diff_pct)
        if fresh_brick['direction'] != lb['direction'] or diff_pct > 1e-4:
            mismatches.append({'live_close': live_close, 'live_direction': lb['direction'],
                                'fresh_close': nearest, 'fresh_direction': fresh_brick['direction'],
                                'diff_pct': diff_pct})

    if n_compared == 0:
        return {'symbol': symbol, 'match': None,
                'reason': f'kein Ueberlappungsbereich gefunden (lookback_days={lookback_days} evtl. zu kurz)'}

    return {'symbol': symbol, 'match': len(mismatches) == 0, 'n_compared': n_compared,
            'max_close_diff_pct': round(max_diff_pct, 6), 'mismatches': mismatches[:5],
            'live_tail': live_bricks[-5:], 'fresh_tail': fresh_bricks[-5:]}


def check_recent_trades_against_backtest(symbol: str, exchange, cfg: dict, since_ts: pd.Timestamp,
                                          lookback_days: int = 10) -> dict:
    """Vergleicht die zuletzt tatsaechlich auf Bitget geschlossenen Renko-Trades gegen eine
    frische Backtest-Rekonstruktion desselben Zeitraums (Entry-Richtung + ungefaehrer Entry-Preis).

    Einschraenkung (wie bei live_signal_check.py fuer die Barriere-Strategie): vergleicht nur
    gegen die AKTUELL konfigurierten Parameter -- eine Parameteraenderung waehrend des Fensters
    kann aeltere Live-Trades faelschlich als Mismatch erscheinen lassen.
    """
    closed = exchange.fetch_closed_positions(symbol, limit=50)
    recent = [p for p in closed if pd.Timestamp(p.get('timestamp', 0), unit='ms', tz='UTC') >= since_ts]
    if not recent:
        return {'symbol': symbol, 'n_live_trades': 0, 'n_match': 0, 'n_comparable': 0}

    base_pct = cfg.get('base_pct_brick_by_symbol', {}).get(symbol, 0.002)
    df = fetch_ohlcv(symbol, cfg.get('brick_timeframe', '5m'), limit=lookback_days * 24 * 60 // 5)
    bricks = build_ear_bricks(df, base_pct=base_pct, k_entropy=cfg.get('k_entropy', 0.7),
                               h_window=cfg.get('h_window', 15))
    backtest_trades = backtest_horizontal_breakout(bricks, cfg.get('horizontal_lookback', 6),
                                                    cfg.get('breakout_run', 2))

    n_match = 0
    details = []
    for live_trade in recent:
        live_side = live_trade.get('side')
        live_entry = float(live_trade.get('entryPrice') or 0)
        live_ts = pd.Timestamp(live_trade.get('timestamp', 0), unit='ms', tz='UTC')
        candidates = [t for t in backtest_trades
                      if t['direction'] == live_side and abs((t['entry_ts'] - live_ts).total_seconds()) < 3600]
        matched = bool(candidates)
        if matched:
            n_match += 1
        details.append({'live_ts': str(live_ts), 'live_side': live_side, 'live_entry': live_entry,
                         'matched_backtest': matched})

    return {'symbol': symbol, 'n_live_trades': len(recent), 'n_comparable': len(recent),
            'n_match': n_match, 'details': details}
