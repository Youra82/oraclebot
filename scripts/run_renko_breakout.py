# scripts/run_renko_breakout.py
# Live-Cron fuer die Portfolio-weite Renko-Breakout-Strategie (2026-09-21). Laeuft alle 5 Minuten
# (Brick-Timeframe), setzt je Symbol die EAR-Brick-Kette inkrementell fort (siehe
# strategy/renko_portfolio_state.py fuer das Kontinuitaets-Prinzip -- direkt aus dem
# dokumentierten zerobot-Vorfall abgeleitet, bei dem Live und Backtest die Kette strukturell
# unterschiedlich aufbauten) und haelt maximal EINE offene Position gleichzeitig ueber das
# gesamte Portfolio (erstes frisches Signal gewinnt, alle anderen werden in dieser Runde
# verworfen statt nachgeholt -- exakt die arbitrate()-Regel aus dem Backtest).
#
# enabled=false in settings.json::renko_breakout_settings ist ein globaler Kill-Switch: das
# Skript baut die Brick-Ketten dann NICHT weiter und platziert keine Orders (komplett inert),
# analog zum dnabot-Muster fuer neue, noch nicht freigegebene Strategien.
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from oraclebot.strategy.renko_portfolio_state import (detect_exit, detect_fresh_entry, load_state,
                                                        save_state, update_symbol_bricks)
from oraclebot.strategy.renko_live_trade import close_renko_position, open_renko_position, resolve_am_outcome
from oraclebot.utils.barrier_gate import check_barrier_gate, mark_barrier_run_complete
from oraclebot.utils.config import load_settings
from oraclebot.utils.data_fetch import fetch_all_timeframes, fetch_ohlcv_incremental
from oraclebot.utils.telegram import send_message

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
STATE_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'state')
BRICK_STATE_PATH = os.path.join(STATE_DIR, 'renko_breakout_bricks.json')
PORTFOLIO_STATE_PATH = os.path.join(STATE_DIR, 'renko_breakout_portfolio.json')
AM_STATE_PATH = os.path.join(STATE_DIR, 'renko_anti_martingale_state.json')
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')


def load_secrets(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_portfolio_state(path: str) -> dict:
    if not os.path.exists(path):
        return {'active_symbol': None, 'active_direction': None}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_portfolio_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def _safe_symbol(symbol: str) -> str:
    return symbol.replace('/', '_').replace(':', '_')


def fetch_new_closed_candles(symbol: str, timeframe: str, min_candles: int,
                              cache_dir: str, last_processed_ts, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Holt frische Kerzen ueber den bestehenden Live-Cache (data_fetch.fetch_ohlcv_incremental)
    und gibt NUR die Teilmenge zurueck, die (a) noch nicht verarbeitet wurde (Index >
    last_processed_ts) UND (b) bereits sicher ABGESCHLOSSEN ist (Kerzenende <= now_utc) -- die
    noch laufende, sich aendernde Kerze wird bewusst NICHT in die Brick-Kette aufgenommen (sonst
    koennte ein spaeter noch korrigierter Schlusskurs bereits einen falschen Brick erzeugt haben)."""
    cache_path = os.path.join(cache_dir, f"renko_ohlcv_{_safe_symbol(symbol)}_{timeframe}.pkl")
    df = fetch_ohlcv_incremental(symbol, timeframe, min_candles=min_candles, cache_path=cache_path)
    timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}[timeframe]
    closed_cutoff = now_utc - pd.Timedelta(minutes=timeframe_minutes)
    df = df[df.index <= closed_cutoff]
    if last_processed_ts is not None:
        df = df[df.index > pd.Timestamp(last_processed_ts)]
    return df


def reconcile_portfolio_state(exchange, symbols: list, portfolio_state: dict, telegram_cfg: dict,
                               am_state_path: str, base_pct: float, growth_factor: float,
                               streak_target: int) -> dict:
    """Vergleicht den lokalen Portfolio-Zustand gegen die tatsaechlichen Boersen-Positionen (nie
    blind dem lokalen Zustand vertrauen -- z.B. falls der Sicherheits-Stop ausgeloest hat, oder
    dieser Prozess nach einem Entry vor dem Speichern des Zustands abgestuerzt ist)."""
    exchange_open = {}
    for symbol in symbols:
        positions = exchange.fetch_open_positions(symbol)
        if positions:
            exchange_open[symbol] = positions[0]

    if len(exchange_open) > 1:
        logger.critical(f"Renko: MEHR ALS EINE offene Position gleichzeitig gefunden: "
                         f"{list(exchange_open.keys())}. Sollte durch die Portfolio-Arbitrierung "
                         f"nie passieren -- manuell pruefen!")
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                     f"ACHTUNG oraclebot Renko: mehrere offene Positionen gleichzeitig "
                     f"({list(exchange_open.keys())}). Bitte manuell pruefen.")

    active_symbol = portfolio_state.get('active_symbol')
    if active_symbol and active_symbol not in exchange_open:
        logger.info(f"Renko: {active_symbol} war laut lokalem Zustand offen, ist es auf der "
                    f"Boerse aber nicht mehr (regulaerer Exit oder Sicherheits-Stop) -- reconciliere.")
        resolve_am_outcome(exchange, am_state_path, base_pct, growth_factor, streak_target)
        portfolio_state = {'active_symbol': None, 'active_direction': None}
    elif not active_symbol and exchange_open:
        found_symbol = next(iter(exchange_open))
        pos = exchange_open[found_symbol]
        direction = 'long' if pos.get('side') == 'long' else 'short'
        logger.warning(f"Renko: Boerse zeigt offene Position fuer {found_symbol} ({direction}), "
                       f"lokaler Zustand kannte sie nicht (vermutlich Absturz nach Entry vor dem "
                       f"Speichern) -- uebernehme.")
        portfolio_state = {'active_symbol': found_symbol, 'active_direction': direction}

    return portfolio_state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                         help="Baut Brick-Ketten weiter und loggt Signale, platziert aber keine "
                              "echten Orders (auch wenn enabled=true in settings.json).")
    args = parser.parse_args()

    settings = load_settings()
    cfg = settings.get('renko_breakout_settings', {})
    if not cfg.get('enabled', False):
        logger.info("renko_breakout_settings.enabled=false -- Renko-Breakout-Strategie inaktiv, nichts zu tun.")
        sys.exit(0)

    symbols = cfg.get('symbols', [])
    brick_tf = cfg.get('brick_timeframe', '5m')
    base_pct_brick = cfg.get('base_pct_brick', 0.002)
    k_entropy = cfg.get('k_entropy', 0.7)
    h_window = cfg.get('h_window', 15)
    horizontal_lookback = cfg.get('horizontal_lookback', 6)
    breakout_run = cfg.get('breakout_run', 2)
    am_base_pct = cfg.get('anti_martingale_base_pct', 1.0)
    am_growth = cfg.get('anti_martingale_growth_factor', 1.5)
    am_streak = cfg.get('anti_martingale_streak_target', 3)
    history_days_buffer = cfg.get('history_days_for_brick_buffer', 3)
    min_candles_warm = history_days_buffer * 24 * 60 // 5 + 50

    secrets = load_secrets(os.path.join(PROJECT_ROOT, 'secret.json'))
    telegram_cfg = secrets.get('telegram', {})
    oraclebot_accounts = secrets.get('oraclebot', [])
    if not oraclebot_accounts or not oraclebot_accounts[0].get('apiKey'):
        logger.error("Keine 'oraclebot'-API-Keys in secret.json gefunden. Breche ab.")
        sys.exit(1)

    from oraclebot.utils.exchange import Exchange
    exchange = Exchange(oraclebot_accounts[0])

    now_utc = pd.Timestamp.now(tz='UTC')
    brick_state = load_state(BRICK_STATE_PATH)
    portfolio_state = load_portfolio_state(PORTFOLIO_STATE_PATH)

    # Taeglicher Live-vs-Backtest-Konsistenzcheck: reuse denselben Cron statt eines eigenen
    # Cronjobs (exakt dasselbe Muster wie predict_next_barrier.py fuer die Barriere-Strategie --
    # eigenes 24h-Gate, eigener Marker, eigenes try/except, damit ein Fehler hier niemals das
    # eigentliche Live-Trading dieses Laufs blockiert).
    daily_check_marker = os.path.join(DATASETS_DIR, 'last_renko_signal_check_run.txt')
    should_run_daily_check, _ = check_barrier_gate(now_utc, daily_check_marker, period_hours=24)
    if should_run_daily_check:
        try:
            from oraclebot.analysis.renko_live_signal_check import (check_brick_chain_consistency,
                                                                      check_recent_trades_against_backtest)
            consistency_lines = []
            for symbol in symbols:
                sym_state = brick_state.get(symbol, {})
                res = check_brick_chain_consistency(symbol, sym_state, base_pct_brick, k_entropy,
                                                     h_window, brick_tf)
                if res.get('match') is False:
                    consistency_lines.append(f"ABWEICHUNG {symbol}: {len(res.get('mismatches', []))} "
                                              f"Bricks weichen ab (max {res.get('max_close_diff_pct', 0):.4f}%)")
                    logger.error(f"Renko-Konsistenzcheck: {symbol} weicht ab: {res.get('mismatches')}")
                elif res.get('match') is None:
                    logger.info(f"Renko-Konsistenzcheck: {symbol} nicht vergleichbar ({res.get('reason')})")

            since_ts = now_utc - pd.Timedelta(hours=24)
            trade_lines = []
            for symbol in symbols:
                tres = check_recent_trades_against_backtest(symbol, exchange, cfg, since_ts)
                if tres['n_live_trades'] > 0:
                    trade_lines.append(f"{symbol}: {tres['n_match']}/{tres['n_comparable']} Live-Trades "
                                        f"matchen den Backtest")

            if consistency_lines or trade_lines:
                report = "oraclebot Renko taeglicher Konsistenzcheck:\n" + "\n".join(consistency_lines + trade_lines)
                logger.info(report)
                if consistency_lines:  # nur bei echten Abweichungen aktiv benachrichtigen
                    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), report)
            else:
                logger.info("Renko-Konsistenzcheck: alle Symbole konsistent, keine Live-Trades im Fenster.")
        except Exception as e:
            logger.error(f"Renko-Konsistenzcheck fehlgeschlagen (Live-Trading laeuft trotzdem weiter): {e}",
                         exc_info=True)
        mark_barrier_run_complete(now_utc, daily_check_marker, period_hours=24)

    portfolio_state = reconcile_portfolio_state(exchange, symbols, portfolio_state, telegram_cfg,
                                                 AM_STATE_PATH, am_base_pct, am_growth, am_streak)

    entry_candidates = []
    for symbol in symbols:
        sym_state = brick_state.get(symbol, {})
        last_ts = sym_state.get('last_candle_ts')

        if last_ts is None:
            logger.info(f"Renko: {symbol} -- kein Zustand vorhanden, hole {history_days_buffer} "
                        f"Tage {brick_tf}-Historie zum Aufwaermen der Brick-Kette...")
            ohlcv = fetch_all_timeframes(symbol, [brick_tf], history_days_buffer,
                                          cache_dir=DATASETS_DIR, use_cache=True)
            timeframe_minutes = {'1m': 1, '5m': 5, '15m': 15, '1h': 60}[brick_tf]
            closed_cutoff = now_utc - pd.Timedelta(minutes=timeframe_minutes)
            new_candles = ohlcv[brick_tf][ohlcv[brick_tf].index <= closed_cutoff]
        else:
            new_candles = fetch_new_closed_candles(symbol, brick_tf, min_candles_warm,
                                                    DATASETS_DIR, last_ts, now_utc)

        if new_candles.empty:
            continue

        sym_state, fresh_bricks = update_symbol_bricks(sym_state, new_candles, base_pct_brick,
                                                         k_entropy, h_window)
        brick_state[symbol] = sym_state
        n_fresh = len(fresh_bricks)
        if n_fresh == 0:
            continue

        if portfolio_state.get('active_symbol') == symbol:
            exit_sig = detect_exit(sym_state['recent_bricks'], n_fresh, portfolio_state['active_direction'])
            if exit_sig is not None:
                logger.info(f"Renko: Exit-Signal fuer {symbol} @ {exit_sig['exit_price']:.6f} "
                            f"({exit_sig['exit_ts']}).")
                if not args.dry_run:
                    close_renko_position(exchange, symbol, 'Gegen-Brick', telegram_cfg, AM_STATE_PATH,
                                          am_base_pct, am_growth, am_streak)
                portfolio_state = {'active_symbol': None, 'active_direction': None}
        elif not portfolio_state.get('active_symbol'):
            entry_sig = detect_fresh_entry(sym_state['recent_bricks'], n_fresh, horizontal_lookback,
                                            breakout_run)
            if entry_sig is not None:
                entry_candidates.append((symbol, entry_sig))

    save_state(BRICK_STATE_PATH, brick_state)

    if not portfolio_state.get('active_symbol') and entry_candidates:
        entry_candidates.sort(key=lambda c: c[1]['entry_ts'])
        symbol, entry_sig = entry_candidates[0]
        if len(entry_candidates) > 1:
            logger.info(f"Renko: {len(entry_candidates)} gleichzeitige Signale in dieser Runde "
                        f"({[c[0] for c in entry_candidates]}) -- nehme das fruheste: {symbol}.")
        logger.info(f"Renko: Entry-Signal {symbol} {entry_sig['direction'].upper()} "
                    f"@ {entry_sig['entry_price']:.6f} ({entry_sig['entry_ts']}).")
        if not args.dry_run:
            result = open_renko_position(exchange, symbol, entry_sig['direction'],
                                          entry_sig['entry_price'], cfg, telegram_cfg, AM_STATE_PATH)
            if result['action'] == 'entered':
                portfolio_state = {'active_symbol': symbol, 'active_direction': entry_sig['direction']}
        else:
            logger.info("Renko: --dry-run aktiv, kein echter Entry.")

    save_portfolio_state(PORTFOLIO_STATE_PATH, portfolio_state)
    logger.info(f"Renko: Lauf fertig. Aktiv: {portfolio_state.get('active_symbol') or '(keine Position)'}")
