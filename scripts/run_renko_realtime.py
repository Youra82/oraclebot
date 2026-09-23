# scripts/run_renko_realtime.py
# Echtzeit-Ausfuehrung der Renko-Breakout-Strategie (2026-09-23) -- ersetzt den 5-Minuten-Cron
# (run_renko_breakout.py) durch einen dauerhaft laufenden Prozess, der per WebSocket auf echte
# Trade-Ticks reagiert statt auf abgeschlossene 5m-Kerzen zu warten.
#
# Hintergrund (Fund 2026-09-23): ein Backtest-Sweep zeigte, dass die Strategie-Edge bei JEDER
# Verzoegerung zwischen Ausbruchs-Signal und Order-Ausfuehrung schlagartig verschwindet (0 Min
# Lag: +420% OOS: 1+ Min Lag: -142% OOS) -- die 5-Minuten-Cron-Architektur (die erst auf eine
# ABGESCHLOSSENE Kerze wartet) verursacht real ~6.5 Minuten Verzoegerung und macht die Strategie
# strukturell unrentabel, unabhaengig vom Cron-Takt. Dieses Skript reagiert stattdessen auf
# Preisbewegungen INNERHALB von Sekunden nach dem tatsaechlichen Ausbruch.
#
# Wiederverwendet die EXISTIERENDE, bereits getestete Brick-/Signal-/Order-Logik unveraendert
# (ear_bricks.py, renko_portfolio_state.py, horizontal_breakout_signal.py, renko_live_trade.py)
# -- nur die Datenquelle (WebSocket-Ticks statt REST-Poll auf 5m-Kerzen, aggregiert zu feinen
# 5-Sekunden-Bars ueber utils/realtime_bars.py) und der Ausloese-Takt (kontinuierlich statt
# alle 5 Minuten) aendern sich.
#
# EIGENER Zustand (renko_realtime_*.json), getrennt vom Cron-basierten System
# (renko_breakout_*.json) -- vermeidet jede Verwechslung/Vermischung zwischen beiden Ansaetzen
# waehrend der Validierungsphase.
import argparse
import asyncio
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger(__name__)

from oraclebot.strategy.renko_portfolio_state import (detect_exit, detect_fresh_entry, load_state,
                                                        save_state, update_symbol_bricks)
from oraclebot.strategy.renko_live_trade import close_renko_position, open_renko_position, resolve_am_outcome
from oraclebot.utils.bitget_ws import BitgetTradeStream
from oraclebot.utils.config import load_settings
from oraclebot.utils.data_fetch import fetch_all_timeframes
from oraclebot.utils.realtime_bars import BarAggregator, bars_to_df
from oraclebot.utils.telegram import send_message

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
STATE_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'state')
BRICK_STATE_PATH = os.path.join(STATE_DIR, 'renko_realtime_bricks.json')
PORTFOLIO_STATE_PATH = os.path.join(STATE_DIR, 'renko_realtime_portfolio.json')
AM_STATE_PATH = os.path.join(STATE_DIR, 'renko_realtime_anti_martingale.json')
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')

BAR_SECONDS = 5
STATE_SAVE_INTERVAL_SECONDS = 30
ENABLED_RECHECK_INTERVAL_SECONDS = 60


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
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, path)  # atomar -- kein halb geschriebener Zustand bei Absturz


def save_state_atomic(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(tmp_path, path)


def seed_symbol_state(symbol: str, base_pct: float, k_entropy: float, h_window: int,
                       history_days: int) -> dict:
    """Baut den initialen Brick-Zustand aus regulaeren 5m-REST-Daten auf (wie der Cron-
    Cold-Start) -- schnell startklar statt stundenlang auf organisch entstehende Echtzeit-Bricks
    zu warten. `buffer_candles` bleibt bewusst LEER: die 5m-Kerzen-Entropie-Historie passt nicht
    zur Granularitaet der kommenden 5s-Echtzeit-Bars, die Entropie-Glaettung baut sich stattdessen
    organisch aus den ersten echten 5s-Bars neu auf (min_periods=1, daher kein Fehler, nur ein
    kurzes, selbstkorrigierendes Einschwingen von wenigen Bars)."""
    from oraclebot.data.ear_bricks import build_ear_bricks

    ohlcv = fetch_all_timeframes(symbol, ['5m'], history_days, cache_dir=DATASETS_DIR, use_cache=True)
    df = ohlcv['5m']
    bricks = build_ear_bricks(df, base_pct=base_pct, k_entropy=k_entropy, h_window=h_window)
    recent_bricks = [{'ts': b['ts'].isoformat(), 'direction': b['direction'], 'close': b['close']}
                      for b in bricks[-60:]]
    last_close = bricks[-1]['close'] if bricks else float(df['close'].iloc[-1])
    last_direction = bricks[-1]['direction'] if bricks else None
    return {'last_candle_ts': None, 'buffer_candles': [], 'last_brick_close': last_close,
            'last_brick_direction': last_direction, 'recent_bricks': recent_bricks}


def reconcile_portfolio_state(exchange, symbols: list, portfolio_state: dict, telegram_cfg: dict,
                               am_state_path: str, base_pct: float, growth_factor: float,
                               streak_target: int) -> dict:
    """Identisch zur Cron-Variante (scripts/run_renko_breakout.py) -- nie blind dem lokalen
    Zustand vertrauen, immer gegen die echten Boersen-Positionen abgleichen."""
    exchange_open = {}
    for symbol in symbols:
        positions = exchange.fetch_open_positions(symbol)
        if positions:
            exchange_open[symbol] = positions[0]

    if len(exchange_open) > 1:
        logger.critical(f"Renko-Realtime: MEHR ALS EINE offene Position: {list(exchange_open.keys())}!")
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                     f"ACHTUNG oraclebot Renko-Realtime: mehrere offene Positionen gleichzeitig "
                     f"({list(exchange_open.keys())}). Bitte manuell pruefen.")

    active_symbol = portfolio_state.get('active_symbol')
    if active_symbol and active_symbol not in exchange_open:
        logger.info(f"Renko-Realtime: {active_symbol} war lokal offen, an der Boerse nicht mehr -- reconciliere.")
        resolve_am_outcome(exchange, am_state_path, base_pct, growth_factor, streak_target)
        portfolio_state = {'active_symbol': None, 'active_direction': None}
    elif not active_symbol and exchange_open:
        found_symbol = next(iter(exchange_open))
        pos = exchange_open[found_symbol]
        direction = 'long' if pos.get('side') == 'long' else 'short'
        logger.warning(f"Renko-Realtime: Boerse zeigt offene Position fuer {found_symbol} ({direction}), "
                       f"lokal unbekannt -- uebernehme.")
        portfolio_state = {'active_symbol': found_symbol, 'active_direction': direction}

    return portfolio_state


async def run(dry_run: bool = False):
    settings = load_settings()
    cfg = settings.get('renko_breakout_settings', {})
    if not cfg.get('enabled', False):
        logger.info("renko_breakout_settings.enabled=false -- nichts zu tun, beende.")
        return

    symbols = cfg.get('symbols', [])
    base_pct_by_symbol = {k: v for k, v in cfg.get('base_pct_brick_by_symbol', {}).items() if k != '_note'}
    k_entropy = cfg.get('k_entropy', 0.7)
    h_window = cfg.get('h_window', 15)
    horizontal_lookback = cfg.get('horizontal_lookback', 6)
    breakout_run = cfg.get('breakout_run', 2)
    am_base_pct = cfg.get('anti_martingale_base_pct', 1.0)
    am_growth = cfg.get('anti_martingale_growth_factor', 1.5)
    am_streak = cfg.get('anti_martingale_streak_target', 3)
    history_days_buffer = cfg.get('history_days_for_brick_buffer', 3)

    secrets = load_secrets(os.path.join(PROJECT_ROOT, 'secret.json'))
    telegram_cfg = secrets.get('telegram', {})
    oraclebot_accounts = secrets.get('oraclebot', [])
    if not oraclebot_accounts or not oraclebot_accounts[0].get('apiKey'):
        logger.error("Keine 'oraclebot'-API-Keys in secret.json gefunden. Breche ab.")
        return

    from oraclebot.utils.exchange import Exchange
    exchange = Exchange(oraclebot_accounts[0])
    loop = asyncio.get_event_loop()

    brick_state = load_state(BRICK_STATE_PATH)
    portfolio_state = load_portfolio_state(PORTFOLIO_STATE_PATH)
    portfolio_state = await loop.run_in_executor(
        None, reconcile_portfolio_state, exchange, symbols, portfolio_state, telegram_cfg,
        AM_STATE_PATH, am_base_pct, am_growth, am_streak)

    aggregators = {}
    for symbol in symbols:
        if symbol not in brick_state:
            logger.info(f"Renko-Realtime: {symbol} -- Cold-Start, baue Anfangszustand aus {history_days_buffer} Tagen 5m-Historie...")
            brick_state[symbol] = await loop.run_in_executor(
                None, seed_symbol_state, symbol, base_pct_by_symbol[symbol], k_entropy, h_window,
                history_days_buffer)
        aggregators[symbol] = BarAggregator(bar_seconds=BAR_SECONDS)
    save_state_atomic(BRICK_STATE_PATH, brick_state)

    mode_label = " [DRY-RUN, keine echten Orders]" if dry_run else ""
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                 f"oraclebot Renko-Realtime gestartet{mode_label} ({len(symbols)} Symbole, {BAR_SECONDS}s-Bars). "
                 f"Aktiv: {portfolio_state.get('active_symbol') or '(keine Position)'}")

    last_state_save = time.monotonic()
    last_enabled_check = time.monotonic()
    soft_paused = False  # enabled=false erkannt, aber noch offene Position -- keine neuen
                          # Entries mehr, aber die aktive Position weiter bis zum Exit ueberwachen

    stream = BitgetTradeStream(symbols)
    logger.info(f"Renko-Realtime: verbinde mit Bitget-WebSocket fuer {symbols}...")

    async for symbol, ts_ms, price in stream.stream():
        agg = aggregators.get(symbol)
        if agg is None:
            continue
        finished_bar = agg.add_tick(ts_ms, price)
        if finished_bar is None:
            continue

        new_candles = bars_to_df([finished_bar])
        sym_state, fresh_bricks = update_symbol_bricks(brick_state[symbol], new_candles,
                                                         base_pct_by_symbol[symbol], k_entropy, h_window)
        brick_state[symbol] = sym_state
        n_fresh = len(fresh_bricks)

        if n_fresh > 0:
            if portfolio_state.get('active_symbol') == symbol:
                exit_sig = detect_exit(sym_state['recent_bricks'], n_fresh, portfolio_state['active_direction'])
                if exit_sig is not None:
                    logger.info(f"Renko-Realtime: Exit-Signal {symbol} @ {exit_sig['exit_price']:.6f}"
                                + (" [DRY-RUN]" if dry_run else ""))
                    if not dry_run:
                        await loop.run_in_executor(None, close_renko_position, exchange, symbol,
                                                    'Gegen-Brick (Echtzeit)', telegram_cfg, AM_STATE_PATH,
                                                    am_base_pct, am_growth, am_streak)
                    portfolio_state = {'active_symbol': None, 'active_direction': None}
                    save_portfolio_state(PORTFOLIO_STATE_PATH, portfolio_state)
            elif not portfolio_state.get('active_symbol') and not soft_paused:
                entry_sig = detect_fresh_entry(sym_state['recent_bricks'], n_fresh, horizontal_lookback, breakout_run)
                if entry_sig is not None:
                    logger.info(f"Renko-Realtime: Entry-Signal {symbol} {entry_sig['direction'].upper()} "
                                f"@ {entry_sig['entry_price']:.6f}" + (" [DRY-RUN]" if dry_run else ""))
                    if dry_run:
                        portfolio_state = {'active_symbol': symbol, 'active_direction': entry_sig['direction']}
                        save_portfolio_state(PORTFOLIO_STATE_PATH, portfolio_state)
                    else:
                        result = await loop.run_in_executor(
                            None, open_renko_position, exchange, symbol, entry_sig['direction'],
                            entry_sig['entry_price'], cfg, telegram_cfg, AM_STATE_PATH)
                        if result['action'] == 'entered':
                            portfolio_state = {'active_symbol': symbol, 'active_direction': entry_sig['direction']}
                            save_portfolio_state(PORTFOLIO_STATE_PATH, portfolio_state)

        now = time.monotonic()
        if now - last_state_save > STATE_SAVE_INTERVAL_SECONDS:
            save_state_atomic(BRICK_STATE_PATH, brick_state)
            last_state_save = now

        if now - last_enabled_check > ENABLED_RECHECK_INTERVAL_SECONDS:
            last_enabled_check = now
            try:
                fresh_settings = load_settings()
                still_enabled = fresh_settings.get('renko_breakout_settings', {}).get('enabled', False)
                if not still_enabled:
                    if portfolio_state.get('active_symbol'):
                        if not soft_paused:
                            logger.info("Renko-Realtime: enabled=false erkannt, aber Position offen -- "
                                        "ueberwache weiter bis zum naechsten Exit-Signal, keine neuen Entries mehr.")
                        soft_paused = True
                    else:
                        logger.info("Renko-Realtime: enabled=false erkannt, keine offene Position -- beende sauber.")
                        save_state_atomic(BRICK_STATE_PATH, brick_state)
                        stream.stop()
                        return
                elif soft_paused:
                    logger.info("Renko-Realtime: enabled=true erkannt -- Pause aufgehoben, Entries wieder aktiv.")
                    soft_paused = False
            except Exception as e:
                logger.error(f"Renko-Realtime: Settings-Neucheck fehlgeschlagen: {e}")

        # HINWEIS: kein automatischer taeglicher Konsistenzcheck hier (anders als beim Cron-
        # basierten run_renko_breakout.py) -- check_brick_chain_consistency() baut zum Vergleich
        # aus 5m-REST-Kerzen neu auf, waehrend dieser Prozess auf 5s-Echtzeit-Bars laeuft. Beide
        # Granularitaeten liefern strukturell unterschiedliche Brick-Ketten -- ein Vergleich
        # zwischen ihnen wuerde staendig falsche Abweichungs-Alarme erzeugen, kein echtes
        # Diagnosewerkzeug.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                         help="Verbindet sich echt und verarbeitet echte Ticks/Bricks/Signale, "
                              "platziert aber keine echten Orders (auch wenn enabled=true).")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))
