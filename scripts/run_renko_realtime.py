# scripts/run_renko_realtime.py
# Echtzeit-Ausfuehrung der Renko-Breakout-Strategie (2026-09-23, Granularitaets-Fix 2026-09-24,
# Mehrfach-Positionen-Umbau 2026-09-25) -- ersetzt den 5-Minuten-Cron (run_renko_breakout.py)
# durch einen dauerhaft laufenden Prozess, der per WebSocket auf echte Trade-Ticks reagiert statt
# auf einen Cron-Tick zu warten, dabei aber weiterhin Bricks aus 5-Minuten-Kerzenschluessen baut
# -- exakt wie der Backtest.
#
# Hintergrund (Fund 2026-09-23): ein Backtest-Sweep zeigte, dass die Strategie-Edge bei JEDER
# Verzoegerung zwischen Ausbruchs-Signal und Order-Ausfuehrung schlagartig verschwindet -- die
# 5-Minuten-Cron-Architektur verursacht real ~6.5 Minuten Verzoegerung und macht die Strategie
# strukturell unrentabel, unabhaengig vom Cron-Takt.
#
# Granularitaets-Fix (2026-09-24): die erste Echtzeit-Version aggregierte den Tick-Strom zu
# 5-SEKUNDEN-Bars statt zu 5-Minuten-Bars -- das war keine schnellere Version derselben Strategie,
# sondern eine strukturell ANDERE, nie backgetestete (build_ear_bricks() schaut nur auf den
# Schlusskurs, 60x mehr Schlusskurse/Zeiteinheit erzeugen 60x mehr, ueberwiegend Rausch-
# getriebene Bricks). Fix: Brick-Engine bekommt seither ausschliesslich echte 5-Minuten-Bars, der
# WebSocket-Strom dient nur noch dazu, den Moment des Fensterabschlusses in Sekunden statt
# Minuten zu erkennen.
#
# MEHRFACH-POSITIONEN-UMBAU (Fund + Fix 2026-09-25, siehe project_oraclebot-Memory "Live-vs-
# Backtest-Divergenz Teil 2"): auch nach dem Granularitaets-Fix blieb eine Live-vs-Backtest-Luecke
# bestehen (50% Live-Winrate vs. ~72-73% in einer grossen, sauberen Backtest-Referenz). Root
# Cause: die vorherige "nur EIN Slot fuer alle 7 Symbole"-Arbitrierung erzeugt bei einem echten
# Gleichstand (mehrere Symbole signalisieren in DERSELBEN 5-Minuten-Kerze, empirisch bestaetigt
# am 2026-09-24 08:20:00 UTC fuer SOL+ADA) einen Wettlauf, den live die zufaellige WebSocket-
# Tick-Ankunftsreihenfolge entscheidet -- nicht reproduzierbar von keinem Backtest. Drei Versuche,
# den Gleichstand ueber eine "bessere" Regel aufzuloesen (Trailing-Exit, Teilmitnahme, Momentum-
# Arbitrierung) scheiterten alle an der Out-of-Sample-Pruefung (siehe research_oraclebot_
# trailing_exit_rejected.md, research_oraclebot_tiebreak_rejected.md). Strukturelle Loesung statt
# Regel-Bastelei: JEDES Symbol handelt ab jetzt UNABHAENGIG -- es gibt keinen Slot mehr, um den
# konkurriert werden koennte, also auch keinen Live-Zufall mehr, der vom Backtest abweichen kann.
# Sauber mit echter Kapital-/Gebuehren-Simulation ueber 28 UND 90 Tage (IS+OOS) validiert: deutlich
# hoeherer, konsistenter Ertrag bei NIEDRIGEREM Max-Drawdown als die alte Ein-Slot-Logik (Diversi-
# fikationseffekt ueber mehrere, nicht perfekt korrelierte gleichzeitige Positionen). Macht
# oraclebot damit auch strukturell aehnlicher zu zerobot (jedes Symbol/Timeframe laeuft dort
# ebenfalls unabhaengig, kein gemeinsamer Slot).
#
# Wiederverwendet die EXISTIERENDE, bereits getestete Brick-/Signal-Logik unveraendert
# (ear_bricks.py, renko_portfolio_state.py, horizontal_breakout_signal.py) -- nur die Positions-
# VERWALTUNG (ein Zustand pro Symbol statt ein gemeinsamer Slot) und die Anti-Martingale-
# Gewinn/Verlust-Erkennung (direkt am Fuellpreis/an der Positions-Historie statt am Kontostand-
# Delta, siehe strategy/anti_martingale.py Moduldoc) haben sich geaendert.
#
# EIGENER Zustand (renko_realtime_*.json), getrennt vom Cron-basierten System
# (renko_breakout_*.json).
import argparse
import asyncio
import json
import logging
import os
import sys
import time

import pandas as pd

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
POSITIONS_STATE_PATH = os.path.join(STATE_DIR, 'renko_realtime_positions.json')
AM_STATE_PATH = os.path.join(STATE_DIR, 'renko_realtime_anti_martingale.json')
DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')

# MUSS mit settings.json::renko_breakout_settings.brick_timeframe uebereinstimmen -- die
# Brick-Engine baut Bricks aus Bars GENAU dieser Groesse, exakt wie der Backtest (siehe
# Moduldoc: der 2026-09-24-Fix). Kein Zufall, dass hier "5m" -> 300s steht, nicht 5s.
_TIMEFRAME_SECONDS = {'1m': 60, '5m': 300, '15m': 900, '1h': 3600}
STATE_SAVE_INTERVAL_SECONDS = 30
ENABLED_RECHECK_INTERVAL_SECONDS = 60


def load_secrets(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_positions_state(path: str, symbols: list) -> dict:
    """Zustand PRO SYMBOL (Mehrfach-Positionen, Fix 2026-09-25) -- {symbol: None} wenn keine
    Position offen, sonst {symbol: {'direction', 'entry_price', 'contracts'}}. Ersetzt den
    frueheren gemeinsamen 'active_symbol'-Einzelzustand."""
    if not os.path.exists(path):
        return {s: None for s in symbols}
    with open(path, 'r', encoding='utf-8') as f:
        state = json.load(f)
    for s in symbols:
        state.setdefault(s, None)
    return state


def save_positions_state(path: str, state: dict) -> None:
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
                       history_days: int, brick_tf: str, now_utc: pd.Timestamp) -> dict:
    """Baut den initialen Brick-Zustand IDENTISCH zum Cron-Cold-Start (scripts/run_renko_breakout.py)
    auf: laedt `history_days` Tage `brick_tf`-REST-Historie, verwirft die noch nicht abgeschlossene
    letzte Kerze, und verarbeitet den Rest durch dieselbe update_symbol_bricks()-Funktion, die
    danach auch die live per WebSocket aggregierten Bars derselben Groesse verarbeitet."""
    ohlcv = fetch_all_timeframes(symbol, [brick_tf], history_days, cache_dir=DATASETS_DIR, use_cache=True)
    df = ohlcv[brick_tf]
    timeframe_minutes = _TIMEFRAME_SECONDS[brick_tf] // 60
    closed_cutoff = now_utc - pd.Timedelta(minutes=timeframe_minutes)
    closed_candles = df[df.index <= closed_cutoff]
    sym_state, _ = update_symbol_bricks({}, closed_candles, base_pct, k_entropy, h_window)
    return sym_state


def reconcile_positions_state(exchange, symbols: list, positions_state: dict, telegram_cfg: dict,
                               am_state_path: str, base_pct: float, growth_factor: float,
                               streak_target: int) -> dict:
    """Nie blind dem lokalen Zustand vertrauen, immer gegen die echten Boersen-Positionen je
    Symbol abgleichen -- mehrere gleichzeitig offene Positionen sind seit dem Mehrfach-
    Positionen-Umbau der NORMALFALL, keine Fehlermeldung mehr wert."""
    new_state = dict(positions_state)
    for symbol in symbols:
        exchange_pos = exchange.fetch_open_positions(symbol)
        local = new_state.get(symbol)

        if local is not None and not exchange_pos:
            logger.info(f"Renko-Realtime: {symbol} war lokal offen, an der Boerse nicht mehr -- reconciliere.")
            resolve_am_outcome(exchange, symbol, am_state_path, base_pct, growth_factor, streak_target)
            new_state[symbol] = None
        elif local is None and exchange_pos:
            pos = exchange_pos[0]
            direction = 'long' if pos.get('side') == 'long' else 'short'
            entry_price = float(pos.get('entryPrice') or 0)
            logger.warning(f"Renko-Realtime: Boerse zeigt offene Position fuer {symbol} ({direction}), "
                           f"lokal unbekannt -- uebernehme.")
            new_state[symbol] = {'direction': direction, 'entry_price': entry_price,
                                  'contracts': float(pos.get('contracts') or 0)}

    return new_state


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
    brick_tf = cfg.get('brick_timeframe', '5m')
    bar_seconds = _TIMEFRAME_SECONDS[brick_tf]

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
    positions_state = load_positions_state(POSITIONS_STATE_PATH, symbols)
    positions_state = await loop.run_in_executor(
        None, reconcile_positions_state, exchange, symbols, positions_state, telegram_cfg,
        AM_STATE_PATH, am_base_pct, am_growth, am_streak)

    now_utc = pd.Timestamp.now(tz='UTC')
    aggregators = {}
    for symbol in symbols:
        if symbol not in brick_state:
            logger.info(f"Renko-Realtime: {symbol} -- Cold-Start, baue Anfangszustand aus {history_days_buffer} Tagen {brick_tf}-Historie...")
            brick_state[symbol] = await loop.run_in_executor(
                None, seed_symbol_state, symbol, base_pct_by_symbol[symbol], k_entropy, h_window,
                history_days_buffer, brick_tf, now_utc)
        aggregators[symbol] = BarAggregator(bar_seconds=bar_seconds)
    save_state_atomic(BRICK_STATE_PATH, brick_state)

    n_open = sum(1 for v in positions_state.values() if v is not None)
    mode_label = " [DRY-RUN, keine echten Orders]" if dry_run else ""
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                 f"oraclebot Renko-Realtime gestartet{mode_label} ({len(symbols)} Symbole, "
                 f"{brick_tf}-Bricks, Mehrfach-Positionen, Echtzeit-Reaktion). "
                 f"Aktuell offen: {n_open}/{len(symbols)}")

    last_state_save = time.monotonic()
    last_enabled_check = time.monotonic()
    soft_paused = False  # enabled=false erkannt, aber noch offene Position(en) -- keine neuen
                          # Entries mehr, aber die aktiven Positionen weiter bis zum Exit ueberwachen

    async def handle_new_bars(symbol: str, new_candles) -> None:
        """Verarbeitet neue, abgeschlossene Bars fuer EIN Symbol: Brick-Kette fortsetzen, Exit-
        oder Entry-Signal pruefen, ggf. ausfuehren -- komplett unabhaengig von allen anderen
        Symbolen (kein gemeinsamer Slot mehr, siehe Moduldoc: Mehrfach-Positionen-Umbau).
        Gemeinsame Logik fuer den normalen Tick-getriebenen Pfad UND den periodischen Stale-Bar-
        Flush weiter unten -- beide muessen exakt gleich behandelt werden."""
        sym_state, fresh_bricks = update_symbol_bricks(brick_state[symbol], new_candles,
                                                         base_pct_by_symbol[symbol], k_entropy, h_window)
        brick_state[symbol] = sym_state
        n_fresh = len(fresh_bricks)
        if n_fresh == 0:
            return

        local_pos = positions_state.get(symbol)
        if local_pos is not None:
            exit_sig = detect_exit(sym_state['recent_bricks'], n_fresh, local_pos['direction'])
            if exit_sig is not None:
                logger.info(f"Renko-Realtime: Exit-Signal {symbol} @ {exit_sig['exit_price']:.6f}"
                            + (" [DRY-RUN]" if dry_run else ""))
                if not dry_run:
                    await loop.run_in_executor(None, close_renko_position, exchange, symbol,
                                                local_pos['direction'], local_pos['entry_price'],
                                                'Gegen-Brick (Echtzeit)', telegram_cfg, AM_STATE_PATH,
                                                am_base_pct, am_growth, am_streak)
                positions_state[symbol] = None
                save_positions_state(POSITIONS_STATE_PATH, positions_state)
        elif not soft_paused:
            entry_sig = detect_fresh_entry(sym_state['recent_bricks'], n_fresh, horizontal_lookback, breakout_run)
            if entry_sig is not None:
                logger.info(f"Renko-Realtime: Entry-Signal {symbol} {entry_sig['direction'].upper()} "
                            f"@ {entry_sig['entry_price']:.6f}" + (" [DRY-RUN]" if dry_run else ""))
                if dry_run:
                    positions_state[symbol] = {'direction': entry_sig['direction'],
                                                'entry_price': entry_sig['entry_price'], 'contracts': 0}
                    save_positions_state(POSITIONS_STATE_PATH, positions_state)
                else:
                    result = await loop.run_in_executor(
                        None, open_renko_position, exchange, symbol, entry_sig['direction'],
                        entry_sig['entry_price'], cfg, telegram_cfg, AM_STATE_PATH)
                    if result['action'] == 'entered':
                        positions_state[symbol] = {'direction': entry_sig['direction'],
                                                    'entry_price': result['entry_price'],
                                                    'contracts': result['contracts']}
                        save_positions_state(POSITIONS_STATE_PATH, positions_state)

    stream = BitgetTradeStream(symbols)
    logger.info(f"Renko-Realtime: verbinde mit Bitget-WebSocket fuer {symbols}...")

    async for symbol, ts_ms, price in stream.stream():
        agg = aggregators.get(symbol)
        if agg is None:
            continue
        finished_bar = agg.add_tick(ts_ms, price)
        if finished_bar is not None:
            await handle_new_bars(symbol, bars_to_df([finished_bar]))

        now = time.monotonic()
        if now - last_state_save > STATE_SAVE_INTERVAL_SECONDS:
            save_state_atomic(BRICK_STATE_PATH, brick_state)
            last_state_save = now

        if now - last_enabled_check > ENABLED_RECHECK_INTERVAL_SECONDS:
            last_enabled_check = now

            # Bei brick_tf=5m (statt der frueheren 5s-Bars) ist ein Symbol OHNE einen einzigen
            # Trade-Tick ueber ein volles 5-Minuten-Fenster deutlich plausibler (ruhige Coins in
            # ruhigen Phasen) -- ohne diesen Flush wuerde ein solches Symbol seinen Bar einfach
            # nie abschliessen und seine Brick-Kette stillschweigend einfrieren, bis der naechste
            # echte Trade eintrifft (siehe utils/realtime_bars.py:flush_stale_bars).
            now_ms = int(time.time() * 1000)
            for sym, agg in aggregators.items():
                flushed = agg.flush_stale_bars(now_ms)
                if flushed:
                    await handle_new_bars(sym, bars_to_df(flushed))

            # Periodischer Abgleich gegen die echten Boersen-Positionen -- NICHT nur einmal beim
            # Start (Fund 2026-09-23: eine manuelle Positions-Schliessung durch den User waehrend
            # eines laufenden Prozesses blieb sonst bis zum naechsten Exit-Signal unbemerkt).
            if not dry_run:
                try:
                    positions_state = await loop.run_in_executor(
                        None, reconcile_positions_state, exchange, symbols, positions_state,
                        telegram_cfg, AM_STATE_PATH, am_base_pct, am_growth, am_streak)
                    save_positions_state(POSITIONS_STATE_PATH, positions_state)
                except Exception as e:
                    logger.error(f"Renko-Realtime: periodischer Reconcile fehlgeschlagen: {e}")

            try:
                fresh_settings = load_settings()
                still_enabled = fresh_settings.get('renko_breakout_settings', {}).get('enabled', False)
                any_open = any(v is not None for v in positions_state.values())
                if not still_enabled:
                    if any_open:
                        if not soft_paused:
                            logger.info("Renko-Realtime: enabled=false erkannt, aber Position(en) offen -- "
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
        # basierten run_renko_breakout.py). Seit dem Granularitaets-Fix (2026-09-24, siehe
        # Moduldoc) baut dieser Prozess Bricks aus derselben brick_tf-Groesse wie der 5m-REST-
        # Neuaufbau -- check_brick_chain_consistency() waere technisch jetzt sinnvoll anwendbar,
        # ist aber bewusst NICHT verdrahtet: das war nicht Teil dieses Fixes.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                         help="Verbindet sich echt und verarbeitet echte Ticks/Bricks/Signale, "
                              "platziert aber keine echten Orders (auch wenn enabled=true).")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))
