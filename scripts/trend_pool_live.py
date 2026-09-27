# scripts/trend_pool_live.py
# Stuendlicher Live-Lauf des Trend-Pools (Cron: `1 * * * *`). Ablauf:
#   1. Sperren pruefen: kein zweiter Lauf parallel, und NICHT handeln, solange die alte Renko-Echtzeit-Engine
#      noch lebt (sie uebernimmt fremde Positionen auf ihren Coins, siehe run_renko_realtime.py).
#   2. Zustand gegen die Boerse abgleichen (strikt: API-Fehler -> Abbruch, nie "Position weg" annehmen).
#   3. Fuer alle ausgewaehlten + gehaltenen Strategien die Trades mit der Backtest-Signalfunktion berechnen.
#   4. decide_actions() (identisch zur Portfolio-Simulation) -> Positionen schliessen/eroeffnen.
#   .venv/bin/python3 scripts/trend_pool_live.py            # echt
#   .venv/bin/python3 scripts/trend_pool_live.py --dry-run  # nur anzeigen, keine Orders
import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oraclebot.strategy.trend_pool import simulate_coin  # noqa: E402
from oraclebot.strategy.trend_pool_decide import decide_actions  # noqa: E402
from oraclebot.strategy.trend_pool_select import week_start  # noqa: E402
from oraclebot.utils.config import PROJECT_ROOT, load_settings  # noqa: E402
from oraclebot.utils.ohlcv_cache import update_cache  # noqa: E402
from oraclebot.utils.telegram import send_message  # noqa: E402

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')
STATE_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'state')
SELECTION_PATH = os.path.join(STATE_DIR, 'trend_pool_selection.json')
POSITIONS_PATH = os.path.join(STATE_DIR, 'trend_pool_positions.json')
LOCK_PATH = os.path.join(STATE_DIR, 'trend_pool_live.lock')
RENKO_PID_PATH = os.path.join(STATE_DIR, 'renko_realtime.pid')
MIN_ORDER_USDT = 5.0
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('trend_pool_live')


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def renko_engine_alive() -> bool:
    if not os.path.exists(RENKO_PID_PATH):
        return False
    try:
        return _pid_alive(int(open(RENKO_PID_PATH).read().strip()))
    except ValueError:
        return False


def acquire_lock() -> bool:
    os.makedirs(STATE_DIR, exist_ok=True)
    if os.path.exists(LOCK_PATH):
        try:
            if _pid_alive(int(open(LOCK_PATH).read().strip())):
                return False
        except ValueError:
            pass
    with open(LOCK_PATH, 'w') as f:
        f.write(str(os.getpid()))
    return True


def load_state() -> dict:
    if not os.path.exists(POSITIONS_PATH):
        return {'positions': {}, 'entered': []}
    with open(POSITIONS_PATH, encoding='utf-8') as f:
        s = json.load(f)
    for p in s['positions'].values():
        p['entry_ts'] = pd.Timestamp(p['entry_ts'])
    return s


def save_state(s: dict):
    out = {'positions': {c: dict(p, entry_ts=p['entry_ts'].isoformat()) for c, p in s['positions'].items()},
           'entered': s['entered'][-2000:]}
    tmp = POSITIONS_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, POSITIONS_PATH)


def run(dry_run: bool = False, now: pd.Timestamp = None):
    settings = load_settings()
    cfg = settings['trend_pool_settings']
    if not cfg.get('enabled', False):
        logger.info('trend_pool_settings.enabled=false -- nichts zu tun.')
        return
    with open(os.path.join(PROJECT_ROOT, 'secret.json'), encoding='utf-8') as f:
        secrets = json.load(f)
    tg = secrets.get('telegram', {})

    def notify(msg):
        logger.info(msg.replace('\n', ' | '))
        if not dry_run:
            send_message(tg.get('bot_token'), tg.get('chat_id'), msg)

    if renko_engine_alive():
        notify('ORACLEBOT Trend-Pool: alte Renko-Engine laeuft noch -- kein Handel, bis sie gestoppt ist '
               '(scripts/retire_renko.sh ausfuehren).')
        return

    now = now or pd.Timestamp.now(tz='UTC')
    bar_close = now.floor('h')
    ws = week_start(now)
    state = load_state()
    entered = {(i, pd.Timestamp(t)) for i, t in state['entered']}

    from oraclebot.utils.exchange import Exchange
    ex = Exchange(secrets['oraclebot'][0])

    # --- 2. Abgleich mit der Boerse (strikt) ---
    for coin in list(state['positions']):
        symbol = f'{coin}/USDT:USDT'
        live = ex.fetch_open_positions_strict(symbol)
        if not live:
            p = state['positions'].pop(coin)
            ex.cancel_all_orders_for_symbol(symbol)
            notify(f"ORACLEBOT Trend-Pool: {coin} ({p['id']}) ist an der Boerse nicht mehr offen "
                   f"(Sicherheits-Stop, Liquidation oder manuell geschlossen) -- aus dem Zustand entfernt.")

    # --- 3. Auswahl (neue UTC-Woche -> hier selbst erstellen, unabhaengig von der Zeitzone des Servers) + Signale ---
    sel = None
    if os.path.exists(SELECTION_PATH):
        with open(SELECTION_PATH, encoding='utf-8') as f:
            sel = json.load(f)
    if sel is None or pd.Timestamp(sel['week_start']) != ws:
        from trend_pool_weekly import run_selection, format_selection, save_selection
        sel = run_selection(cfg, ex.exchange, ex.fetch_balance_total_usdt(), now=now)
        if not dry_run:
            save_selection(sel)
        notify(format_selection(sel))
    selected = [r['id'] for r in sel['selected']]
    need = sorted(set(selected) | {p['id'] for p in state['positions'].values()})
    trades = {}
    for coin in sorted({i.split('|')[0] for i in need}):
        h1 = update_cache(coin, cfg['anchor'], DATASETS_DIR, now=now)
        trades.update(simulate_coin(h1, coin, cfg['base_pct_1h_by_coin'][coin], [i for i in need if i.startswith(coin + '|')],
                                    cfg['k_entropy'], cfg['h_window'], cfg['horizontal_lookback'], cfg['breakout_run'],
                                    cfg['cost_pct'], cfg['funding_pct_8h']))
    # fremde Positionen auf Pool-Coins (z.B. manuell) blockieren den Coin, werden aber nicht angefasst
    view = {c: {'id': p['id'], 'dir': p['dir'], 'entry_ts': p['entry_ts']} for c, p in state['positions'].items()}
    actions = decide_actions(bar_close, ws, trades, selected, view, entered, cfg.get('entry_grace_hours', 2))
    for a in [x for x in actions if x['action'] == 'open']:
        if ex.fetch_open_positions_strict(f"{a['coin']}/USDT:USDT"):
            logger.warning(f"{a['coin']}: offene Position ohne Trend-Pool-Zustand -- Einstieg uebersprungen.")
            actions.remove(a)
    logger.info(f"Bar {bar_close}, Woche {ws.date()}, Auswahl {len(selected)}, Positionen {list(state['positions'])}, Aktionen {actions}")

    # --- 4. Ausfuehren ---
    for a in [x for x in actions if x['action'] == 'close']:
        coin, symbol = a['coin'], f"{a['coin']}/USDT:USDT"
        p = state['positions'][coin]
        if dry_run:
            print('DRY-RUN close', a); continue
        ex.cancel_all_orders_for_symbol(symbol)
        ex.close_position(symbol)
        state['positions'].pop(coin)
        save_state(state)
        notify(f"ORACLEBOT Trend-Pool EXIT {coin} {'LONG' if p['dir'] > 0 else 'SHORT'} ({p['id']}), Grund: {a['reason']}, "
               f"Einstieg {p['entry_px']:.6g}")
    opens = [x for x in actions if x['action'] == 'open']
    if opens:
        equity = ex.fetch_balance_total_usdt()
        for a in opens:
            coin, symbol = a['coin'], f"{a['coin']}/USDT:USDT"
            entered.add((a['id'], a['entry_ts']))
            state['entered'].append([a['id'], a['entry_ts'].isoformat()])
            free = ex.fetch_balance_usdt()
            margin = min(equity / cfg['top_k'], free * 0.98)
            price = float(ex.exchange.fetch_ticker(symbol)['last'])
            notional = margin * cfg['leverage']
            amount = notional / price
            min_amt = ex.fetch_min_amount_tradable(symbol)
            if notional < MIN_ORDER_USDT or amount < min_amt:
                notify(f"ORACLEBOT Trend-Pool: {coin} ({a['id']}) ausgelassen -- Position {notional:.2f} USDT unter Bitget-Minimum.")
                save_state(state); continue
            if dry_run:
                print('DRY-RUN open', a, f'notional {notional:.2f}'); continue
            side = 'buy' if a['dir'] > 0 else 'sell'
            ex.set_margin_mode(symbol, cfg['margin_mode'])
            ex.set_leverage(symbol, int(cfg['leverage']), cfg['margin_mode'])
            try:
                order = ex.place_market_order(symbol, side, amount, margin_mode=cfg['margin_mode'])
            except Exception as e:
                notify(f"ORACLEBOT Trend-Pool: Einstieg {coin} fehlgeschlagen: {e}")
                save_state(state); continue
            live = ex.fetch_open_positions_strict(symbol)
            fill = float((live[0].get('entryPrice') if live else None) or order.get('average') or price)
            filled = float(live[0].get('contracts')) if live else float(order.get('filled') or amount)
            stop = fill * (1 - a['dir'] * cfg['safety_stop_pct'] / 100)
            try:
                ex.place_trigger_market_order(symbol, 'sell' if a['dir'] > 0 else 'buy', filled, stop, reduce=True)
            except Exception as e:
                ex.close_position(symbol)
                notify(f"ACHTUNG ORACLEBOT Trend-Pool: Sicherheits-Stop fuer {coin} fehlgeschlagen ({e}) -- Position sofort geschlossen.")
                save_state(state); continue
            state['positions'][coin] = {'id': a['id'], 'dir': a['dir'], 'entry_ts': a['entry_ts'], 'entry_px': fill,
                                        'signal_px': a['entry_px'], 'contracts': filled, 'notional': notional}
            save_state(state)
            notify(f"ORACLEBOT Trend-Pool ENTRY {coin} {'LONG' if a['dir'] > 0 else 'SHORT'} ({a['id']})\n"
                   f"Fill {fill:.6g} (Signal-Kerzenschluss {a['entry_px']:.6g}), Nominal {notional:.2f} USDT, "
                   f"Hebel {cfg['leverage']}x, Sicherheits-Stop {stop:.6g} ({cfg['safety_stop_pct']}%)")
    if not dry_run:
        save_state(state)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    if not acquire_lock():
        logger.info('Anderer Lauf aktiv -- beende.')
        return
    try:
        run(a.dry_run)
    finally:
        try:
            os.remove(LOCK_PATH)
        except OSError:
            pass


if __name__ == '__main__':
    main()
