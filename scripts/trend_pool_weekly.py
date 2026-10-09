# scripts/trend_pool_weekly.py
# Woechentliche Auswahl fuer den Trend-Pool (wird von trend_pool_live.py zu Beginn jeder UTC-Woche automatisch aufgerufen). Aktualisiert die 1h-Caches aller Pool-Coins,
# berechnet alle Pool-Strategien mit der Live-Signalfunktion, waehlt Top-K nach dem Rueckblick-Score (nur Coins,
# deren Bitget-Mindestorder in einen Slot passt) und schreibt artifacts/state/trend_pool_selection.json.
#   .venv/bin/python3 scripts/trend_pool_weekly.py            # normal
#   .venv/bin/python3 scripts/trend_pool_weekly.py --dry-run  # nur anzeigen, nichts speichern/senden
import argparse
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from oraclebot.analysis.trend_pool_portfolio import tradable_coins  # noqa: E402
from oraclebot.strategy.trend_pool import pool_ids, simulate_coin  # noqa: E402
from oraclebot.strategy.trend_pool_select import select_strategies, week_start  # noqa: E402
from oraclebot.utils.config import PROJECT_ROOT, load_settings  # noqa: E402
from oraclebot.utils.ohlcv_cache import update_cache  # noqa: E402
from oraclebot.utils.telegram import send_message  # noqa: E402

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')
STATE_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'state')
SELECTION_PATH = os.path.join(STATE_DIR, 'trend_pool_selection.json')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('trend_pool_weekly')


def _secrets():
    with open(os.path.join(PROJECT_ROOT, 'secret.json'), encoding='utf-8') as f:
        return json.load(f)


def run_selection(cfg: dict, exchange, equity: float, now: pd.Timestamp = None) -> dict:
    now = now or pd.Timestamp.now(tz='UTC')
    ws = week_start(now)
    ids = pool_ids(cfg)
    trades, prices = {}, {}
    for c in cfg['coins']:
        h1 = update_cache(c, cfg['anchor'], DATASETS_DIR, now=now)
        prices[c] = float(h1['close'].iloc[-1])
        trades.update(simulate_coin(h1, c, cfg['base_pct_1h_by_coin'][c], ids, cfg['k_entropy'], cfg['h_window'],
                                    cfg['horizontal_lookback'], cfg['breakout_run'], cfg['cost_pct'], cfg['funding_pct_8h']))
    markets = exchange.markets
    min_amount = {c: float(markets[f'{c}/USDT:USDT']['limits']['amount']['min'] or 0) for c in cfg['coins']}
    slot_notional = equity / cfg['top_k'] * cfg['leverage']
    allowed = tradable_coins(cfg['coins'], prices, min_amount, slot_notional)
    sel = select_strategies(trades, ws, cfg['lookback_weeks'], cfg['top_k'], cfg.get('max_per_coin', 1), allowed_coins=allowed)
    return {'week_start': ws.isoformat(), 'created': now.isoformat(), 'equity': equity, 'slot_notional': slot_notional,
            'lookback_weeks': cfg['lookback_weeks'], 'top_k': cfg['top_k'],
            'not_tradable': sorted(set(cfg['coins']) - allowed),
            'selected': [{'id': r['id'], 'score': round(r['score'], 3), 'n': r['n']} for r in sel]}


def format_selection(sel: dict) -> str:
    lines = [f"ORACLEBOT Trend-Pool: Auswahl Woche ab {sel['week_start'][:10]}",
             f"Kapital {sel['equity']:.2f} USDT, Slot {sel['slot_notional']:.2f} USDT Nominal, Rueckblick {sel['lookback_weeks']}W",
             f"Nicht handelbar (Mindestorder): {', '.join(sel['not_tradable']) or '-'}"]
    lines += [f"{i + 1}. {r['id']}  Score {r['score']:+.2f}% ({r['n']} Trades)" for i, r in enumerate(sel['selected'])]
    if not sel['selected']:
        lines.append('Keine Strategie mit positivem Score -- diese Woche keine neuen Trades.')
    return '\n'.join(lines)


def save_selection(sel: dict):
    os.makedirs(STATE_DIR, exist_ok=True)
    tmp = SELECTION_PATH + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(sel, f, indent=2)
    os.replace(tmp, SELECTION_PATH)


def main():
    """Manueller Aufruf (z.B. zur Kontrolle). Im Betrieb erstellt scripts/trend_pool_live.py die Auswahl selbst,
    sobald eine neue UTC-Woche beginnt -- ein eigener Cronjob ist dafuer nicht noetig."""
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    cfg = load_settings()['trend_pool_settings']
    if not cfg.get('enabled', False):
        logger.info('trend_pool_settings.enabled=false -- nichts zu tun.')
        return
    secrets = _secrets()
    from oraclebot.utils.exchange import Exchange
    ex = Exchange(secrets['oraclebot'][0])
    sel = run_selection(cfg, ex.exchange, ex.fetch_margin_balances()['realized_equity'])
    msg = format_selection(sel)
    print(msg)
    if a.dry_run:
        return
    save_selection(sel)
    tg = secrets.get('telegram', {})
    send_message(tg.get('bot_token'), tg.get('chat_id'), msg)


if __name__ == '__main__':
    main()
