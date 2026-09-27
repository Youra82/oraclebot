# src/oraclebot/strategy/trend_pool_decide.py
# Reine Entscheidungslogik fuer den Trend-Pool: wird vom stuendlichen Live-Lauf (scripts/trend_pool_live.py)
# UND von der Portfolio-Simulation (analysis/trend_pool_portfolio.py) identisch aufgerufen.
#
# Regeln (entsprechen dem getesteten Wochen-Schema):
#   - Eine Position je Coin (Hedge-Modus wird nie beidseitig genutzt).
#   - Offene Position schliessen, sobald der zugehoerige Strategie-Trade laut Signalfunktion beendet ist
#     (oder der Trade in der neu berechneten Kette nicht mehr existiert -> Sicherheits-Schliessen).
#   - Neue Positionen NUR fuer Strategien der aktuellen Wochenauswahl, NUR fuer Trades, die in dieser Woche
#     beginnen, und nur, solange der Einstieg hoechstens `entry_grace_hours` zurueckliegt.
#   - Eine abgewaehlte Strategie behaelt ihre offene Position bis zu deren eigenem Ausstieg.
from bisect import bisect_right

import pandas as pd


def _entries(trades: list) -> list:
    # Trades sind chronologisch nach entry_ts (von simulate_trades so erzeugt)
    return [t['entry_ts'] for t in trades]


def _trade_by_entry(trades: list, entry_ts, entries: list = None):
    i = bisect_right(entries if entries is not None else _entries(trades), entry_ts) - 1
    return trades[i] if i >= 0 and trades[i]['entry_ts'] == entry_ts else None


def open_trade_at(trades: list, bar_close: pd.Timestamp, entries: list = None):
    """Der Trade, der zum Zeitpunkt `bar_close` offen ist (Einstieg <= bar_close < Ausstieg), sonst None."""
    i = bisect_right(entries if entries is not None else _entries(trades), bar_close) - 1
    if i < 0:
        return None
    t = trades[i]
    return t if (t['exit_ts'] is None or t['exit_ts'] > bar_close) else None


def decide_actions(bar_close: pd.Timestamp, week_start: pd.Timestamp, trades_by_id: dict, selected: list,
                   positions: dict, entered: set, entry_grace_hours: float = 0.0,
                   entries_by_id: dict = None) -> list:
    """Gibt Aktionen zurueck, zuerst alle 'close', dann alle 'open':
        {'action': 'close', 'coin', 'id', 'reason'}
        {'action': 'open', 'coin', 'id', 'dir', 'entry_ts', 'entry_px'}
    `positions`: {coin: {'id', 'dir', 'entry_ts'}}, `entered`: Menge (id, entry_ts) bereits eroeffneter Trades,
    `selected`: Liste von Strategie-IDs in Prioritaetsreihenfolge (bester Score zuerst).
    `entries_by_id`: optional vorberechnete entry_ts-Listen je Strategie (nur Geschwindigkeit)."""
    eb = entries_by_id or {}
    closes, opens = [], []
    freed = set()
    for coin, pos in positions.items():
        trades = trades_by_id.get(pos['id'])
        if trades is None:
            continue  # Strategie in diesem Lauf nicht berechnet -> nichts entscheiden
        t = _trade_by_entry(trades, pos['entry_ts'], eb.get(pos['id']))
        if t is None:
            closes.append({'action': 'close', 'coin': coin, 'id': pos['id'], 'reason': 'signal_lost'})
            freed.add(coin)
        elif t['exit_ts'] is not None and t['exit_ts'] <= bar_close:
            closes.append({'action': 'close', 'coin': coin, 'id': pos['id'], 'reason': t.get('exit_reason', 'exit')})
            freed.add(coin)
    busy = {c for c in positions if c not in freed}
    for sid in selected:
        coin = sid.split('|')[0]
        if coin in busy:
            continue
        trades = trades_by_id.get(sid)
        if not trades:
            continue
        t = open_trade_at(trades, bar_close, eb.get(sid))
        if t is None or t['entry_ts'] < week_start or (sid, t['entry_ts']) in entered:
            continue
        if (bar_close - t['entry_ts']).total_seconds() / 3600 > entry_grace_hours:
            continue
        opens.append({'action': 'open', 'coin': coin, 'id': sid, 'dir': t['dir'], 'entry_ts': t['entry_ts'],
                      'entry_px': t['entry_px']})
        busy.add(coin)
    return closes + opens
