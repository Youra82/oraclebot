# src/oraclebot/analysis/trend_pool_portfolio.py
# Portfolio-Backtest des Trend-Pools mit DENSELBEN Funktionen wie der Live-Bot:
#   Auswahl  -> strategy/trend_pool_select.select_strategies (jeden Montag 00:00 UTC)
#   Aktionen -> strategy/trend_pool_decide.decide_actions (an jedem Ereigniszeitpunkt; zu allen anderen
#               Stunden erzeugt decide_actions per Konstruktion keine Aktion, daher gleichwertig zum
#               stuendlichen Live-Lauf)
# Kapitalmodell wie live: Slot-Marge = Kapital / top_k (hoechstens freie Marge), Nominal = Marge * Hebel,
# Positionen unter der Bitget-Mindestgroesse werden ausgelassen, Sicherheits-Stop als Boersen-Trigger bei
# `safety_stop_pct` Gegenlauf (1h-Hoch/Tief), Ergebnis je Trade = pnl_pct der Signalfunktion (Close-Fills,
# Gebuehr, Slippage, pauschales Funding).
import heapq
from bisect import bisect_left

import numpy as np
import pandas as pd

from oraclebot.strategy.trend_pool_decide import decide_actions, _entries
from oraclebot.strategy.trend_pool_select import select_strategies


def safety_stop_hit(h1: pd.DataFrame, trade: dict, stop_pct: float, closes_ts: pd.DatetimeIndex = None):
    """Erster Zeitpunkt (1h-Kerzenschluss) zwischen Einstieg und Ausstieg, an dem der Kurs `stop_pct` gegen
    die Position gelaufen ist; Rueckgabe (ts, fill_px) oder None."""
    if not stop_pct:
        return None
    cts = closes_ts if closes_ts is not None else h1.index + pd.Timedelta(hours=1)
    i0 = bisect_left(cts, trade['entry_ts']) + 1
    i1 = len(cts) if trade['exit_ts'] is None else bisect_left(cts, trade['exit_ts']) + 1
    if i0 >= i1:
        return None
    d, e = trade['dir'], trade['entry_px']
    stop = e * (1 - d * stop_pct / 100)
    seg = h1.iloc[i0:i1]
    hit = (seg['low'].to_numpy() <= stop) if d > 0 else (seg['high'].to_numpy() >= stop)
    if not hit.any():
        return None
    j = int(np.argmax(hit))
    o = float(seg['open'].iloc[j])
    fill = min(o, stop) if d > 0 else max(o, stop)
    if trade['exit_ts'] is not None and cts[i0 + j] >= trade['exit_ts']:
        return None  # Signal-Ausstieg kommt zuerst bzw. gleichzeitig
    return cts[i0 + j], fill


def last_close_before(h1: pd.DataFrame, ts: pd.Timestamp):
    i = bisect_left(h1.index + pd.Timedelta(hours=1), ts)
    return float(h1['close'].iloc[i - 1]) if i > 0 else None


def tradable_coins(coins: list, prices: dict, min_amount: dict, slot_notional: float, min_order_usdt: float = 5.0) -> set:
    """Coins, deren Bitget-Mindestorder in einen Slot passt (gleiche Regel live und im Backtest)."""
    return {c for c in coins if prices.get(c) and max(min_order_usdt, min_amount.get(c, 0.0) * prices[c]) <= slot_notional}


def run_portfolio(trades_by_id: dict, h1_by_coin: dict, start: pd.Timestamp, end: pd.Timestamp, cfg: dict,
                  min_amount: dict, start_equity: float = 25.0, min_order_usdt: float = 5.0) -> dict:
    lookback, top_k = cfg['lookback_weeks'], cfg['top_k']
    lev, stop_pct = cfg['leverage'], cfg.get('safety_stop_pct', 0)
    cost, fund = cfg.get('cost_pct', 0.16), cfg.get('funding_pct_8h', 0.01)
    entries_by_id = {sid: _entries(tr) for sid, tr in trades_by_id.items()}
    cts_by_coin = {c: h.index + pd.Timedelta(hours=1) for c, h in h1_by_coin.items()}
    equity, positions, entered = start_equity, {}, set()
    log, eq_curve, weekly, skipped = [], [], [], {'zu_klein': 0, 'keine_marge': 0}
    ws = start - pd.Timedelta(days=start.weekday())
    ws = ws.normalize()

    def pos_close(coin, ts, pnl_pct, reason):
        nonlocal equity
        p = positions.pop(coin)
        pnl = p['notional'] * pnl_pct / 100
        equity += pnl
        log.append({'coin': coin, 'id': p['id'], 'dir': p['dir'], 'entry_ts': p['entry_ts'], 'exit_ts': ts,
                    'pnl_pct': pnl_pct, 'pnl_usdt': pnl, 'notional': p['notional'], 'reason': reason})
        eq_curve.append((ts, equity))

    while ws < end:
        we = ws + pd.Timedelta(days=7)
        eq_week_start = equity
        allowed = tradable_coins(cfg['coins'], {c: last_close_before(h1_by_coin[c], ws) for c in cfg['coins']},
                                 min_amount, equity / top_k * lev, min_order_usdt)
        sel = select_strategies(trades_by_id, ws, lookback, top_k, cfg.get('max_per_coin', 1), allowed_coins=allowed)
        sel_ids = [r['id'] for r in sel]
        ev = []
        for sid in sel_ids:
            en = entries_by_id[sid]
            for i in range(bisect_left(en, ws), bisect_left(en, we)):
                heapq.heappush(ev, en[i])
        for p in positions.values():
            for tts in (p['stop_ts'], p['trade']['exit_ts']):
                if tts is not None and ws <= tts < we:
                    heapq.heappush(ev, tts)
        seen = set()
        while ev:
            T = heapq.heappop(ev)
            if T in seen or T >= we or T > end:
                continue
            seen.add(T)
            for coin in [c for c, p in positions.items() if p['stop_ts'] is not None and p['stop_ts'] == T]:
                p = positions[coin]
                hrs = (T - p['entry_ts']).total_seconds() / 3600
                pnl_pct = p['dir'] * (p['stop_px'] - p['entry_px']) / p['entry_px'] * 100 - cost - fund * hrs / 8
                pos_close(coin, T, pnl_pct, 'safety_stop')
            view = {c: {'id': p['id'], 'dir': p['dir'], 'entry_ts': p['entry_ts']} for c, p in positions.items()}
            for a in decide_actions(T, ws, trades_by_id, sel_ids, view, entered, 0.0, entries_by_id):
                if a['action'] == 'close':
                    p = positions[a['coin']]
                    pos_close(a['coin'], T, p['trade']['pnl_pct'], a['reason'])
                else:
                    used = sum(p['margin'] for p in positions.values())
                    margin = min(equity / top_k, equity - used)
                    notional = margin * lev
                    entered.add((a['id'], a['entry_ts']))
                    if margin <= 0:
                        skipped['keine_marge'] += 1
                        continue
                    if notional < max(min_order_usdt, min_amount.get(a['coin'], 0.0) * a['entry_px']):
                        skipped['zu_klein'] += 1
                        continue
                    tr = next(t for t in trades_by_id[a['id']] if t['entry_ts'] == a['entry_ts'])
                    hit = safety_stop_hit(h1_by_coin[a['coin']], tr, stop_pct, cts_by_coin[a['coin']])
                    positions[a['coin']] = {'id': a['id'], 'dir': a['dir'], 'entry_ts': a['entry_ts'],
                                            'entry_px': a['entry_px'], 'margin': margin, 'notional': notional,
                                            'trade': tr, 'stop_ts': hit[0] if hit else None,
                                            'stop_px': hit[1] if hit else None}
                    for tts in (positions[a['coin']]['stop_ts'], tr['exit_ts']):
                        if tts is not None and tts < we:
                            heapq.heappush(ev, tts)
        weekly.append({'week': ws, 'equity': equity, 'ret_pct': (equity / eq_week_start - 1) * 100 if eq_week_start > 0 else 0,
                       'selected': sel_ids})
        if equity <= 0:
            break
        ws = we
    eq = pd.Series([w['equity'] for w in weekly], index=[w['week'] for w in weekly])
    dd = (eq / eq.cummax() - 1).min() * 100 if len(eq) else 0.0
    return {'equity_end': equity, 'weekly': pd.DataFrame(weekly).set_index('week') if weekly else pd.DataFrame(),
            'trades': pd.DataFrame(log), 'max_dd_pct': dd, 'skipped': skipped, 'open_positions': positions}
