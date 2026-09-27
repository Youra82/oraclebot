# src/oraclebot/strategy/trend_pool_select.py
# Woechentliche Auswahl aus dem Trend-Pool (Prinzip wie run_analysis.sh Mode 1 der anderen Bots):
# Score = Summe pnl_pct aller Trades, die im Rueckblick-Fenster [Wochenstart - lookback, Wochenstart)
# GESCHLOSSEN wurden -- nur Information, die zum Wochenstart wirklich bekannt ist. Top-K mit Score > 0,
# hoechstens `max_per_coin` Strategien je Coin. Dieselbe Funktion nutzen Live-Auswahl und Portfolio-Backtest.
import pandas as pd


def week_start(ts: pd.Timestamp) -> pd.Timestamp:
    """Montag 00:00 UTC der Woche von `ts`."""
    ts = pd.Timestamp(ts).tz_convert('UTC') if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts).tz_localize('UTC')
    d = ts.normalize()
    return d - pd.Timedelta(days=d.weekday())


def score_strategy(trades: list, ws: pd.Timestamp, lookback_weeks: int) -> tuple:
    lo = ws - pd.Timedelta(weeks=lookback_weeks)
    closed = [t['pnl_pct'] for t in trades if t.get('exit_ts') is not None and lo <= t['exit_ts'] < ws]
    return sum(closed), len(closed)


def select_strategies(trades_by_id: dict, ws: pd.Timestamp, lookback_weeks: int, top_k: int,
                      max_per_coin: int = 1, allowed_coins: set = None, min_trades: int = 1) -> list:
    """Rueckgabe: Liste {id, coin, score, n} der ausgewaehlten Strategien, bester Score zuerst."""
    rows = []
    for sid, trades in trades_by_id.items():
        coin = sid.split('|')[0]
        if allowed_coins is not None and coin not in allowed_coins:
            continue
        s, n = score_strategy(trades, ws, lookback_weeks)
        if n >= min_trades and s > 0:
            rows.append({'id': sid, 'coin': coin, 'score': s, 'n': n})
    rows.sort(key=lambda r: (-r['score'], r['id']))
    chosen, per_coin = [], {}
    for r in rows:
        if per_coin.get(r['coin'], 0) >= max_per_coin:
            continue
        chosen.append(r)
        per_coin[r['coin']] = per_coin.get(r['coin'], 0) + 1
        if len(chosen) >= top_k:
            break
    return chosen
