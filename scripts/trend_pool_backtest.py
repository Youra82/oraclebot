# scripts/trend_pool_backtest.py
# Portfolio-Backtest des Trend-Pools mit exakt den Live-Funktionen (Signal, Auswahl, Entscheidung).
#   .venv/bin/python3 scripts/trend_pool_backtest.py                 # Cache aktualisieren, 2023-06 bis heute
#   .venv/bin/python3 scripts/trend_pool_backtest.py --no-fetch --start 2025-01-01 --equity 25 --top-k 5 --leverage 3
import argparse
import logging
import os
import sys

import ccxt
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from oraclebot.analysis.trend_pool_portfolio import run_portfolio  # noqa: E402
from oraclebot.strategy.trend_pool import pool_ids, simulate_coin  # noqa: E402
from oraclebot.utils.config import PROJECT_ROOT, load_settings  # noqa: E402
from oraclebot.utils.ohlcv_cache import cache_path, update_cache  # noqa: E402

DATASETS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('trend_pool_backtest')


def load_all(cfg: dict, fetch: bool) -> dict:
    h1 = {}
    for c in cfg['coins']:
        if fetch:
            h1[c] = update_cache(c, cfg['anchor'], DATASETS_DIR)
        else:
            h1[c] = pd.read_pickle(cache_path(DATASETS_DIR, c))
    return h1


def simulate_all(cfg: dict, h1: dict) -> dict:
    ids = pool_ids(cfg)
    out = {}
    for c in cfg['coins']:
        out.update(simulate_coin(h1[c], c, cfg['base_pct_1h_by_coin'][c], ids, cfg['k_entropy'], cfg['h_window'],
                                 cfg['horizontal_lookback'], cfg['breakout_run'], cfg['cost_pct'], cfg['funding_pct_8h']))
    return out


def min_amounts(coins: list) -> dict:
    m = ccxt.bitget({'options': {'defaultType': 'swap'}}).load_markets()
    return {c: float(m[f'{c}/USDT:USDT']['limits']['amount']['min'] or 0) for c in coins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--no-fetch', action='store_true')
    ap.add_argument('--start', default='2023-06-05')
    ap.add_argument('--end', default=None)
    ap.add_argument('--equity', type=float, default=25.0)
    ap.add_argument('--lookback', type=int, default=None)
    ap.add_argument('--top-k', type=int, default=None)
    ap.add_argument('--leverage', type=float, default=None)
    a = ap.parse_args()
    cfg = dict(load_settings()['trend_pool_settings'])
    for k, v in (('lookback_weeks', a.lookback), ('top_k', a.top_k), ('leverage', a.leverage)):
        if v is not None:
            cfg[k] = v
    h1 = load_all(cfg, not a.no_fetch)
    trades = simulate_all(cfg, h1)
    end = pd.Timestamp(a.end, tz='UTC') if a.end else min(h.index[-1] for h in h1.values()) + pd.Timedelta(hours=1)
    res = run_portfolio(trades, h1, pd.Timestamp(a.start, tz='UTC'), end, cfg, min_amounts(cfg['coins']), a.equity)
    w, tr = res['weekly'], res['trades']
    print(f"\nTrend-Pool Portfolio {a.start} bis {end:%Y-%m-%d} | Rueckblick {cfg['lookback_weeks']}W, Top {cfg['top_k']}, "
          f"Hebel {cfg['leverage']}x, Stop {cfg['safety_stop_pct']}%")
    print(f"Start {a.equity:.2f} USDT -> Ende {res['equity_end']:.2f} USDT | Max-Drawdown {res['max_dd_pct']:.1f}% | "
          f"Trades {len(tr)} | ausgelassen {res['skipped']}")
    if len(tr):
        print(f"Winrate {(tr.pnl_usdt > 0).mean() * 100:.1f}% | Sicherheits-Stops {(tr.reason == 'safety_stop').sum()}")
        by_year = w.groupby(w.index.year)['equity'].last()
        print('Kapital je Jahresende:', ' | '.join(f"{y}: {v:.2f}" for y, v in by_year.items()))
        print(f"Wochen im Plus {(w.ret_pct > 0).mean() * 100:.0f}% | schlechteste Woche {w.ret_pct.min():+.1f}% | beste {w.ret_pct.max():+.1f}%")
    out = os.path.join(PROJECT_ROOT, 'artifacts', 'results')
    os.makedirs(out, exist_ok=True)
    tr.to_csv(os.path.join(out, 'trend_pool_backtest_trades.csv'), index=False)
    w.drop(columns=['selected']).to_csv(os.path.join(out, 'trend_pool_backtest_weekly.csv'))


if __name__ == '__main__':
    main()
