# src/oraclebot/strategy/trend_pool.py
# Trend-Pool: Zwei-Ebenen-Renko-Strategien (User-Idee 2026-09-27). EINE Signalfunktion fuer Backtest,
# Portfolio-Simulation UND Live-Bot -- Live berechnet die Trades exakt so wie der Backtest (volle Kette ab
# `anchor`), der Live-Bot handelt nur die jeweils juengste Aenderung. Dadurch koennen Live und Backtest
# strukturell nicht auseinanderlaufen (Lehre aus [[research_zerobot_live_vs_backtest_2026_08]] und dem
# Brick-Preis-Fehler: alle Fills hier am ECHTEN Schlusskurs der 1h-Kerze, nie an einer Brick-Kante).
#
# Strategie = (Coin, Trend-Zeitrahmen htf, Brick-Multiplikator mult, Einstieg):
#   Trend      = Richtung des letzten Bricks auf htf (nur abgeschlossene htf-Kerzen).
#   Einstieg A = erster 1h-Brick in Trendrichtung nach mind. einem Gegen-Brick (Ruecksetzer-Ende)
#   Einstieg C = zweiter 1h-Brick in Trendrichtung nach einem Ruecksetzer (User-Skizze)
#   Einstieg B = Ausbruch: `horizontal_lookback` gemischte 1h-Bricks, dann `breakout_run` gleichgerichtete
#   Einstieg D = sofort beim Dreher des htf-Bricks (Stop-and-Reverse auf htf-Renko)
#   Ausstieg   = sobald der htf-Brick die Richtung wechselt (bei D: gleichzeitig Gegenposition).
import numpy as np
import pandas as pd

from oraclebot.data.ear_bricks import build_ear_bricks

TF_HOURS = {'1h': 1, '2h': 2, '4h': 4, '8h': 8, '1D': 24}
_PANDAS_RULE = {'1h': '1h', '2h': '2h', '4h': '4h', '8h': '8h', '1D': '1D'}


def strategy_id(coin: str, htf: str, mult: float, entry: str) -> str:
    return f"{coin}|{htf}|{mult:g}|{entry}"


def parse_strategy_id(sid: str) -> tuple:
    coin, htf, mult, entry = sid.split('|')
    return coin, htf, float(mult), entry


def pool_ids(cfg: dict) -> list:
    """Alle Strategie-IDs des Pools laut settings.json::trend_pool_settings."""
    ids = []
    for coin in cfg['coins']:
        for htf in cfg['htf_abc']:
            for mult in cfg['mults']:
                for entry in cfg['entries_abc']:
                    ids.append(strategy_id(coin, htf, mult, entry))
        if cfg.get('include_d', True):
            for htf in cfg['htf_d']:
                for mult in cfg['mults']:
                    ids.append(strategy_id(coin, htf, mult, 'D'))
    return ids


def resample_ohlc(h1: pd.DataFrame, tf: str) -> pd.DataFrame:
    if tf == '1h':
        return h1
    return h1.resample(_PANDAS_RULE[tf], label='left', closed='left').agg(
        {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()


def brick_dirs_per_bar(df: pd.DataFrame, base_pct: float, k_entropy: float, h_window: int) -> list:
    """Je Kerze die Liste der dort neu entstandenen Brick-Richtungen (+1/-1), via build_ear_bricks."""
    out = [[] for _ in range(len(df))]
    for b in build_ear_bricks(df, base_pct=base_pct, k_entropy=k_entropy, h_window=h_window):
        out[b['candle_idx']].append(1 if b['direction'] == 'up' else -1)
    return out


def htf_trend_on_1h(h1: pd.DataFrame, base_1h: float, htf: str, mult: float, k_entropy: float,
                    h_window: int) -> np.ndarray:
    """Trendrichtung (+1/-1/0) zu jedem 1h-KERZENSCHLUSS: Richtung des letzten htf-Bricks aus htf-Kerzen,
    die zu diesem Zeitpunkt schon abgeschlossen sind (kein Blick in die laufende htf-Kerze)."""
    hdf = resample_ohlc(h1, htf)
    base_htf = base_1h * mult * np.sqrt(TF_HOURS[htf])
    ev = brick_dirs_per_bar(hdf, base_htf, k_entropy, h_window)
    cur, dirs = 0, []
    for e in ev:
        if e:
            cur = e[-1]
        dirs.append(cur)
    trend = pd.Series(dirs, index=hdf.index + pd.Timedelta(hours=TF_HOURS[htf]))
    bar_close = h1.index + pd.Timedelta(hours=1)
    return trend.reindex(bar_close, method='ffill').fillna(0).astype(int).to_numpy()


def _entry_signals(ltf_events: list, entry: str, lb: int, run: int) -> np.ndarray:
    """Einstiegssignal (+1/-1/0) je 1h-Kerze aus den 1h-Bricks (fuer A/B/C, unabhaengig vom Trend)."""
    sig = np.zeros(len(ltf_events), dtype=int)
    bricks, run_dir, run_len = [], None, 0
    for i, evs in enumerate(ltf_events):
        s = 0
        for d in evs:
            prev = bricks[-1] if bricks else None
            bricks.append(d)
            k = len(bricks) - 1
            run_len = run_len + 1 if d == run_dir else 1
            run_dir = d
            if entry == 'A':
                if prev is not None and prev == -d:
                    s = d
            elif entry == 'C':
                if run_len == 2 and k >= 2 and bricks[k - 2] == -d:
                    s = d
            elif entry == 'B':
                if run_len == run:
                    ws = k - run + 1 - lb
                    if ws >= 0 and len(set(bricks[ws:k - run + 1])) > 1:
                        s = d
        if len(bricks) > 200:
            bricks = bricks[-50:]
        sig[i] = s
    return sig


def simulate_trades(h1: pd.DataFrame, trend: np.ndarray, entry: str, entry_sig: np.ndarray = None,
                    cost_pct: float = 0.16, funding_pct_8h: float = 0.01) -> list:
    """Trades einer Strategie ueber die ganze Historie. Zeitstempel = 1h-KERZENSCHLUSS (UTC), Fill =
    Schlusskurs dieser Kerze. Der letzte Trade hat exit_ts=None, falls er am Ende noch offen ist.
    pnl_pct nach Kosten (Gebuehr+Slippage) und pauschalem Funding (Backtest-Annahme)."""
    c = h1['close'].to_numpy()
    closes_ts = h1.index + pd.Timedelta(hours=1)
    trades, pos = [], None

    def _close(i, reason):
        hours = (closes_ts[i] - pos['entry_ts']).total_seconds() / 3600
        pos.update(exit_ts=closes_ts[i], exit_px=float(c[i]), hours=hours, exit_reason=reason,
                   pnl_pct=pos['dir'] * (c[i] - pos['entry_px']) / pos['entry_px'] * 100
                   - cost_pct - funding_pct_8h * hours / 8)

    for i in range(1, len(c)):
        if entry == 'D':
            if trend[i] != trend[i - 1] and trend[i] != 0:
                if pos is not None:
                    _close(i, 'htf_flip'); pos = None
                if trend[i - 1] != 0:
                    pos = {'dir': int(trend[i]), 'entry_ts': closes_ts[i], 'entry_px': float(c[i]), 'exit_ts': None}
                    trades.append(pos)
            continue
        if pos is not None and trend[i] != pos['dir']:
            _close(i, 'htf_flip'); pos = None
        if pos is None and entry_sig[i] != 0 and entry_sig[i] == trend[i]:
            pos = {'dir': int(entry_sig[i]), 'entry_ts': closes_ts[i], 'entry_px': float(c[i]), 'exit_ts': None}
            trades.append(pos)
    return trades


def simulate_coin(h1: pd.DataFrame, coin: str, base_1h: float, ids: list, k_entropy: float, h_window: int,
                  lb: int, run: int, cost_pct: float = 0.16, funding_pct_8h: float = 0.01) -> dict:
    """Alle Strategien EINES Coins; Trend- und Einstiegsreihen werden je (htf, mult) bzw. Einstieg nur einmal
    berechnet. Rueckgabe {strategy_id: [trades]} mit id/coin in jedem Trade."""
    ltf_events = None
    trend_cache, sig_cache, out = {}, {}, {}
    for sid in ids:
        c_, htf, mult, entry = parse_strategy_id(sid)
        if c_ != coin:
            continue
        key = (htf, mult)
        if key not in trend_cache:
            trend_cache[key] = htf_trend_on_1h(h1, base_1h, htf, mult, k_entropy, h_window)
        sig = None
        if entry != 'D':
            if ltf_events is None:
                ltf_events = brick_dirs_per_bar(h1, base_1h, k_entropy, h_window)
            if entry not in sig_cache:
                sig_cache[entry] = _entry_signals(ltf_events, entry, lb, run)
            sig = sig_cache[entry]
        trades = simulate_trades(h1, trend_cache[key], entry, sig, cost_pct, funding_pct_8h)
        for t in trades:
            t['id'] = sid
            t['coin'] = coin
        out[sid] = trades
    return out
