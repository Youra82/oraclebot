import numpy as np
import pandas as pd
import pytest

from oraclebot.analysis.trend_pool_portfolio import run_portfolio, tradable_coins
from oraclebot.strategy.trend_pool import (parse_strategy_id, pool_ids, simulate_coin, strategy_id)
from oraclebot.strategy.trend_pool_decide import decide_actions, open_trade_at
from oraclebot.strategy.trend_pool_select import select_strategies, week_start
from oraclebot.utils import ohlcv_cache

K, HW, LB, RUN = 0.7, 15, 6, 2


def make_h1(n=3000, seed=1, start='2024-01-01', drift=0.0):
    rng = np.random.default_rng(seed)
    # Trendphasen + Rauschen, damit alle Einstiegsarten Trades erzeugen
    regime = np.repeat(rng.choice([-1, 1], size=n // 200 + 1), 200)[:n]
    r = regime * 0.0015 + drift + rng.normal(0, 0.006, n)
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.002, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.002, n)))
    idx = pd.date_range(start, periods=n, freq='1h', tz='UTC')
    return pd.DataFrame({'open': open_, 'high': high, 'low': low, 'close': close, 'volume': 1.0}, index=idx)


IDS = [strategy_id('X', h, m, e) for h in ('4h', '1D') for m in (1, 2) for e in ('A', 'B', 'C', 'D')]


def _sim(h1):
    return simulate_coin(h1, 'X', 0.006, IDS, K, HW, LB, RUN)


def test_strategy_id_roundtrip():
    assert parse_strategy_id(strategy_id('SOL', '4h', 1.5, 'C')) == ('SOL', '4h', 1.5, 'C')
    cfg = {'coins': ['A', 'B'], 'htf_abc': ['4h'], 'mults': [1, 2], 'entries_abc': ['A', 'C'], 'htf_d': ['1h'], 'include_d': True}
    assert len(pool_ids(cfg)) == 2 * (1 * 2 * 2 + 1 * 2)


def test_all_entry_types_produce_trades():
    res = _sim(make_h1())
    for e in ('A', 'B', 'C', 'D'):
        assert any(len(res[i]) > 0 for i in IDS if i.endswith('|' + e)), e


@pytest.mark.parametrize('cut', [700, 1234, 1999, 2500, 2999])
def test_live_equals_backtest_on_truncated_history(cut):
    """Kernzusicherung: Live rechnet zu jeder Stunde mit der Historie BIS zur letzten fertigen Kerze. Die daraus
    abgeleiteten Trades muessen exakt dem vollen Backtest bis zu diesem Zeitpunkt entsprechen, und der dann offene
    Trade muss derselbe sein -- sonst wuerden Live und Backtest auseinanderlaufen."""
    h1 = make_h1()
    full = _sim(h1)
    part = _sim(h1.iloc[:cut])
    t_now = h1.index[cut - 1] + pd.Timedelta(hours=1)          # Schluss der letzten Kerze im Ausschnitt
    for sid in IDS:
        f_done = [t for t in full[sid] if t['exit_ts'] is not None and t['exit_ts'] <= t_now]
        p_done = [t for t in part[sid] if t['exit_ts'] is not None]
        assert [(t['entry_ts'], t['exit_ts'], round(t['pnl_pct'], 10)) for t in p_done] == \
               [(t['entry_ts'], t['exit_ts'], round(t['pnl_pct'], 10)) for t in f_done], sid
        f_open, p_open = open_trade_at(full[sid], t_now), open_trade_at(part[sid], t_now)
        assert (f_open is None) == (p_open is None), sid
        if f_open is not None:
            assert (f_open['entry_ts'], f_open['dir'], f_open['entry_px']) == (p_open['entry_ts'], p_open['dir'], p_open['entry_px'])


def test_fills_are_real_closes_not_brick_edges():
    h1 = make_h1()
    closes = pd.Series(h1['close'].to_numpy(), index=h1.index + pd.Timedelta(hours=1))
    for sid, trades in _sim(h1).items():
        for t in trades:
            assert t['entry_px'] == closes[t['entry_ts']]
            if t['exit_ts'] is not None:
                assert t['exit_px'] == closes[t['exit_ts']]


def _tr(entry, exit_, d=1, px=10.0, sid='X|4h|1|C'):
    return {'id': sid, 'coin': sid.split('|')[0], 'dir': d, 'entry_ts': pd.Timestamp(entry, tz='UTC'),
            'entry_px': px, 'exit_ts': pd.Timestamp(exit_, tz='UTC') if exit_ else None, 'pnl_pct': 1.0}


def test_decide_opens_only_at_entry_bar_within_grace_and_week():
    ws = pd.Timestamp('2026-09-21', tz='UTC')
    trades = {'X|4h|1|C': [_tr('2026-09-20 10:00', None), ], 'Y|4h|1|C': [_tr('2026-09-22 10:00', None, sid='Y|4h|1|C')]}
    sel = ['X|4h|1|C', 'Y|4h|1|C']
    a = decide_actions(pd.Timestamp('2026-09-22 10:00', tz='UTC'), ws, trades, sel, {}, set(), 0)
    assert [x['coin'] for x in a] == ['Y']                     # X begann vor Wochenstart -> kein Einstieg
    assert decide_actions(pd.Timestamp('2026-09-22 12:00', tz='UTC'), ws, trades, sel, {}, set(), 1) == []
    assert len(decide_actions(pd.Timestamp('2026-09-22 12:00', tz='UTC'), ws, trades, sel, {}, set(), 2)) == 1
    done = {('Y|4h|1|C', pd.Timestamp('2026-09-22 10:00', tz='UTC'))}
    assert decide_actions(pd.Timestamp('2026-09-22 10:00', tz='UTC'), ws, trades, sel, {}, done, 0) == []


def test_decide_closes_on_exit_and_reverses_d_same_bar():
    ws = pd.Timestamp('2026-09-21', tz='UTC')
    sid = 'X|4h|1|D'
    trades = {sid: [_tr('2026-09-21 04:00', '2026-09-22 08:00', 1, sid=sid), _tr('2026-09-22 08:00', None, -1, sid=sid)]}
    pos = {'X': {'id': sid, 'dir': 1, 'entry_ts': pd.Timestamp('2026-09-21 04:00', tz='UTC')}}
    a = decide_actions(pd.Timestamp('2026-09-22 08:00', tz='UTC'), ws, trades, [sid], pos, set(), 0)
    assert [x['action'] for x in a] == ['close', 'open'] and a[1]['dir'] == -1
    # abgewaehlte Strategie: Schliessen ja, kein neuer Einstieg
    a = decide_actions(pd.Timestamp('2026-09-22 08:00', tz='UTC'), ws, trades, [], pos, set(), 0)
    assert [x['action'] for x in a] == ['close']


def test_decide_one_position_per_coin_and_signal_lost():
    ws = pd.Timestamp('2026-09-21', tz='UTC')
    t1, t2 = 'X|4h|1|C', 'X|1D|1|A'
    trades = {t1: [_tr('2026-09-22 10:00', None, sid=t1)], t2: [_tr('2026-09-22 10:00', None, sid=t2)]}
    a = decide_actions(pd.Timestamp('2026-09-22 10:00', tz='UTC'), ws, trades, [t1, t2], {}, set(), 0)
    assert len(a) == 1 and a[0]['id'] == t1
    pos = {'X': {'id': t1, 'dir': 1, 'entry_ts': pd.Timestamp('2026-09-21 03:00', tz='UTC')}}
    a = decide_actions(pd.Timestamp('2026-09-22 11:00', tz='UTC'), ws, trades, [], pos, set(), 0)
    assert a == [{'action': 'close', 'coin': 'X', 'id': t1, 'reason': 'signal_lost'}]


def test_selection_score_window_positive_only_and_max_per_coin():
    ws = pd.Timestamp('2026-09-21', tz='UTC')

    def t(exit_, pnl):
        return {'entry_ts': pd.Timestamp(exit_, tz='UTC') - pd.Timedelta(hours=5), 'exit_ts': pd.Timestamp(exit_, tz='UTC'), 'pnl_pct': pnl}
    trades = {'A|4h|1|C': [t('2026-09-20', 5.0)], 'A|1D|1|C': [t('2026-09-19', 3.0)], 'B|4h|1|C': [t('2026-08-01', 50.0), t('2026-09-10', 1.0)],
              'C|4h|1|C': [t('2026-09-18', -2.0)], 'D|4h|1|C': [t('2026-09-21 01:00', 99.0)]}
    sel = select_strategies(trades, ws, 4, 10, 1)
    assert [r['id'] for r in sel] == ['A|4h|1|C', 'B|4h|1|C']      # B: alter Trade ausserhalb, D: nach Wochenstart
    assert [r['id'] for r in select_strategies(trades, ws, 4, 10, 1, allowed_coins={'B'})] == ['B|4h|1|C']
    assert week_start(pd.Timestamp('2026-09-27 13:00', tz='UTC')) == ws


def test_tradable_coins_respects_min_order():
    assert tradable_coins(['A', 'B', 'C'], {'A': 100.0, 'B': 1.0, 'C': 1.0}, {'A': 0.1, 'B': 1.0, 'C': 1.0}, 7.5) == {'B', 'C'}
    assert tradable_coins(['B'], {'B': 1.0}, {'B': 1.0}, 4.0) == set()      # unter 5 USDT Bitget-Minimum


def test_portfolio_runs_and_uses_same_selection():
    h1 = {'X': make_h1(4000, 3), 'Y': make_h1(4000, 4)}
    trades = {}
    for c in h1:
        trades.update(simulate_coin(h1[c], c, 0.006, [i.replace('X|', c + '|') for i in IDS], K, HW, LB, RUN))
    cfg = {'coins': ['X', 'Y'], 'lookback_weeks': 4, 'top_k': 2, 'max_per_coin': 1, 'leverage': 3, 'safety_stop_pct': 25}
    r = run_portfolio(trades, h1, h1['X'].index[0] + pd.Timedelta(days=35), h1['X'].index[-1], cfg, {'X': 0.0, 'Y': 0.0}, 100.0)
    assert r['equity_end'] > 0 and len(r['weekly']) > 10
    if len(r['trades']):
        assert set(r['trades'].coin) <= {'X', 'Y'}
        for _, d in r['trades'].groupby('coin'):
            assert (d.entry_ts.values[1:] >= d.exit_ts.values[:-1]).all()   # nie zwei Positionen je Coin gleichzeitig


class _FakeEx:
    """bildet Bitgets history-candles nach: Kerzen mit Oeffnungszeit in [startTime, endTime), max. limit"""
    def __init__(self, bars):
        self.bars = bars

    def publicMixGetV2MixMarketHistoryCandles(self, p):
        s, e, lim = int(p['startTime']), int(p['endTime']), int(p['limit'])
        return {'data': [[str(b[0])] + [str(x) for x in b[1:]] + ['0'] for b in self.bars if s <= b[0] < e][-lim:]}


def test_ohlcv_cache_excludes_running_bar_and_is_append_only(tmp_path):
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    bars = [[int((t0 + pd.Timedelta(hours=i)).timestamp() * 1000), 1, 2, 0.5, 1.5 + i, 10] for i in range(30)]
    now = t0 + pd.Timedelta(hours=25, minutes=10)          # Kerze 25:00 laeuft noch
    df = ohlcv_cache.update_cache('X', '2026-01-01', str(tmp_path), now=now, exchange=_FakeEx(bars))
    assert df.index[-1] == t0 + pd.Timedelta(hours=24)
    bars[3][4] = 999.0                                      # nachtraegliche Aenderung darf den Cache nicht umschreiben
    df2 = ohlcv_cache.update_cache('X', '2026-01-01', str(tmp_path), now=now + pd.Timedelta(hours=2), exchange=_FakeEx(bars))
    assert df2['close'].iloc[3] == 4.5 and df2.index[-1] == t0 + pd.Timedelta(hours=26)


def test_ohlcv_cache_raises_when_stale(tmp_path):
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    bars = [[int((t0 + pd.Timedelta(hours=i)).timestamp() * 1000), 1, 2, 0.5, 1.5, 10] for i in range(5)]
    with pytest.raises(ohlcv_cache.OhlcvFetchError):
        ohlcv_cache.update_cache('X', '2026-01-01', str(tmp_path), now=t0 + pd.Timedelta(hours=20), exchange=_FakeEx(bars))


class _FlakyEx(_FakeEx):
    """liefert fuer den ersten Abruf ab `flaky_since` einmal eine leere Antwort (wie Bitget unter Last)"""
    def __init__(self, bars, flaky_since):
        super().__init__(bars)
        self.flaky_since, self.hit = flaky_since, False

    def publicMixGetV2MixMarketHistoryCandles(self, p):
        if int(p['startTime']) == self.flaky_since and not self.hit:
            self.hit = True
            return {'data': []}
        return super().publicMixGetV2MixMarketHistoryCandles(p)


def test_ohlcv_cache_transient_empty_response_loses_no_bars(tmp_path):
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    bars = [[int((t0 + pd.Timedelta(hours=i)).timestamp() * 1000), 1, 2, 0.5, 1.5, 10] for i in range(700)]
    flaky = bars[200][0]                                     # zweiter 200er-Block kommt zuerst leer zurueck
    df = ohlcv_cache.update_cache('X', '2026-01-01', str(tmp_path), now=t0 + pd.Timedelta(hours=700, minutes=5),
                                  exchange=_FlakyEx(bars, flaky), sleep_on_empty=0)
    assert len(df) == 700 and df.index.to_series().diff().max() == pd.Timedelta(hours=1)


def test_ohlcv_cache_real_gap_is_skipped_exactly(tmp_path):
    t0 = pd.Timestamp('2026-01-01', tz='UTC')
    hours = [i for i in range(900) if not (300 <= i < 650)]  # echte Luecke von 350 Stunden
    bars = [[int((t0 + pd.Timedelta(hours=i)).timestamp() * 1000), 1, 2, 0.5, 1.5, 10] for i in hours]
    df = ohlcv_cache.update_cache('X', '2026-01-01', str(tmp_path), now=t0 + pd.Timedelta(hours=900, minutes=5),
                                  exchange=_FakeEx(bars), sleep_on_empty=0)
    assert len(df) == len(hours) and df.index[300] == t0 + pd.Timedelta(hours=650)
