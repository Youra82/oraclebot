import pandas as pd
import pytest

from oraclebot.analysis.live_signal_check import (_reference_period_start, append_report_to_log,
                                                    build_signal_comparison, fetch_recent_live_trades,
                                                    format_telegram_report)
from oraclebot.data.live_features import FeatureReconstructionError


def test_reference_period_start_is_previous_4h_period():
    ctime = pd.Timestamp('2026-09-11 16:01:35', tz='UTC')
    assert _reference_period_start(ctime, '4h') == pd.Timestamp('2026-09-11 12:00:00', tz='UTC')


def test_reference_period_start_at_exact_boundary():
    ctime = pd.Timestamp('2026-09-11 12:00:00', tz='UTC')
    assert _reference_period_start(ctime, '4h') == pd.Timestamp('2026-09-11 08:00:00', tz='UTC')


class FakeExchange:
    def __init__(self, positions):
        self._positions = positions

    def fetch_closed_positions(self, symbol, limit=100):
        return self._positions


def _make_position(ctime_ms, utime_ms, hold_side, pnl, open_price=100.0, close_price=101.0):
    return {'info': {'ctime': str(ctime_ms), 'utime': str(utime_ms), 'holdSide': hold_side,
                      'pnl': str(pnl), 'netProfit': str(pnl * 0.9),
                      'openAvgPrice': str(open_price), 'closeAvgPrice': str(close_price)}}


def test_fetch_recent_live_trades_filters_by_since_and_derives_ref_ts():
    old_ctime = int(pd.Timestamp('2026-09-01 12:01:00', tz='UTC').timestamp() * 1000)
    recent_ctime = int(pd.Timestamp('2026-09-11 16:01:35', tz='UTC').timestamp() * 1000)
    positions = [
        _make_position(old_ctime, old_ctime + 1000, 'short', -1.0),
        _make_position(recent_ctime, recent_ctime + 1000, 'long', 2.0),
    ]
    exchange = FakeExchange(positions)
    since_ts = pd.Timestamp('2026-09-10', tz='UTC')

    trades = fetch_recent_live_trades(exchange, 'BTC/USDT:USDT', '4h', since_ts)

    assert len(trades) == 1
    assert trades[0]['direction'] == 'long'
    assert trades[0]['pnl'] == 2.0
    assert trades[0]['ref_ts'] == pd.Timestamp('2026-09-11 12:00:00', tz='UTC')


def test_fetch_recent_live_trades_sorted_chronologically():
    t1 = int(pd.Timestamp('2026-09-11 12:00:00', tz='UTC').timestamp() * 1000)
    t2 = int(pd.Timestamp('2026-09-10 08:00:00', tz='UTC').timestamp() * 1000)
    exchange = FakeExchange([_make_position(t1, t1, 'short', 1.0), _make_position(t2, t2, 'long', 1.0)])
    trades = fetch_recent_live_trades(exchange, 'BTC/USDT:USDT', '4h', pd.Timestamp('2026-09-01', tz='UTC'))
    assert trades[0]['ctime'] < trades[1]['ctime']


class FakePredictor:
    def __init__(self, cls, conf):
        self.cls = cls
        self.conf = conf

    def predict_one(self, feature_row):
        return self.cls, self.conf


def test_build_signal_comparison_counts_match_and_mismatch(monkeypatch):
    ctime = int(pd.Timestamp('2026-09-11 16:01:35', tz='UTC').timestamp() * 1000)
    exchange = FakeExchange([_make_position(ctime, ctime, 'short', -1.0)])
    predictor = FakePredictor(cls=1, conf=0.9)  # 'long' bei hoher Konfidenz -- Live war 'short'

    def fake_build_vector(symbol, reference_tf, context_tfs, barrier_cfg, artifacts_dir, ref_ts=None, now_utc=None,
                           min_candles=120, context_min_candles=None):
        return {'ref_ts': ref_ts, 'entry_price': 100.0, 'feature_row': [0.0], 'blocks': []}

    monkeypatch.setattr('oraclebot.analysis.live_signal_check.build_reference_feature_vector', fake_build_vector)

    barrier_cfg = {'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h', 'context_timeframes': [],
                   'min_confidence': 0.6, 'barrier_pct': 1.0}
    report = build_signal_comparison(barrier_cfg, predictor, exchange, '/tmp/artifacts',
                                      since_ts=pd.Timestamp('2026-09-01', tz='UTC'),
                                      now_utc=pd.Timestamp('2026-09-12', tz='UTC'))

    assert report['n_trades'] == 1
    assert report['n_comparable'] == 1
    assert report['n_match'] == 0
    assert report['n_mismatch'] == 1
    assert report['n_live_losses'] == 1
    assert report['trades'][0]['model_direction'] == 'long'
    assert report['trades'][0]['live_direction'] == 'short'


def test_build_signal_comparison_handles_feature_reconstruction_error(monkeypatch):
    ctime = int(pd.Timestamp('2026-09-11 16:01:35', tz='UTC').timestamp() * 1000)
    exchange = FakeExchange([_make_position(ctime, ctime, 'short', -1.0)])
    predictor = FakePredictor(cls=0, conf=0.9)

    def fake_build_vector(*args, **kwargs):
        raise FeatureReconstructionError("zu alt fuer den Live-Cache")

    monkeypatch.setattr('oraclebot.analysis.live_signal_check.build_reference_feature_vector', fake_build_vector)

    barrier_cfg = {'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h', 'context_timeframes': []}
    report = build_signal_comparison(barrier_cfg, predictor, exchange, '/tmp/artifacts',
                                      since_ts=pd.Timestamp('2026-09-01', tz='UTC'))

    assert report['n_trades'] == 1
    assert report['n_comparable'] == 0
    assert report['n_match'] == 0
    assert 'error' in report['trades'][0]


def test_build_signal_comparison_empty_when_no_trades():
    exchange = FakeExchange([])
    predictor = FakePredictor(cls=1, conf=0.9)
    barrier_cfg = {'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h', 'context_timeframes': []}
    report = build_signal_comparison(barrier_cfg, predictor, exchange, '/tmp/artifacts',
                                      since_ts=pd.Timestamp('2026-09-01', tz='UTC'))
    assert report['n_trades'] == 0
    assert report['live_win_rate'] is None


def test_format_telegram_report_no_trades():
    report = {'n_trades': 0, 'since': '2026-09-01T00:00:00+00:00'}
    text = format_telegram_report(report, 'BTC/USDT:USDT')
    assert 'keine geschlossenen Live-Trades' in text


def test_format_telegram_report_flags_mismatches():
    report = {'n_trades': 2, 'since': '2026-09-01T00:00:00+00:00', 'live_win_rate': 0.5,
              'n_match': 1, 'n_mismatch': 1, 'n_comparable': 2}
    text = format_telegram_report(report, 'BTC/USDT:USDT')
    assert '1 Mismatch' in text
    assert '1/2 Match' in text


def test_append_report_to_log_writes_jsonl_line(tmp_path):
    log_path = str(tmp_path / 'nested' / 'signal_comparison_history.jsonl')
    append_report_to_log({'n_trades': 1}, log_path)
    append_report_to_log({'n_trades': 2}, log_path)

    with open(log_path, encoding='utf-8') as f:
        lines = f.readlines()
    assert len(lines) == 2
    import json
    assert json.loads(lines[1])['n_trades'] == 2
