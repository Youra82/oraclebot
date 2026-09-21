import importlib.util
import os

import pandas as pd
import pytest

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_renko_breakout.py')
spec = importlib.util.spec_from_file_location('run_renko_breakout', SCRIPT_PATH)
run_renko_breakout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_renko_breakout)


class FakeExchange:
    """Simuliert nur fetch_open_positions() -- reconcile_portfolio_state() braucht nichts
    anderes. Kein echter API-Zugriff, damit dieser Test ohne Bitget-Verbindung laeuft."""

    def __init__(self, open_positions: dict):
        self._open = open_positions  # {symbol: {'side': 'long'/'short'}}

    def fetch_open_positions(self, symbol):
        pos = self._open.get(symbol)
        return [pos] if pos else []


def test_reconcile_keeps_state_when_matching():
    ex = FakeExchange({'NEAR/USDT:USDT': {'side': 'long'}})
    state = {'active_symbol': 'NEAR/USDT:USDT', 'active_direction': 'long'}
    result = run_renko_breakout.reconcile_portfolio_state(
        ex, ['NEAR/USDT:USDT', 'DOT/USDT:USDT'], state, {}, '/tmp/does_not_matter.json', 1.0, 1.5, 3)
    assert result == state


def test_reconcile_clears_state_when_position_closed_externally(tmp_path, monkeypatch):
    ex = FakeExchange({})  # keine offene Position mehr
    state = {'active_symbol': 'NEAR/USDT:USDT', 'active_direction': 'long'}
    am_path = str(tmp_path / 'am_state.json')

    calls = []
    monkeypatch.setattr(run_renko_breakout, 'resolve_am_outcome',
                         lambda *a, **k: calls.append(a) or {})

    result = run_renko_breakout.reconcile_portfolio_state(
        ex, ['NEAR/USDT:USDT', 'DOT/USDT:USDT'], state, {}, am_path, 1.0, 1.5, 3)
    assert result == {'active_symbol': None, 'active_direction': None}
    assert len(calls) == 1  # Anti-Martingale-Ausgang wurde aufgeloest


def test_reconcile_adopts_untracked_exchange_position():
    ex = FakeExchange({'SOL/USDT:USDT': {'side': 'short'}})
    state = {'active_symbol': None, 'active_direction': None}
    result = run_renko_breakout.reconcile_portfolio_state(
        ex, ['NEAR/USDT:USDT', 'SOL/USDT:USDT'], state, {}, '/tmp/x.json', 1.0, 1.5, 3)
    assert result == {'active_symbol': 'SOL/USDT:USDT', 'active_direction': 'short'}


def test_fetch_new_closed_candles_excludes_still_forming_candle(monkeypatch):
    idx = pd.date_range('2026-09-21 10:00', periods=5, freq='5min', tz='UTC')
    fake_df = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0}, index=idx)
    monkeypatch.setattr(run_renko_breakout, 'fetch_ohlcv_incremental', lambda *a, **k: fake_df)

    now_utc = pd.Timestamp('2026-09-21 10:24', tz='UTC')  # letzte Kerze (10:20) laeuft noch
    result = run_renko_breakout.fetch_new_closed_candles(
        'NEAR/USDT:USDT', '5m', 100, '/tmp', last_processed_ts=None, now_utc=now_utc)
    assert result.index.max() == pd.Timestamp('2026-09-21 10:15', tz='UTC')


def test_fetch_new_closed_candles_excludes_already_processed():
    idx = pd.date_range('2026-09-21 10:00', periods=5, freq='5min', tz='UTC')
    fake_df = pd.DataFrame({'open': 1.0, 'high': 1.0, 'low': 1.0, 'close': 1.0}, index=idx)
    run_renko_breakout.fetch_ohlcv_incremental = lambda *a, **k: fake_df

    now_utc = pd.Timestamp('2026-09-21 10:30', tz='UTC')
    result = run_renko_breakout.fetch_new_closed_candles(
        'NEAR/USDT:USDT', '5m', 100, '/tmp',
        last_processed_ts=pd.Timestamp('2026-09-21 10:05', tz='UTC'), now_utc=now_utc)
    assert list(result.index) == [pd.Timestamp('2026-09-21 10:10', tz='UTC'),
                                   pd.Timestamp('2026-09-21 10:15', tz='UTC'),
                                   pd.Timestamp('2026-09-21 10:20', tz='UTC')]
