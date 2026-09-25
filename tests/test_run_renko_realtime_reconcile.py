import importlib.util
import os

SCRIPT_PATH = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'run_renko_realtime.py')
spec = importlib.util.spec_from_file_location('run_renko_realtime', SCRIPT_PATH)
run_renko_realtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_renko_realtime)


class FakeExchange:
    """Simuliert fetch_open_positions() je Symbol -- reconcile_positions_state() braucht nichts
    anderes. Kein echter API-Zugriff, damit dieser Test ohne Bitget-Verbindung laeuft."""

    def __init__(self, open_positions: dict):
        self._open = open_positions  # {symbol: {'side': ..., 'entryPrice': ..., 'contracts': ...}}

    def fetch_open_positions(self, symbol):
        pos = self._open.get(symbol)
        return [pos] if pos else []


SYMBOLS = ['NEAR/USDT:USDT', 'DOT/USDT:USDT', 'SOL/USDT:USDT']


def test_reconcile_keeps_matching_positions_untouched():
    ex = FakeExchange({'NEAR/USDT:USDT': {'side': 'long', 'entryPrice': 4.5, 'contracts': 2.0}})
    state = {'NEAR/USDT:USDT': {'direction': 'long', 'entry_price': 4.5, 'contracts': 2.0},
             'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}
    result = run_renko_realtime.reconcile_positions_state(
        ex, SYMBOLS, state, {}, '/tmp/does_not_matter.json', 2.0, 1.5, 3)
    assert result == state


def test_reconcile_clears_position_closed_externally(tmp_path, monkeypatch):
    ex = FakeExchange({})  # keine offenen Positionen mehr auf der Boerse
    state = {'NEAR/USDT:USDT': {'direction': 'long', 'entry_price': 4.5, 'contracts': 2.0},
             'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}
    am_path = str(tmp_path / 'am_state.json')

    calls = []
    monkeypatch.setattr(run_renko_realtime, 'resolve_am_outcome',
                         lambda *a, **k: calls.append(a) or {})

    result = run_renko_realtime.reconcile_positions_state(
        ex, SYMBOLS, state, {}, am_path, 2.0, 1.5, 3)

    assert result == {'NEAR/USDT:USDT': None, 'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}
    assert len(calls) == 1
    assert calls[0][1] == 'NEAR/USDT:USDT'  # resolve_am_outcome(exchange, symbol, ...)


def test_reconcile_adopts_untracked_exchange_position():
    ex = FakeExchange({'SOL/USDT:USDT': {'side': 'short', 'entryPrice': 115.0, 'contracts': 0.1}})
    state = {'NEAR/USDT:USDT': None, 'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}
    result = run_renko_realtime.reconcile_positions_state(
        ex, SYMBOLS, state, {}, '/tmp/x.json', 2.0, 1.5, 3)
    assert result['SOL/USDT:USDT'] == {'direction': 'short', 'entry_price': 115.0, 'contracts': 0.1}
    assert result['NEAR/USDT:USDT'] is None


def test_reconcile_handles_multiple_simultaneously_open_positions_without_alarm():
    # Der Kernpunkt des Mehrfach-Positionen-Umbaus: mehrere gleichzeitig offene Positionen sind
    # jetzt der Normalfall, keine "MEHR ALS EINE offene Position"-Fehlermeldung mehr.
    ex = FakeExchange({
        'NEAR/USDT:USDT': {'side': 'long', 'entryPrice': 4.5, 'contracts': 2.0},
        'DOT/USDT:USDT': {'side': 'short', 'entryPrice': 1.1, 'contracts': 10.0},
    })
    state = {'NEAR/USDT:USDT': {'direction': 'long', 'entry_price': 4.5, 'contracts': 2.0},
             'DOT/USDT:USDT': {'direction': 'short', 'entry_price': 1.1, 'contracts': 10.0},
             'SOL/USDT:USDT': None}
    result = run_renko_realtime.reconcile_positions_state(
        ex, SYMBOLS, state, {}, '/tmp/x.json', 2.0, 1.5, 3)
    assert result == state


def test_load_positions_state_defaults_all_symbols_to_none_when_no_file(tmp_path):
    path = str(tmp_path / 'positions.json')
    state = run_renko_realtime.load_positions_state(path, SYMBOLS)
    assert state == {'NEAR/USDT:USDT': None, 'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}


def test_save_then_load_positions_state_roundtrip(tmp_path):
    path = str(tmp_path / 'positions.json')
    state = {'NEAR/USDT:USDT': {'direction': 'long', 'entry_price': 4.5, 'contracts': 2.0},
             'DOT/USDT:USDT': None, 'SOL/USDT:USDT': None}
    run_renko_realtime.save_positions_state(path, state)
    loaded = run_renko_realtime.load_positions_state(path, SYMBOLS)
    assert loaded == state
