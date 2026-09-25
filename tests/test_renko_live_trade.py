import os

from oraclebot.strategy import renko_live_trade
from oraclebot.strategy import anti_martingale


class FakeExchange:
    """Minimaler Fake fuer die in close_renko_position()/resolve_am_outcome() genutzten
    Exchange-Methoden -- kein echter API-Zugriff, damit diese Tests ohne Bitget-Verbindung laufen."""

    def __init__(self, open_positions=None, close_order=None, closed_history=None, close_raises=None):
        self._open = open_positions or []
        self._close_order = close_order
        self._closed_history = closed_history or []
        self._close_raises = close_raises
        self.cancel_calls = []

    def fetch_open_positions(self, symbol):
        return self._open

    def close_position(self, symbol):
        if self._close_raises:
            raise self._close_raises
        return self._close_order

    def cancel_all_orders_for_symbol(self, symbol):
        self.cancel_calls.append(symbol)

    def fetch_closed_positions(self, symbol, limit=1):
        return self._closed_history


def _am_state(path, stake_pct=2.0, consecutive_wins=0):
    anti_martingale.save_state(path, {'stake_pct': stake_pct, 'consecutive_wins': consecutive_wins})
    return path


def test_close_long_position_wins_when_fill_price_above_entry(tmp_path):
    am_path = _am_state(str(tmp_path / 'am.json'))
    ex = FakeExchange(open_positions=[{'side': 'long'}], close_order={'average': 105.0})

    result = renko_live_trade.close_renko_position(
        ex, 'NEAR/USDT:USDT', 'long', entry_price=100.0, exit_reason='Gegen-Brick',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['action'] == 'closed'
    assert result['is_win'] is True
    state = anti_martingale.load_state(am_path, base_pct=2.0)
    assert state['consecutive_wins'] == 1
    assert state['stake_pct'] == 4.0


def test_close_long_position_loses_when_fill_price_below_entry(tmp_path):
    am_path = _am_state(str(tmp_path / 'am.json'), stake_pct=8.0, consecutive_wins=1)
    ex = FakeExchange(open_positions=[{'side': 'long'}], close_order={'average': 95.0})

    result = renko_live_trade.close_renko_position(
        ex, 'NEAR/USDT:USDT', 'long', entry_price=100.0, exit_reason='Gegen-Brick',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['is_win'] is False
    state = anti_martingale.load_state(am_path, base_pct=2.0)
    assert state['consecutive_wins'] == 0
    assert state['stake_pct'] == 2.0


def test_close_short_position_wins_when_fill_price_below_entry(tmp_path):
    # Short: Gewinn wenn der Preis FAELLT -- Vorzeichen muss umgedreht sein gegenueber long.
    am_path = _am_state(str(tmp_path / 'am.json'))
    ex = FakeExchange(open_positions=[{'side': 'short'}], close_order={'average': 90.0})

    result = renko_live_trade.close_renko_position(
        ex, 'AVAX/USDT:USDT', 'short', entry_price=100.0, exit_reason='Gegen-Brick',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['is_win'] is True


def test_close_falls_back_to_history_when_fill_price_missing(tmp_path):
    # close_order liefert keinen brauchbaren Preis (average/price beide leer) -- muss auf die
    # echte Positions-Historie zurueckfallen statt faelschlich is_win=False anzunehmen.
    am_path = _am_state(str(tmp_path / 'am.json'))
    ex = FakeExchange(open_positions=[{'side': 'long'}], close_order={},
                       closed_history=[{'realizedPnl': 0.42}])

    result = renko_live_trade.close_renko_position(
        ex, 'NEAR/USDT:USDT', 'long', entry_price=100.0, exit_reason='Gegen-Brick',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['is_win'] is True


def test_close_already_closed_externally_uses_history(tmp_path):
    # Position ist auf der Boerse bereits weg (z.B. Sicherheits-Stop ausgeloest) -- kein eigener
    # Fuellpreis verfuegbar, MUSS ueber die Positions-Historie aufloesen.
    am_path = _am_state(str(tmp_path / 'am.json'))
    ex = FakeExchange(open_positions=[], closed_history=[{'realizedPnl': -1.5}])

    result = renko_live_trade.close_renko_position(
        ex, 'NEAR/USDT:USDT', 'long', entry_price=100.0, exit_reason='Sicherheits-Stop',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['action'] == 'already_closed'
    assert result['is_win'] is False
    assert ex.cancel_calls == ['NEAR/USDT:USDT']


def test_close_with_no_history_available_is_conservatively_a_loss(tmp_path):
    am_path = _am_state(str(tmp_path / 'am.json'), stake_pct=16.0, consecutive_wins=2)
    ex = FakeExchange(open_positions=[], closed_history=[])

    result = renko_live_trade.close_renko_position(
        ex, 'NEAR/USDT:USDT', 'long', entry_price=100.0, exit_reason='unklar',
        telegram_cfg={}, am_state_path=am_path, base_pct=2.0, growth_factor=2.0, streak_target=3)

    assert result['is_win'] is False
    state = anti_martingale.load_state(am_path, base_pct=2.0)
    assert state['stake_pct'] == 2.0  # zurueck auf Basis, nicht faelschlich weiter compoundiert


def test_resolve_am_outcome_updates_shared_state_from_history(tmp_path):
    am_path = _am_state(str(tmp_path / 'am.json'))
    ex = FakeExchange(closed_history=[{'realizedPnl': 3.0}])

    state = renko_live_trade.resolve_am_outcome(ex, 'DOT/USDT:USDT', am_path, base_pct=2.0,
                                                 growth_factor=2.0, streak_target=3)

    assert state['consecutive_wins'] == 1
    assert state['stake_pct'] == 4.0
