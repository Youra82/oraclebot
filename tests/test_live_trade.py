import json
import os
from unittest.mock import MagicMock, patch

from oraclebot.strategy.live_trade import execute_live_trade

SYMBOL = 'BTC/USDT:USDT'
STRAT_CFG = {'leverage': 5, 'margin_mode': 'isolated', 'risk_per_trade_pct': 1.0}
TELEGRAM_CFG = {'bot_token': 'TOKEN', 'chat_id': 'CHAT'}

LONG_SIGNAL = {
    'direction': 'long', 'entry': 60000.0, 'stop_loss': 59000.0, 'take_profit': 62000.0,
    'sl_distance': 1000.0, 'tp_distance': 2000.0, 'confidence': 0.6,
}
NO_TRADE_SIGNAL = {'direction': None, 'reason': 'low_confidence', 'confidence': 0.3}


def make_exchange(balance=1000.0, open_positions=None, min_amount=0.0001):
    exchange = MagicMock()
    exchange.fetch_open_positions.return_value = open_positions or []
    exchange.fetch_balance_usdt.return_value = balance
    exchange.fetch_min_amount_tradable.return_value = min_amount
    exchange.place_market_order.return_value = {'average': 60000.0, 'filled': 0.01}
    exchange.place_trigger_market_order.return_value = {'id': 'trigger1'}
    return exchange


@patch('oraclebot.strategy.live_trade.send_message')
def test_skips_when_position_already_open(mock_send):
    exchange = make_exchange(open_positions=[{'side': 'long', 'entryPrice': 60000, 'unrealizedPnl': 5.0}])
    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)
    assert result == {'action': 'skipped', 'reason': 'position_open'}
    exchange.place_market_order.assert_not_called()
    exchange.cancel_all_orders_for_symbol.assert_not_called()


@patch('oraclebot.strategy.live_trade.send_message')
def test_cancels_leftover_trigger_orders_when_no_position_open(mock_send):
    # Kein offene Position -> ein evtl. nicht ausgeloester SL/TP-Trigger der letzten,
    # bereits geschlossenen Position muss aufgeraeumt werden, bevor irgendetwas anderes passiert.
    exchange = make_exchange(open_positions=[])
    execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)
    exchange.cancel_all_orders_for_symbol.assert_called_once_with(SYMBOL)


@patch('oraclebot.strategy.live_trade.send_message')
def test_skips_when_no_signal(mock_send):
    exchange = make_exchange()
    result = execute_live_trade(exchange, NO_TRADE_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)
    assert result['action'] == 'skipped'
    assert result['reason'] == 'low_confidence'
    exchange.place_market_order.assert_not_called()


@patch('oraclebot.strategy.live_trade.send_message')
def test_skips_when_balance_too_low(mock_send):
    exchange = make_exchange(balance=1.0)
    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)
    assert result == {'action': 'skipped', 'reason': 'insufficient_balance'}
    exchange.place_market_order.assert_not_called()


@patch('oraclebot.strategy.live_trade.send_message')
def test_skips_when_below_exchange_min_amount(mock_send):
    exchange = make_exchange(balance=1000.0, min_amount=10.0)  # absurd high min to force skip
    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)
    assert result == {'action': 'skipped', 'reason': 'below_min_amount'}
    exchange.place_market_order.assert_not_called()


@patch('oraclebot.strategy.live_trade.send_message')
def test_successful_long_entry_places_entry_then_sl_then_tp(mock_send):
    exchange = make_exchange()
    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    assert result['action'] == 'entered'
    exchange.set_margin_mode.assert_called_once_with(SYMBOL, 'isolated')
    exchange.set_leverage.assert_called_once_with(SYMBOL, 5, 'isolated')

    # Entry: buy (long)
    entry_args = exchange.place_market_order.call_args
    assert entry_args[0][0] == SYMBOL
    assert entry_args[0][1] == 'buy'

    # SL + TP: both sell (closing a long), reduceOnly, SL placed before TP
    assert exchange.place_trigger_market_order.call_count == 2
    sl_call, tp_call = exchange.place_trigger_market_order.call_args_list
    assert sl_call[0][1] == 'sell' and sl_call[1]['reduce'] is True
    assert tp_call[0][1] == 'sell' and tp_call[1]['reduce'] is True

    # Fill price was 60000 (from place_market_order mock) -> SL/TP anchored to it
    sl_price = sl_call[0][3]
    tp_price = tp_call[0][3]
    assert sl_price == 60000.0 - LONG_SIGNAL['sl_distance']
    assert tp_price == 60000.0 + LONG_SIGNAL['tp_distance']
    mock_send.assert_called_once()


@patch('oraclebot.strategy.live_trade.send_message')
def test_short_entry_uses_buy_side_for_exit_orders(mock_send):
    short_signal = dict(LONG_SIGNAL, direction='short', entry=60000.0, stop_loss=61000.0)
    exchange = make_exchange()
    exchange.place_market_order.return_value = {'average': 60000.0, 'filled': 0.01}

    result = execute_live_trade(exchange, short_signal, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    assert result['action'] == 'entered'
    entry_args = exchange.place_market_order.call_args
    assert entry_args[0][1] == 'sell'  # short entry = sell
    sl_call, tp_call = exchange.place_trigger_market_order.call_args_list
    assert sl_call[0][1] == 'buy'  # closing a short = buy
    assert tp_call[0][1] == 'buy'


@patch('oraclebot.strategy.live_trade.send_message')
def test_margin_cap_reduces_oversized_position(mock_send):
    # Tiny SL distance -> risk-based sizing would demand far more than the account can margin.
    tight_sl_signal = dict(LONG_SIGNAL, stop_loss=59999.0, sl_distance=1.0)
    exchange = make_exchange(balance=100.0)  # small balance, 5x leverage -> max ~0.0083 BTC notional cap
    exchange.place_market_order.return_value = {'average': 60000.0, 'filled': 0.008}

    execute_live_trade(exchange, tight_sl_signal, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    placed_amount = exchange.place_market_order.call_args[0][2]
    max_by_margin = (100.0 * 5) / 60000.0 * 0.99
    assert placed_amount <= max_by_margin + 1e-9


@patch('oraclebot.strategy.live_trade.send_message')
def test_sl_failure_closes_position_and_alerts_no_tp_placed(mock_send):
    exchange = make_exchange()
    exchange.place_trigger_market_order.side_effect = Exception("SL rejected by exchange")

    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    assert result == {'action': 'failed', 'reason': 'sl_placement_failed'}
    exchange.close_position.assert_called_once_with(SYMBOL)
    mock_send.assert_called_once()
    assert 'ACHTUNG' in mock_send.call_args[0][2]


@patch('oraclebot.strategy.live_trade.send_message')
def test_entry_order_failure_returns_failed_without_placing_exits(mock_send):
    exchange = make_exchange()
    exchange.place_market_order.side_effect = Exception("insufficient funds")

    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    assert result == {'action': 'failed', 'reason': 'entry_order_failed'}
    exchange.place_trigger_market_order.assert_not_called()


@patch('oraclebot.strategy.live_trade.send_message')
def test_tp_failure_still_counts_as_entered_since_sl_protects_position(mock_send):
    exchange = make_exchange()
    # First call (SL) succeeds, second call (TP) fails.
    exchange.place_trigger_market_order.side_effect = [{'id': 'sl1'}, Exception("TP rejected")]

    result = execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, STRAT_CFG, TELEGRAM_CFG)

    assert result['action'] == 'entered'
    exchange.close_position.assert_not_called()


# --- Anti-Martingale-Integration ---

ANTI_MARTINGALE_CFG = dict(STRAT_CFG, anti_martingale_enabled=True, anti_martingale_base_pct=10.0,
                            anti_martingale_growth_factor=2.0, anti_martingale_streak_target=3)


@patch('oraclebot.strategy.live_trade.send_message')
def test_anti_martingale_sizes_position_from_stake_pct_not_risk(mock_send, tmp_path):
    state_path = os.path.join(tmp_path, 'state.json')
    exchange = make_exchange(balance=1000.0)  # 10% Einsatz -> 100 USDT Margin * 5x Hebel / 60000 Entry

    execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, ANTI_MARTINGALE_CFG, TELEGRAM_CFG, state_path=state_path)

    expected_contracts = (1000.0 * 0.10 * 5) / 60000.0
    placed_amount = exchange.place_market_order.call_args[0][2]
    assert abs(placed_amount - expected_contracts) < 1e-9


@patch('oraclebot.strategy.live_trade.send_message')
def test_anti_martingale_records_pending_position_after_entry(mock_send, tmp_path):
    state_path = os.path.join(tmp_path, 'state.json')
    exchange = make_exchange(balance=1000.0)

    execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, ANTI_MARTINGALE_CFG, TELEGRAM_CFG, state_path=state_path)

    with open(state_path) as f:
        state = json.load(f)
    assert state['pending_position'] is not None
    assert state['pending_position']['balance_before'] == 1000.0


@patch('oraclebot.strategy.live_trade.send_message')
def test_anti_martingale_resolves_win_and_doubles_stake_before_next_entry(mock_send, tmp_path):
    state_path = os.path.join(tmp_path, 'state.json')
    with open(state_path, 'w') as f:
        json.dump({'stake_pct': 10.0, 'consecutive_wins': 0,
                    'pending_position': {'balance_before': 1000.0, 'expected_win_balance': 1100.0,
                                          'expected_loss_balance': 900.0}}, f)

    # Kein offener Position mehr gefunden -> die vorherige ist geschlossen; Guthaben liegt naeher
    # an der Gewinn- als an der Verlust-Erwartung -> als Gewinn gewertet, Einsatz verdoppelt.
    exchange = make_exchange(balance=1100.0, open_positions=[])

    execute_live_trade(exchange, LONG_SIGNAL, SYMBOL, ANTI_MARTINGALE_CFG, TELEGRAM_CFG, state_path=state_path)

    expected_contracts = (1100.0 * 0.20 * 5) / 60000.0  # 20% = verdoppelter Einsatz nach dem Gewinn
    placed_amount = exchange.place_market_order.call_args[0][2]
    assert abs(placed_amount - expected_contracts) < 1e-9
