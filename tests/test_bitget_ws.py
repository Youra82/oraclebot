import json

from oraclebot.utils.bitget_ws import (from_bitget_inst_id, parse_trade_message, to_bitget_inst_id)


def test_to_bitget_inst_id():
    assert to_bitget_inst_id("NEAR/USDT:USDT") == "NEARUSDT"
    assert to_bitget_inst_id("BTC/USDT:USDT") == "BTCUSDT"


def test_from_bitget_inst_id():
    assert from_bitget_inst_id("NEARUSDT") == "NEAR/USDT:USDT"
    assert from_bitget_inst_id("XRPUSDT") == "XRP/USDT:USDT"


def test_roundtrip():
    for symbol in ["NEAR/USDT:USDT", "DOT/USDT:USDT", "SOL/USDT:USDT", "ADA/USDT:USDT",
                    "AVAX/USDT:USDT", "SUI/USDT:USDT", "XRP/USDT:USDT"]:
        assert from_bitget_inst_id(to_bitget_inst_id(symbol)) == symbol


def test_parse_real_trade_push_message():
    # Format aus der offiziellen Bitget-API-Doku (USDT-FUTURES trade channel).
    raw = json.dumps({
        "action": "snapshot",
        "arg": {"instType": "USDT-FUTURES", "channel": "trade", "instId": "NEARUSDT"},
        "data": [{"ts": "1695716760565", "price": "4.3921", "size": "12.5", "side": "buy",
                   "tradeId": "1111111111"}],
        "ts": 1695716761589,
    })
    result = parse_trade_message(raw)
    assert result == [("NEAR/USDT:USDT", 1695716760565, 4.3921)]


def test_parse_message_with_multiple_trades_in_one_push():
    raw = json.dumps({
        "action": "update",
        "arg": {"instType": "USDT-FUTURES", "channel": "trade", "instId": "SUIUSDT"},
        "data": [
            {"ts": "1000", "price": "1.01", "size": "5", "side": "buy", "tradeId": "1"},
            {"ts": "1001", "price": "1.02", "size": "3", "side": "sell", "tradeId": "2"},
        ],
    })
    result = parse_trade_message(raw)
    assert result == [("SUI/USDT:USDT", 1000, 1.01), ("SUI/USDT:USDT", 1001, 1.02)]


def test_parse_pong_returns_empty():
    assert parse_trade_message("pong") == []


def test_parse_subscribe_ack_returns_empty():
    raw = json.dumps({"event": "subscribe", "arg": {"instType": "USDT-FUTURES", "channel": "trade",
                                                       "instId": "NEARUSDT"}})
    assert parse_trade_message(raw) == []


def test_parse_error_event_returns_empty_not_raises():
    raw = json.dumps({"event": "error", "code": 30001, "msg": "invalid op"})
    assert parse_trade_message(raw) == []


def test_parse_non_trade_channel_ignored():
    raw = json.dumps({
        "action": "snapshot",
        "arg": {"instType": "USDT-FUTURES", "channel": "ticker", "instId": "NEARUSDT"},
        "data": [{"lastPr": "4.39"}],
    })
    assert parse_trade_message(raw) == []


def test_parse_malformed_json_does_not_raise():
    assert parse_trade_message("{not valid json") == []


def test_parse_unknown_inst_id_ignored_not_raises():
    raw = json.dumps({
        "arg": {"instType": "USDT-FUTURES", "channel": "trade", "instId": "WEIRDCOINWITHOUTUSDT"},
        "data": [{"ts": "1000", "price": "1.0", "size": "1", "side": "buy", "tradeId": "1"}],
    })
    # WEIRDCOINWITHOUTUSDT -> from_bitget_inst_id strips last 4 chars -> 'WEIRDCOINWITHOUT' + '/USDT:USDT'
    # (kein Crash, auch wenn das Symbol nicht zu den abonnierten 7 gehoert)
    result = parse_trade_message(raw)
    assert len(result) == 1


def test_parse_entry_with_missing_price_field_is_skipped_not_raises():
    raw = json.dumps({
        "arg": {"instType": "USDT-FUTURES", "channel": "trade", "instId": "NEARUSDT"},
        "data": [{"ts": "1000", "size": "1", "side": "buy", "tradeId": "1"}],  # kein 'price'
    })
    assert parse_trade_message(raw) == []
