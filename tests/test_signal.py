from oraclebot.strategy.signal import compute_position_size


def test_position_size_scales_inversely_with_sl_distance():
    tight_size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=99.0)
    wide_size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=95.0)
    assert tight_size > wide_size
    # Risiko = 1% von 1000 = 10 USDT; SL-Abstand 1 -> Groesse 10
    assert abs(tight_size - 10.0) < 1e-9


def test_position_size_zero_when_sl_equals_entry():
    size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=100.0)
    assert size == 0.0
