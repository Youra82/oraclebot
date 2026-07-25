from oraclebot.strategy.barrier_signal import compute_barrier_signal


def test_low_confidence_produces_no_trade():
    signal = compute_barrier_signal(predicted_class=1, confidence=0.4, entry_price=100.0, min_confidence=0.6)
    assert signal['direction'] is None
    assert signal['reason'] == 'low_confidence'


def test_up_first_produces_long_signal_with_correct_ordering():
    signal = compute_barrier_signal(predicted_class=1, confidence=0.7, entry_price=100.0,
                                     min_confidence=0.6, barrier_pct=1.0)
    assert signal['direction'] == 'long'
    assert signal['stop_loss'] < signal['entry'] < signal['take_profit']
    assert signal['stop_loss'] == 99.0
    assert signal['take_profit'] == 101.0


def test_down_first_produces_short_signal_with_correct_ordering():
    signal = compute_barrier_signal(predicted_class=0, confidence=0.7, entry_price=100.0,
                                     min_confidence=0.6, barrier_pct=1.0)
    assert signal['direction'] == 'short'
    assert signal['take_profit'] < signal['entry'] < signal['stop_loss']
    assert signal['stop_loss'] == 101.0
    assert signal['take_profit'] == 99.0


def test_sl_distance_equals_tp_distance_symmetric_barrier():
    signal = compute_barrier_signal(predicted_class=1, confidence=0.9, entry_price=200.0,
                                     min_confidence=0.6, barrier_pct=2.0)
    assert signal['sl_distance'] == signal['tp_distance'] == 4.0


def test_confidence_exactly_at_threshold_still_trades():
    signal = compute_barrier_signal(predicted_class=1, confidence=0.6, entry_price=100.0, min_confidence=0.6)
    assert signal['direction'] == 'long'
