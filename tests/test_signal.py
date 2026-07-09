from oraclebot.strategy.signal import compute_position_size, compute_trade_signal


def make_prediction(trend, range_cat=1, trend_confidence=0.6):
    return {
        'trend': trend,
        'range': range_cat,
        'step_probabilities': {'trend': trend_confidence},
    }


def test_neutral_trend_produces_no_trade():
    signal = compute_trade_signal(make_prediction(trend=1), prev_close=100.0, atr=2.0)
    assert signal['direction'] is None
    assert signal['reason'] == 'neutral_trend'


def test_low_confidence_produces_no_trade():
    signal = compute_trade_signal(make_prediction(trend=2, trend_confidence=0.3), prev_close=100.0, atr=2.0,
                                   min_trend_confidence=0.4)
    assert signal['direction'] is None
    assert signal['reason'] == 'low_confidence'


def test_bullish_trend_produces_long_signal_with_correct_ordering():
    signal = compute_trade_signal(make_prediction(trend=2), prev_close=100.0, atr=2.0)
    assert signal['direction'] == 'long'
    assert signal['stop_loss'] < signal['entry'] < signal['take_profit']


def test_bearish_trend_produces_short_signal_with_correct_ordering():
    signal = compute_trade_signal(make_prediction(trend=0), prev_close=100.0, atr=2.0)
    assert signal['direction'] == 'short'
    assert signal['take_profit'] < signal['entry'] < signal['stop_loss']


def test_risk_reward_ratio_is_respected():
    signal = compute_trade_signal(make_prediction(trend=2), prev_close=100.0, atr=2.0, risk_reward=3.0)
    assert abs(signal['tp_distance'] - 3.0 * signal['sl_distance']) < 1e-9


def test_sl_distance_scales_with_predicted_range():
    small_range = compute_trade_signal(make_prediction(trend=2, range_cat=0), prev_close=100.0, atr=2.0)
    large_range = compute_trade_signal(make_prediction(trend=2, range_cat=3), prev_close=100.0, atr=2.0)
    assert small_range['sl_distance'] < large_range['sl_distance']


def test_position_size_scales_inversely_with_sl_distance():
    tight_size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=99.0)
    wide_size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=95.0)
    assert tight_size > wide_size
    # Risiko = 1% von 1000 = 10 USDT; SL-Abstand 1 -> Groesse 10
    assert abs(tight_size - 10.0) < 1e-9


def test_position_size_zero_when_sl_equals_entry():
    size = compute_position_size(balance=1000.0, risk_per_trade_pct=1.0, entry=100.0, stop_loss=100.0)
    assert size == 0.0
