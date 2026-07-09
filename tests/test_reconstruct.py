from oraclebot.model.reconstruct import reconstruct_candle


def test_bullish_candle_has_close_above_open():
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=2, range_cat=1,
                                 close_position_cat=2, upper_wick_cat=0, lower_wick_cat=0)
    assert result['close'] > result['open']
    assert result['body_top'] == result['close']
    assert result['body_bottom'] == result['open']


def test_bearish_candle_has_close_below_open():
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=0, range_cat=1,
                                 close_position_cat=0, upper_wick_cat=0, lower_wick_cat=0)
    assert result['close'] < result['open']
    assert result['body_top'] == result['open']
    assert result['body_bottom'] == result['close']


def test_neutral_candle_has_close_equal_open():
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=1, range_cat=1,
                                 close_position_cat=1, upper_wick_cat=1, lower_wick_cat=1)
    assert result['close'] == result['open']


def test_coordinates_are_ordered_correctly():
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=2, range_cat=2,
                                 close_position_cat=2, upper_wick_cat=1, lower_wick_cat=2)
    assert result['low'] <= result['body_bottom'] <= result['body_top'] <= result['high']


def test_wick_sizes_are_nonnegative():
    result = reconstruct_candle(prev_close=50000.0, atr=800.0, trend=0, range_cat=3,
                                 close_position_cat=0, upper_wick_cat=2, lower_wick_cat=0)
    assert result['upper_wick_size'] >= 0
    assert result['lower_wick_size'] >= 0


def test_zero_wicks_and_full_body_reproduces_open_and_close_as_high_low():
    # upper_wick=0 (klein, aber nicht exakt 0 -- Bucket-Reprae­sentant ist 0.075) und
    # trend bullisch: High sollte nur knapp ueber dem Close liegen.
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=2, range_cat=0,
                                 close_position_cat=2, upper_wick_cat=0, lower_wick_cat=0)
    assert result['high'] > result['close']
    assert result['low'] < result['open']


def test_close_position_consistency_flag_matches_geometry():
    # Bullische Kerze mit sehr kleinen Wicks -> Close nahe High -> upper_third erwartet konsistent
    result = reconstruct_candle(prev_close=100.0, atr=2.0, trend=2, range_cat=1,
                                 close_position_cat=2, upper_wick_cat=0, lower_wick_cat=0)
    assert result['close_position_consistent'] is True

    # Dieselbe Kerzenform, aber close_position-Vorhersage widerspricht der Geometrie
    inconsistent = reconstruct_candle(prev_close=100.0, atr=2.0, trend=2, range_cat=1,
                                       close_position_cat=0, upper_wick_cat=0, lower_wick_cat=0)
    assert inconsistent['close_position_consistent'] is False
