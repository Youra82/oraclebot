import numpy as np
import pandas as pd
import torch

from oraclebot.data.targets import TARGET_NAMES
from oraclebot.model.transformer import TARGET_CARDINALITIES, MarketTransformer
from oraclebot.strategy.backtest import determine_trade_outcome, predict_for_examples, run_signal_backtest

TIMEFRAMES = ['1d', '4h']
WINDOW_SIZES = {'1d': 5, '4h': 10}
N_FEATURES = 19


class DummyScaler:
    def transform_array(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)


def make_signal(direction, entry=100.0, sl_distance=2.0, risk_reward=2.0):
    if direction == 'long':
        stop_loss, take_profit = entry - sl_distance, entry + sl_distance * risk_reward
    else:
        stop_loss, take_profit = entry + sl_distance, entry - sl_distance * risk_reward
    return {'direction': direction, 'entry': entry, 'stop_loss': stop_loss, 'take_profit': take_profit,
            'sl_distance': sl_distance, 'tp_distance': sl_distance * risk_reward, 'confidence': 0.5}


def make_day_bars(prices: list, start='2024-01-01') -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(prices), freq='4h', tz='UTC')
    rows = [{'open': p, 'high': p, 'low': p, 'close': p} for p in prices]
    return pd.DataFrame(rows, index=idx)


# --- determine_trade_outcome ---

def test_long_hits_take_profit():
    signal = make_signal('long')  # entry=100, sl=98, tp=104
    day_bars = make_day_bars([100, 101, 105, 103])
    outcome = determine_trade_outcome(signal, day_bars)
    assert outcome['outcome'] == 'win'
    assert outcome['exit_price'] == signal['take_profit']
    assert outcome['r_multiple'] > 0


def test_long_hits_stop_loss():
    signal = make_signal('long')  # sl=98
    day_bars = make_day_bars([100, 99, 97, 103])
    outcome = determine_trade_outcome(signal, day_bars)
    assert outcome['outcome'] == 'loss'
    assert outcome['exit_price'] == signal['stop_loss']
    assert outcome['r_multiple'] < 0


def test_short_hits_take_profit():
    signal = make_signal('short')  # entry=100, sl=102, tp=96
    day_bars = make_day_bars([100, 99, 95, 97])
    outcome = determine_trade_outcome(signal, day_bars)
    assert outcome['outcome'] == 'win'
    assert outcome['exit_price'] == signal['take_profit']


def test_neither_sl_nor_tp_hit_times_out_at_close():
    signal = make_signal('long')  # sl=98, tp=104
    day_bars = make_day_bars([100, 100.5, 99.5, 101.0])
    outcome = determine_trade_outcome(signal, day_bars)
    assert outcome['outcome'] == 'timeout'
    assert outcome['exit_price'] == 101.0


def test_sl_takes_priority_when_both_hit_in_same_bar():
    signal = make_signal('long')  # sl=98, tp=104
    idx = pd.date_range('2024-01-01', periods=1, freq='4h', tz='UTC')
    day_bars = pd.DataFrame([{'open': 100, 'high': 106, 'low': 96, 'close': 100}], index=idx)
    outcome = determine_trade_outcome(signal, day_bars)
    assert outcome['outcome'] == 'loss'


def test_empty_day_bars_exits_at_entry():
    signal = make_signal('long')
    outcome = determine_trade_outcome(signal, pd.DataFrame(columns=['open', 'high', 'low', 'close']))
    assert outcome['outcome'] == 'timeout'
    assert outcome['exit_price'] == signal['entry']


# --- run_signal_backtest (Integrationstest mit kleinem Modell) ---

def make_model():
    return MarketTransformer(
        n_features=N_FEATURES, timeframes=TIMEFRAMES, window_sizes=WINDOW_SIZES,
        d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32, dropout=0.0,
    )


def make_ohlcv_by_symbol(symbol='BTC/USDT:USDT', n_days=10):
    daily_idx = pd.date_range('2024-01-01', periods=n_days, freq='D', tz='UTC')
    daily = pd.DataFrame({'open': 100.0, 'high': 105.0, 'low': 95.0, 'close': 100.0,
                           'volume': 500.0}, index=daily_idx)
    intraday_idx = pd.date_range('2024-01-01', periods=n_days * 6, freq='4h', tz='UTC')
    intraday = pd.DataFrame({'open': 100.0, 'high': 101.0, 'low': 99.0, 'close': 100.0,
                              'volume': 100.0}, index=intraday_idx)
    return {symbol: {'1d': daily, '4h': intraday}}


def make_examples(symbol='BTC/USDT:USDT', n=4, seed=0):
    rng = np.random.default_rng(seed)
    daily_idx = pd.date_range('2024-01-01', periods=10, freq='D', tz='UTC')
    examples = []
    for i in range(n):
        ex = {tf: rng.normal(size=(WINDOW_SIZES[tf], N_FEATURES)).tolist() for tf in TIMEFRAMES}
        ex['target'] = {name: int(rng.integers(0, TARGET_CARDINALITIES[name])) for name in TARGET_NAMES}
        ex['symbol'] = symbol
        ex['reference_time'] = daily_idx[i].isoformat()
        ex['date'] = daily_idx[i + 1].isoformat()
        examples.append(ex)
    return examples


def test_predict_for_examples_returns_coords_and_prediction():
    model = make_model()
    scaler = DummyScaler()
    ohlcv_by_symbol = make_ohlcv_by_symbol()
    examples = make_examples()

    results = predict_for_examples(examples, model, scaler, ohlcv_by_symbol, TIMEFRAMES)

    assert len(results) == len(examples)
    for r in results:
        assert 'prediction' in r and 'coords' in r
        assert r['prev_close'] == 100.0


def test_run_signal_backtest_produces_valid_stats():
    model = make_model()
    scaler = DummyScaler()
    ohlcv_by_symbol = make_ohlcv_by_symbol()
    examples = make_examples(n=6)

    strategy_cfg = {'min_trend_confidence': 0.0, 'sl_range_fraction': 0.5, 'risk_reward': 2.0,
                     'risk_per_trade_pct': 1.0, 'beam_width': 3}
    result = run_signal_backtest(examples, model, scaler, ohlcv_by_symbol, TIMEFRAMES, strategy_cfg,
                                  intraday_timeframe='4h')

    assert result['trades_count'] + result['skipped_no_trade'] == len(examples)
    assert 0.0 <= result['win_rate'] <= 100.0
    assert result['max_drawdown_pct'] >= 0.0
    assert len(result['equity_curve']) == result['trades_count'] + 1


def test_run_signal_backtest_zero_confidence_threshold_never_skips_any_trade():
    model = make_model()
    scaler = DummyScaler()
    ohlcv_by_symbol = make_ohlcv_by_symbol()
    examples = make_examples(n=4)

    # min_trend_confidence=0 -> bei binaerem trend (kein Neutral-Bucket mehr) wird nie uebersprungen
    strategy_cfg = {'min_trend_confidence': 0.0, 'sl_range_fraction': 0.5, 'risk_reward': 2.0,
                     'risk_per_trade_pct': 1.0, 'beam_width': 3}
    result = run_signal_backtest(examples, model, scaler, ohlcv_by_symbol, TIMEFRAMES, strategy_cfg,
                                  intraday_timeframe='4h')
    assert result['skipped_no_trade'] == 0
    for t in result['trades']:
        assert t['direction'] in ('long', 'short')
