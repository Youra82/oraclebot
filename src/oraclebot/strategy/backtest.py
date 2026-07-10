# src/oraclebot/strategy/backtest.py
# Walk-Forward-Backtest des Handelssignals (signal.py) gegen echte historische Kerzen.
#
# WICHTIG: sollte auf Out-of-Sample-Beispielen laufen (die Validierungs-Beispiele aus dem
# Training, nicht die Trainingsbeispiele selbst) -- sonst wird effektiv die Faehigkeit
# gemessen, auswendig gelernte Daten zu reproduzieren, nicht die Vorhersagequalitaet auf
# ungesehenen Daten (siehe die Overfitting-Analyse im Training).
import numpy as np
import pandas as pd
import ta
import torch

from oraclebot.data.features import FEATURE_NAMES
from oraclebot.model.reconstruct import reconstruct_candle
from oraclebot.strategy.signal import compute_position_size, compute_trade_signal


def _lookup_prev_close_and_atr(daily_df: pd.DataFrame, ref_time: pd.Timestamp, atr_window: int) -> tuple:
    """Anker-Preis + ATR aus der Referenz-Tageskerze (wie predict_next_candle.py).

    Faellt auf die einfache mittlere Tagesrange zurueck, falls weniger als `atr_window + 1`
    Kerzen vor der Referenzkerze vorliegen -- die `ta`-Bibliothek wirft sonst einen IndexError
    statt NaN zu produzieren (siehe gleiches Problem in features.py/targets.py).
    """
    daily_up_to_ref = daily_df.loc[:ref_time]
    prev_close = float(daily_up_to_ref['close'].iloc[-1])
    if len(daily_up_to_ref) < atr_window + 1:
        atr_value = float((daily_up_to_ref['high'] - daily_up_to_ref['low']).mean())
    else:
        atr_series = ta.volatility.AverageTrueRange(
            high=daily_up_to_ref['high'], low=daily_up_to_ref['low'], close=daily_up_to_ref['close'],
            window=atr_window).average_true_range()
        atr_value = float(atr_series.iloc[-1])
    return prev_close, atr_value


def predict_for_examples(examples: list, model, scaler, ohlcv_by_symbol: dict, timeframes: list,
                          atr_window: int = 14, beam_width: int = 5, device=None,
                          tree_ensemble=None) -> list:
    """Fuehrt fuer jedes Beispiel eine Modell-Vorhersage durch und rekonstruiert die Preis-Koordinaten.

    Nutzt die bereits im Beispiel gespeicherten Feature-Fenster (kein erneutes compute_features).

    `tree_ensemble`: optionaler TreeEnsemblePredictor (Hybrid-Ansatz, 2026-07-10) -- ersetzt die
    Transformer-eigenen Vorhersagen fuer TREE_TARGETS (trend/close_position/upper_wick/
    lower_wick, die schwachen, kollapsanfaelligen Ziele) durch RandomForest-Vorhersagen auf
    denselben Features. range/gap_yn/inside_outside_day/high_first bleiben vom Transformer.

    Returns:
        Liste von dicts: {symbol, date, reference_time, target, prediction, prev_close, atr, coords}.
    """
    device = device or torch.device('cpu')
    model.eval()
    results = []

    for example in examples:
        symbol = example['symbol']
        daily_df = ohlcv_by_symbol[symbol]['1d']
        ref_time = pd.Timestamp(example['reference_time'])
        if ref_time not in daily_df.index:
            continue

        features = {
            tf: torch.tensor(scaler.transform_array(np.array(example[tf], dtype=np.float32)),
                              dtype=torch.float32).unsqueeze(0).to(device)
            for tf in timeframes
        }
        with torch.no_grad():
            prediction = model.predict_beam(features, beam_width=beam_width)

        if tree_ensemble is not None:
            tree_prediction = tree_ensemble.predict(example, scaler, timeframes)
            for target in tree_ensemble.models:
                prediction[target] = tree_prediction[target]
                prediction['step_probabilities'][target] = tree_prediction['tree_probabilities'][target]

        prev_close, atr_value = _lookup_prev_close_and_atr(daily_df, ref_time, atr_window)
        coords = reconstruct_candle(
            prev_close=prev_close, atr=atr_value, trend=prediction['trend'], range_cat=prediction['range'],
            close_position_cat=prediction['close_position'], upper_wick_cat=prediction['upper_wick'],
            lower_wick_cat=prediction['lower_wick'])

        results.append({
            'symbol': symbol, 'date': example['date'], 'reference_time': example['reference_time'],
            'target': example['target'], 'prediction': prediction,
            'prev_close': prev_close, 'atr': atr_value, 'coords': coords,
        })

    return results


def determine_trade_outcome(signal: dict, day_bars: pd.DataFrame) -> dict:
    """Prueft anhand echter Intraday-Kerzen (z.B. 4h) INNERHALB des Ziel-Tages, ob SL oder TP

    zuerst getroffen wurde -- chronologisch, nicht nur "beides waere getroffen worden".
    Werden weder SL noch TP getroffen, wird am Tagesende zum letzten verfuegbaren Schlusskurs
    glattgestellt ('timeout').

    Args:
        signal: Rueckgabe von compute_trade_signal() (braucht 'direction' != None).
        day_bars: OHLC-Kerzen des Ziel-Tages, chronologisch sortiert.

    Returns:
        dict mit 'outcome' ('win'/'loss'/'timeout'), 'exit_price', 'pnl_price', 'r_multiple'.
    """
    direction = signal['direction']
    entry, sl, tp = signal['entry'], signal['stop_loss'], signal['take_profit']

    for _, bar in day_bars.iterrows():
        if direction == 'long':
            hit_sl = bar['low'] <= sl
            hit_tp = bar['high'] >= tp
        else:
            hit_sl = bar['high'] >= sl
            hit_tp = bar['low'] <= tp
        if hit_sl:
            # Konservative Annahme, falls SL und TP in derselben Intraday-Kerze beide
            # getroffen wuerden: SL zuerst (Standard-Backtest-Konvention).
            return _make_outcome('loss', sl, signal)
        if hit_tp:
            return _make_outcome('win', tp, signal)

    exit_price = float(day_bars['close'].iloc[-1]) if len(day_bars) else entry
    return _make_outcome('timeout', exit_price, signal)


def _make_outcome(result: str, exit_price: float, signal: dict) -> dict:
    direction_sign = 1 if signal['direction'] == 'long' else -1
    pnl_price = direction_sign * (exit_price - signal['entry'])
    r_multiple = pnl_price / signal['sl_distance'] if signal['sl_distance'] else 0.0
    return {'outcome': result, 'exit_price': exit_price, 'pnl_price': pnl_price, 'r_multiple': r_multiple}


def run_signal_backtest(examples: list, model, scaler, ohlcv_by_symbol: dict, timeframes: list,
                         strategy_cfg: dict, intraday_timeframe: str = '4h',
                         atr_window: int = 14, start_capital: float = 100.0,
                         device=None, tree_ensemble=None) -> dict:
    """Walk-Forward-Backtest: fuer jedes Beispiel Vorhersage -> Signal -> echtes Ergebnis pruefen.

    Args:
        examples: Trainingsbeispiele (idealerweise die Out-of-Sample-Validierungsmenge).
        strategy_cfg: dict mit 'min_trend_confidence', 'sl_range_fraction', 'risk_reward',
            'risk_per_trade_pct', 'beam_width'.
        tree_ensemble: optionaler TreeEnsemblePredictor (Hybrid-Ansatz), siehe predict_for_examples.

    Returns:
        dict mit 'trades' (Liste), 'trades_count', 'win_rate', 'total_pnl_pct',
        'max_drawdown_pct', 'end_capital', 'equity_curve', 'skipped_no_trade'.
    """
    predictions = predict_for_examples(
        examples, model, scaler, ohlcv_by_symbol, timeframes,
        atr_window=atr_window, beam_width=strategy_cfg.get('beam_width', 5), device=device,
        tree_ensemble=tree_ensemble)

    trades = []
    capital = start_capital
    equity_curve = [capital]
    skipped_no_trade = 0

    for pred in predictions:
        signal = compute_trade_signal(
            pred['prediction'], prev_close=pred['prev_close'], atr=pred['atr'],
            min_trend_confidence=strategy_cfg.get('min_trend_confidence', 0.40),
            sl_range_fraction=strategy_cfg.get('sl_range_fraction', 0.5),
            risk_reward=strategy_cfg.get('risk_reward', 2.0))

        if signal['direction'] is None:
            skipped_no_trade += 1
            continue

        intraday_df = ohlcv_by_symbol[pred['symbol']][intraday_timeframe]
        day_start = pd.Timestamp(pred['date'])
        day_end = day_start + pd.Timedelta(days=1)
        day_bars = intraday_df[(intraday_df.index >= day_start) & (intraday_df.index < day_end)]

        outcome = determine_trade_outcome(signal, day_bars)
        size = compute_position_size(
            capital, strategy_cfg.get('risk_per_trade_pct', 1.0), signal['entry'], signal['stop_loss'])
        pnl_usd = outcome['pnl_price'] * size
        capital += pnl_usd
        equity_curve.append(capital)

        trades.append({
            'symbol': pred['symbol'], 'date': pred['date'], 'direction': signal['direction'],
            'entry': signal['entry'], 'stop_loss': signal['stop_loss'], 'take_profit': signal['take_profit'],
            'confidence': signal['confidence'], 'outcome': outcome['outcome'],
            'exit_price': outcome['exit_price'], 'r_multiple': outcome['r_multiple'],
            'pnl_usd': pnl_usd, 'capital_after': capital,
        })

    n_trades = len(trades)
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    win_rate = (wins / n_trades * 100) if n_trades else 0.0
    total_pnl_pct = (capital - start_capital) / start_capital * 100

    peak = start_capital
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    return {
        'trades': trades,
        'trades_count': n_trades,
        'win_rate': win_rate,
        'total_pnl_pct': total_pnl_pct,
        'max_drawdown_pct': max_dd * 100,
        'end_capital': capital,
        'equity_curve': equity_curve,
        'skipped_no_trade': skipped_no_trade,
    }
