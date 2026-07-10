# scripts/predict_next_candle.py
# Live-Inferenz: laedt frische Marktdaten, wendet den trainierten Checkpoint an,
# gibt die kategoriale Vorhersage + rekonstruierte Preis-Koordinaten fuer die naechste
# noch nicht abgeschlossene Tageskerze aus.
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd
import ta
import torch

from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.data.scaler import FeatureScaler
from oraclebot.data.targets import (CLOSE_POS_LABELS, GAP_LABELS, HIGH_FIRST_LABELS,
                                     INSIDE_OUTSIDE_LABELS, RANGE_LABELS, TARGET_NAMES,
                                     TREND_LABELS, WICK_LABELS)
from oraclebot.model.reconstruct import reconstruct_candle
from oraclebot.model.transformer import MarketTransformer
from oraclebot.model.tree_ensemble import TreeEnsemblePredictor
from oraclebot.strategy.signal import compute_position_size, compute_trade_signal
from oraclebot.utils.data_fetch import fetch_ohlcv
from oraclebot.utils.telegram import send_message

LABELS_BY_TARGET = {
    'trend': TREND_LABELS, 'range': RANGE_LABELS, 'close_position': CLOSE_POS_LABELS,
    'upper_wick': WICK_LABELS, 'lower_wick': WICK_LABELS, 'gap_yn': GAP_LABELS,
    'inside_outside_day': INSIDE_OUTSIDE_LABELS, 'high_first': HIGH_FIRST_LABELS,
}

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def _drop_incomplete_last_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Entfernt die letzte Kerze, falls sie noch nicht abgeschlossen ist.

    Wichtig fuer Live-Inferenz zu JEDER Tageszeit: das Modell hat im Training nur jemals
    garantiert abgeschlossene Kerzen als Input gesehen (siehe No-Lookahead-Regel in
    dataset.py). Ohne diesen Schritt wuerde z.B. um 06:45 UTC nicht nur die Tageskerze,
    sondern auch die aktuellen 4h-/1h-/15m-Kerzen als "abgeschlossen" behandelt, obwohl sie
    es nicht sind.
    """
    if df.empty:
        return df
    now = pd.Timestamp.now(tz='UTC')
    last_open = df.index[-1]
    close_time = last_open + pd.DateOffset(months=1) if timeframe == '1M' \
        else last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return df.iloc[:-1] if now < close_time else df


def load_settings(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def required_fetch_limit(tf: str, window: int, feature_kwargs: dict) -> int:
    """Wie viele Kerzen fuer `tf` mindestens geholt werden muessen: das Modell-Fenster
    (`window`) plus das Warmup, das compute_features() fuer Indikatoren mit langem Anlauf
    braucht (v.a. EMA-50/MACD-Slow+Signal), plus etwas Sicherheitsmarge.

    Live-Inferenz braucht NICHT dieselbe Datenmenge wie das Training (history_days) --
    Rolling-Indikatoren und das Sliding-Feature-Fenster sind lokal, unabhaengig von der
    Gesamthistorie. Der alte Code fetchte pro Timeframe `history_days`-skaliert (bei 15m mit
    history_days=1000 z.B. ~96000 Kerzen unkached), was bei Bitgets ~100-Kerzen-Chunk-Limit
    viele Minuten pro Lauf gekostet hat -- unpraktikabel fuer taeglichen Cron-Betrieb auf
    einem Rechner ohne nennenswerte Ressourcen (2026-07-10).
    """
    warmup = max(feature_kwargs.get('atr_window', 14), feature_kwargs.get('ema_window', 50),
                 feature_kwargs.get('volume_window', 20), feature_kwargs.get('velocity_window', 10),
                 feature_kwargs.get('rsi_window', 14),
                 feature_kwargs.get('macd_slow', 26) + feature_kwargs.get('macd_signal_window', 9)) + 1
    return window + warmup + 20


def load_secrets(path: str) -> dict:
    """Laedt secret.json, falls vorhanden -- sonst leeres dict (Telegram-Versand wird dann
    stillschweigend uebersprungen, siehe telegram.send_message()). Kein Fehler, damit das
    Script auch ohne konfigurierten Telegram-Bot laeuft (z.B. beim ersten lokalen Test)."""
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def format_telegram_message(symbol: str, target_date, prediction: dict, coords: dict, signal: dict) -> str:
    trend_label = 'LONG (bullisch)' if prediction['trend'] == 1 else 'SHORT (baerisch)'
    lines = [
        f"OracleBot Prognose -- {symbol}",
        f"Fuer Kerze: {target_date.date()}",
        "",
        f"Trend: {trend_label} (Konfidenz {signal['confidence']:.1%})",
        f"Erwartete Spanne: {coords['low']:.2f} - {coords['high']:.2f}",
        f"Body: {coords['body_bottom']:.2f} - {coords['body_top']:.2f}",
    ]
    if signal['direction'] is None:
        lines.append(f"\nKein Handelssignal ({signal['reason']}).")
    else:
        lines.append(f"\nSignal: {signal['direction'].upper()}")
        lines.append(f"Entry: {signal['entry']:.2f}")
        lines.append(f"SL: {signal['stop_loss']:.2f}  TP: {signal['take_profit']:.2f}")
    return "\n".join(lines)


if __name__ == '__main__':
    settings_path = os.path.join(os.path.dirname(__file__), '..', 'settings.json')
    settings = load_settings(settings_path)
    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']

    symbol = 'BTC/USDT:USDT'
    timeframes = model_cfg['timeframes']
    window_sizes = ds_cfg['window_sizes']
    feature_kwargs_by_tf = ds_cfg.get('feature_settings_by_timeframe', {})

    torch.set_num_threads(train_cfg.get('num_threads', 4))

    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets')
    checkpoint_path = os.path.join(artifacts_dir, 'market_transformer_best.pt')
    scaler_path = os.path.join(artifacts_dir, 'scaler_full.pkl')

    logger.info(f"Lade frische Marktdaten fuer {symbol} (kein Cache -- soll der aktuelle Stand sein)...")
    ohlcv_by_timeframe = {}
    for tf in timeframes:
        feature_kwargs = {**ds_cfg['feature_settings'], **feature_kwargs_by_tf.get(tf, {})}
        limit = required_fetch_limit(tf, window_sizes[tf], feature_kwargs)
        df = fetch_ohlcv(symbol, tf, limit=limit + 1)  # +1 Puffer, falls die letzte Kerze noch laeuft und gedroppt wird
        df = _drop_incomplete_last_candle(df, tf)
        ohlcv_by_timeframe[tf] = df
        logger.info(f"  {tf}: {len(df)} Kerzen, letzte abgeschlossene: {df.index[-1]}")

    daily_df = ohlcv_by_timeframe[ds_cfg['reference_timeframe']]
    last_closed_date = daily_df.index[-1]
    target_date = last_closed_date + (last_closed_date - daily_df.index[-2])
    logger.info(f"\nLetzte abgeschlossene Tageskerze: {last_closed_date.date()}")
    logger.info(f"Vorhersage gilt fuer: {target_date.date()}")

    logger.info("\nBaue Feature-Fenster je Timeframe...")
    features_by_timeframe = {}
    for tf in timeframes:
        feats = compute_features(ohlcv_by_timeframe[tf], **{**ds_cfg['feature_settings'], **feature_kwargs_by_tf.get(tf, {})})
        window = feats[FEATURE_NAMES].iloc[-window_sizes[tf]:]
        if len(window) < window_sizes[tf]:
            raise RuntimeError(f"Zu wenig Historie fuer {tf}: {len(window)} < {window_sizes[tf]} benoetigt.")
        features_by_timeframe[tf] = window

    scaler = FeatureScaler.load(scaler_path)
    feature_tensors = {
        tf: torch.tensor(scaler.transform(window), dtype=torch.float32).unsqueeze(0)
        for tf, window in features_by_timeframe.items()
    }

    model = MarketTransformer(
        n_features=len(FEATURE_NAMES), timeframes=timeframes, window_sizes=window_sizes,
        d_model=model_cfg['d_model'], nhead=model_cfg['nhead'], num_encoder_layers=model_cfg['num_encoder_layers'],
        dim_feedforward=model_cfg['dim_feedforward'], dropout=model_cfg['dropout'],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()
    logger.info(f"\nModell geladen: {checkpoint_path}")

    prediction = model.predict_beam(feature_tensors, beam_width=model_cfg['beam_width'])

    # Hybrid-Ansatz (2026-07-10): trend/close_position/upper_wick/lower_wick kommen, falls
    # trainiert, vom RandomForest-Ensemble statt vom Transformer-Decoder -- siehe tree_ensemble.py.
    tree_ensemble_path = os.path.join(artifacts_dir, 'tree_ensemble.pkl')
    if os.path.exists(tree_ensemble_path):
        tree_ensemble = TreeEnsemblePredictor.load(tree_ensemble_path)
        example_like = {tf: window for tf, window in features_by_timeframe.items()}
        tree_prediction = tree_ensemble.predict(example_like, scaler, timeframes)
        for name in tree_ensemble.models:
            prediction[name] = tree_prediction[name]
            prediction['step_probabilities'][name] = tree_prediction['tree_probabilities'][name]
        logger.info("Hybrid-Vorhersage: trend/close_position/upper_wick/lower_wick vom RandomForest-Ensemble.")

    logger.info(f"\nVorhersage fuer die Tageskerze am {target_date.date()}:")
    for name in TARGET_NAMES:
        cat = prediction[name]
        label = LABELS_BY_TARGET[name][cat]
        logger.info(f"  {name:20s}: {label} (Kategorie {cat})")
    logger.info(f"\n  Timeframe-Gewichte: { {k: round(v, 3) for k, v in prediction['timeframe_weights'].items()} }")

    prev_close = float(daily_df['close'].iloc[-1])
    atr_value = float(ta.volatility.AverageTrueRange(
        high=daily_df['high'], low=daily_df['low'], close=daily_df['close'],
        window=ds_cfg['target_settings']['atr_window']).average_true_range().iloc[-1])

    coords = reconstruct_candle(
        prev_close=prev_close, atr=atr_value, trend=prediction['trend'], range_cat=prediction['range'],
        close_position_cat=prediction['close_position'], upper_wick_cat=prediction['upper_wick'],
        lower_wick_cat=prediction['lower_wick'])

    logger.info(f"\nRekonstruierte Preis-Koordinaten (Anker: Close {last_closed_date.date()}={prev_close:.2f}, ATR={atr_value:.2f}):")
    logger.info(f"  High (Docht oben):    {coords['high']:.2f}")
    logger.info(f"  Body oben:            {coords['body_top']:.2f}")
    logger.info(f"  Body unten:           {coords['body_bottom']:.2f}")
    logger.info(f"  Low (Docht unten):    {coords['low']:.2f}")
    logger.info(f"  (Open={coords['open']:.2f}, Close={coords['close']:.2f}, "
                f"close_position_consistent={coords['close_position_consistent']})")

    strat_cfg = settings.get('strategy_settings', {})
    signal = compute_trade_signal(
        prediction, prev_close=prev_close, atr=atr_value,
        min_trend_confidence=strat_cfg.get('min_trend_confidence', 0.40),
        sl_range_fraction=strat_cfg.get('sl_range_fraction', 0.5),
        risk_reward=strat_cfg.get('risk_reward', 2.0),
    )

    logger.info(f"\nHandelssignal (Trend-Konfidenz={signal['confidence']:.1%}):")
    if signal['direction'] is None:
        logger.info(f"  Kein Trade ({signal['reason']}).")
    else:
        logger.info(f"  Richtung:     {signal['direction'].upper()}")
        logger.info(f"  Entry:        {signal['entry']:.2f}")
        logger.info(f"  Stop-Loss:    {signal['stop_loss']:.2f}  (Abstand {signal['sl_distance']:.2f})")
        logger.info(f"  Take-Profit:  {signal['take_profit']:.2f}  (Abstand {signal['tp_distance']:.2f})")
        size = compute_position_size(
            balance=1000.0, risk_per_trade_pct=strat_cfg.get('risk_per_trade_pct', 1.0),
            entry=signal['entry'], stop_loss=signal['stop_loss'])
        logger.info(f"  Positionsgroesse bei 1000 USDT Beispiel-Balance: {size:.6f} BTC")

    if strat_cfg.get('live_trading_enabled', False):
        logger.warning("\nlive_trading_enabled=true, aber Order-Platzierung ist noch nicht implementiert "
                        "(nur Signal-Berechnung, kein Exchange-Anschluss). Es wird KEIN echter Trade platziert.")
    else:
        logger.info("\n(Dry-Run: live_trading_enabled=false in settings.json -- es wird kein echter Trade platziert.)")

    # Telegram-Benachrichtigung ist bewusst UNABHAENGIG von live_trading_enabled: die Prognose
    # soll auch dann ankommen, wenn live_trading_enabled=false ist (reiner Beobachtungsmodus).
    notif_cfg = settings.get('notification_settings', {})
    if notif_cfg.get('telegram_enabled', False):
        secret_path = os.path.join(os.path.dirname(__file__), '..', 'secret.json')
        telegram_cfg = load_secrets(secret_path).get('telegram', {})
        message = format_telegram_message(symbol, target_date, prediction, coords, signal)
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)
    else:
        logger.info("(notification_settings.telegram_enabled=false -- keine Telegram-Nachricht gesendet.)")
