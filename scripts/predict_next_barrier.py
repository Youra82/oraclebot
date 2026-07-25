# scripts/predict_next_barrier.py
# Live-Inferenz + Trading fuer das Barriere-Modell (4h-Kadenz statt taeglich): laedt die
# aktuellste ABGESCHLOSSENE Referenzkerze (Standard 4h), sagt vorher ob zuerst +barrier_pct%
# oder -barrier_pct% erreicht wird, und platziert bei ausreichender Konfidenz einen Live-Trade.
# Analog zu predict_next_candle.py, aber deutlich einfacher (kein Transformer, kein
# Multi-Timeframe-Fenster, keine Preis-Rekonstruktion -- das Barriere-Modell braucht nur die
# aktuellste Referenzkerze).
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from oraclebot.data.features import compute_features
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.strategy.barrier_signal import compute_barrier_signal
from oraclebot.utils.barrier_gate import check_barrier_gate, mark_barrier_run_complete
from oraclebot.utils.data_fetch import fetch_ohlcv_incremental
from oraclebot.utils.telegram import send_message

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def _drop_incomplete_last_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Wie in predict_next_candle.py -- das Modell hat im Training nur abgeschlossene Kerzen
    gesehen (siehe No-Lookahead-Regel in barrier_targets.py)."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz='UTC')
    last_open = df.index[-1]
    close_time = last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return df.iloc[:-1] if now < close_time else df


def load_settings(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_secrets(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                         help="Ignoriert das Zeitfenster-Gate und den Perioden-Marker, laeuft "
                              "sofort. Fuer manuelles Testen -- markiert die Periode NICHT als erledigt.")
    parser.add_argument('--simulate-now', type=str, default=None,
                         help="Ueberschreibt die fuers Gate verwendete UTC-Zeit ('YYYY-MM-DD HH:MM'). "
                              "Nur fuers Testen des Gate+Marker-Zusammenspiels.")
    parser.add_argument('--marker-path', type=str, default=None)
    args = parser.parse_args()

    settings_path = os.path.join(os.path.dirname(__file__), '..', 'settings.json')
    settings = load_settings(settings_path)
    barrier_cfg = settings['barrier_strategy_settings']

    symbol = barrier_cfg.get('symbol', 'BTC/USDT:USDT')
    reference_tf = barrier_cfg.get('reference_timeframe', '4h')
    intraday_tf = barrier_cfg.get('intraday_timeframe', '15m')
    barrier_pct = barrier_cfg.get('barrier_pct', 1.0)
    min_confidence = barrier_cfg.get('min_confidence', 0.60)

    now_utc = pd.Timestamp(args.simulate_now, tz='UTC') if args.simulate_now else pd.Timestamp.now(tz='UTC')
    if args.simulate_now:
        logger.warning(f"--simulate-now aktiv: Gate+Marker verwenden {now_utc} statt der echten Systemzeit.")
    marker_path = args.marker_path or os.path.join(
        os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'last_barrier_run.txt')
    if not args.force:
        should_run, skip_reason = check_barrier_gate(now_utc, marker_path, period_hours=4)
        if not should_run:
            print(skip_reason)
            sys.exit(0)

    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets')
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    model_path = os.path.join(artifacts_dir, f"barrier_model_{safe_symbol}_{reference_tf}.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Kein Barriere-Modell gefunden: {model_path}. Erst train_barrier_model.py ausfuehren.")
        sys.exit(1)
    predictor = BarrierPredictor.load(model_path)

    logger.info(f"Lade Marktdaten fuer {symbol} ({reference_tf}, inkrementeller Live-Cache)...")
    cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_{reference_tf}.pkl")
    # Genug Historie fuer das laengste Feature-Warmup (EMA-50/MACD) plus Sicherheitsmarge.
    min_candles = 120
    df = fetch_ohlcv_incremental(symbol, reference_tf, min_candles=min_candles, cache_path=cache_path)
    df = _drop_incomplete_last_candle(df, reference_tf)
    logger.info(f"  {reference_tf}: {len(df)} Kerzen, letzte abgeschlossene: {df.index[-1]}")

    # Sicherheitsnetz gegen stille Cache-/Fetch-Fehler (wie predict_next_candle.py): die letzte
    # abgeschlossene Referenzkerze darf nicht aelter als das 2-fache der Periodenlaenge sein.
    staleness = now_utc - df.index[-1]
    max_staleness = pd.Timedelta(minutes=TIMEFRAME_MINUTES[reference_tf]) * 2
    if staleness > max_staleness and not args.force:
        message = (f"ACHTUNG oraclebot (Barriere-Strategie): letzte abgeschlossene {reference_tf}-Kerze "
                    f"ist {staleness} alt (Grenze: {max_staleness}). Moeglicher Fetch-/Cache-Fehler -- "
                    f"breche ab statt eine Prognose fuer eine veraltete Kerze zu senden.")
        logger.error(message)
        secrets = load_secrets(os.path.join(os.path.dirname(__file__), '..', 'secret.json'))
        telegram_cfg = secrets.get('telegram', {})
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)
        sys.exit(1)

    feat = compute_features(df, **barrier_cfg['feature_settings'])
    if len(feat) == 0:
        logger.error("compute_features() lieferte keine Zeilen (zu wenig Historie fuer Warmup). Breche ab.")
        sys.exit(1)

    from oraclebot.data.features import FEATURE_NAMES
    last_row = feat[FEATURE_NAMES].iloc[-1]
    entry_price = float(df.loc[feat.index[-1], 'close'])
    predicted_class, confidence = predictor.predict_one(last_row.values)
    from oraclebot.data.barrier_targets import BARRIER_LABELS
    logger.info(f"\nReferenzkerze: {feat.index[-1]} | Entry: {entry_price:.2f}")
    logger.info(f"Vorhersage: {BARRIER_LABELS[predicted_class]} (Konfidenz: {confidence:.1%})")

    signal = compute_barrier_signal(predicted_class, confidence, entry_price,
                                     min_confidence=min_confidence, barrier_pct=barrier_pct)

    if signal['direction'] is None:
        logger.info(f"Kein Trade ({signal['reason']}, Konfidenz {signal['confidence']:.1%} < {min_confidence:.1%}).")
    else:
        logger.info(f"Signal: {signal['direction'].upper()} | SL: {signal['stop_loss']:.2f} | TP: {signal['take_profit']:.2f}")

    secret_path = os.path.join(os.path.dirname(__file__), '..', 'secret.json')
    secrets = load_secrets(secret_path)
    telegram_cfg = secrets.get('telegram', {})

    if barrier_cfg.get('live_trading_enabled', False):
        oraclebot_accounts = secrets.get('oraclebot', [])
        if not oraclebot_accounts or not oraclebot_accounts[0].get('apiKey'):
            logger.error("live_trading_enabled=true, aber keine 'oraclebot'-API-Keys in secret.json gefunden.")
        else:
            from oraclebot.strategy.live_trade import execute_live_trade
            from oraclebot.utils.exchange import Exchange
            exchange = Exchange(oraclebot_accounts[0])
            am_state_path = os.path.join(artifacts_dir, '..', 'state', 'barrier_anti_martingale_state.json')
            result = execute_live_trade(exchange, signal, symbol, barrier_cfg, telegram_cfg, state_path=am_state_path)
            logger.info(f"\nLive-Trading-Ergebnis: {result}")
    else:
        logger.info("\n(Dry-Run: barrier_strategy_settings.live_trading_enabled=false -- kein echter Trade.)")

    if settings.get('notification_settings', {}).get('telegram_enabled', False):
        dir_text = signal['direction'].upper() if signal['direction'] else 'KEIN TRADE'
        message = (
            f"oraclebot Barriere-Signal: {symbol} ({reference_tf})\n"
            f"Referenzkerze: {feat.index[-1]}\n"
            f"Entry: {entry_price:.2f}\n"
            f"Vorhersage: {BARRIER_LABELS[predicted_class]} (Konfidenz {confidence:.1%})\n"
            f"Richtung: {dir_text}"
        )
        if signal['direction']:
            message += f"\nSL: {signal['stop_loss']:.2f}\nTP: {signal['take_profit']:.2f}"
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    if not args.force:
        mark_barrier_run_complete(now_utc, marker_path, period_hours=4)
