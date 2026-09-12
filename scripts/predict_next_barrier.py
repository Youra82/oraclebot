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

from oraclebot.data.features import FEATURE_NAMES
from oraclebot.data.live_features import (FeatureReconstructionError, TIMEFRAME_MINUTES,
                                           build_reference_feature_vector)
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.strategy.barrier_signal import compute_barrier_signal
from oraclebot.utils.barrier_gate import check_barrier_gate, mark_barrier_run_complete
from oraclebot.utils.config import load_barrier_config
from oraclebot.utils.config import load_settings as load_settings_json
from oraclebot.utils.telegram import send_message


def _log_feature_block(label: str, ts, names: list, values: list) -> None:
    """Loggt einen Feature-Block als eine JSON-Zeile (Timestamp + Name->Wert, 8 Nachkommastellen).

    Eingefuehrt 2026-09-05 zur Live-vs-Offline-Diagnose: ein rekonstruierter Backtest mit der
    exakt gleichen, aus Git extrahierten Modell-Datei ergab fuer dieselbe historische
    Referenzkerze (identischer Entry-Preis) die GEGENTEILIGE Richtung bei abweichender Konfidenz
    (Live: down_first 66.4% vs. Offline-Rekonstruktion: up_first 63.2%, Kerze vom 2026-08-31
    08:00 UTC, 4 Stunden alt zum Entscheidungszeitpunkt -- also kein "Kerze noch nicht
    settled"-Effekt). Da nur Konfidenz/Richtung geloggt wurden, liess sich nicht feststellen,
    welcher der ~21 Werte je Block (Referenz + 5 Kontext-Timeframes) dafuer verantwortlich ist.
    Diese Zeilen ermoeglichen den direkten Abgleich mit den 'features' aus dem Offline-Datensatz
    (artifacts/datasets/barrier_*.jsonl) fuer dieselbe Referenzkerze."""
    payload = {'block': label, 'ts': str(ts), **{n: round(float(v), 8) for n, v in zip(names, values)}}
    logger.info(f"FEATURE-DUMP {json.dumps(payload, sort_keys=False)}")


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

    settings = load_settings_json()
    barrier_cfg = load_barrier_config(settings)

    symbol = barrier_cfg.get('symbol', 'BTC/USDT:USDT')
    reference_tf = barrier_cfg.get('reference_timeframe', '4h')
    intraday_tf = barrier_cfg.get('intraday_timeframe', '15m')
    context_tfs = barrier_cfg.get('context_timeframes', [])
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

    # Taeglicher Live-vs-Modell-Signalvergleich: reuse denselben 15-Minuten-Cron statt eines
    # eigenen Cronjobs, ueber ein eigenes 24h-Zeitfenster-Gate (check_barrier_gate mit
    # period_hours=24 feuert einmal taeglich in den ersten 30 Minuten nach 00:00 UTC, eigener
    # Marker verhindert Doppel-Laeufe -- exakt dasselbe Muster wie das 4h-Gate oben). Laeuft VOR
    # der eigentlichen Signal-/Trade-Logik und in einem eigenen try/except, damit ein Fehler hier
    # niemals das eigentliche Live-Trading dieses Laufs verhindert.
    daily_check_marker = os.path.join(artifacts_dir, 'last_signal_check_run.txt')
    should_run_daily_check, _ = check_barrier_gate(now_utc, daily_check_marker, period_hours=24)
    if should_run_daily_check or args.force:
        try:
            from oraclebot.analysis.live_signal_check import (append_report_to_log, build_signal_comparison,
                                                                format_telegram_report)
            check_secrets = load_secrets(os.path.join(os.path.dirname(__file__), '..', 'secret.json'))
            check_accounts = check_secrets.get('oraclebot', [])
            if check_accounts and check_accounts[0].get('apiKey'):
                from oraclebot.utils.exchange import Exchange
                check_exchange = Exchange(check_accounts[0])
                since_ts = now_utc - pd.Timedelta(hours=24)
                daily_report = build_signal_comparison(barrier_cfg, predictor, check_exchange, artifacts_dir,
                                                        since_ts=since_ts, now_utc=now_utc)
                append_report_to_log(daily_report, os.path.join(artifacts_dir, 'signal_comparison_history.jsonl'))
                logger.info(f"Taeglicher Signal-Check: {daily_report['n_match']}/{daily_report['n_comparable']} "
                            f"Match ({daily_report['n_trades']} Live-Trades seit {since_ts}).")
                if settings.get('notification_settings', {}).get('telegram_enabled', False):
                    check_telegram_cfg = check_secrets.get('telegram', {})
                    send_message(check_telegram_cfg.get('bot_token'), check_telegram_cfg.get('chat_id'),
                                 format_telegram_report(daily_report, symbol))
            else:
                logger.info("Taeglicher Signal-Check uebersprungen: keine 'oraclebot'-API-Keys in secret.json.")
        except Exception as e:
            logger.error(f"Taeglicher Signal-Check fehlgeschlagen (Live-Trading laeuft trotzdem weiter): {e}",
                         exc_info=True)
        if should_run_daily_check and not args.force:
            mark_barrier_run_complete(now_utc, daily_check_marker, period_hours=24)

    logger.info(f"Lade Marktdaten fuer {symbol} ({reference_tf}, inkrementeller Live-Cache)...")
    try:
        result = build_reference_feature_vector(symbol, reference_tf, context_tfs, barrier_cfg,
                                                  artifacts_dir, ref_ts=None, now_utc=now_utc)
    except FeatureReconstructionError as e:
        logger.error(f"{e} Breche ab.")
        sys.exit(1)

    ref_ts = result['ref_ts']
    entry_price = result['entry_price']
    feature_row = result['feature_row']
    for label, ts, values in result['blocks']:
        _log_feature_block(label, ts, FEATURE_NAMES, values)
    logger.info(f"  {reference_tf}: letzte abgeschlossene Kerze: {ref_ts}")

    # Sicherheitsnetz gegen stille Cache-/Fetch-Fehler (wie predict_next_candle.py): die letzte
    # abgeschlossene Referenzkerze darf nicht aelter als das 2-fache der Periodenlaenge sein.
    staleness = now_utc - ref_ts
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

    # Kontext-Timeframes duerfen ebenfalls nicht zu weit hinter der Referenzkerze zurueckliegen
    # (moeglicher Fetch-/Cache-Fehler -- siehe build_reference_feature_vector fuer die eigentliche
    # merge_asof-Logik, hier nur noch die Frische-Pruefung je Block).
    for ctx_tf, ctx_ts, _ in result['blocks'][1:]:
        ctx_gap = ref_ts - ctx_ts
        max_ctx_gap = pd.Timedelta(minutes=TIMEFRAME_MINUTES[ctx_tf]) * 2
        if ctx_gap > max_ctx_gap and not args.force:
            message = (f"ACHTUNG oraclebot (Barriere-Strategie): Kontext-Timeframe {ctx_tf} ist "
                        f"{ctx_gap} hinter der Referenzkerze zurueck (Grenze: {max_ctx_gap}). "
                        f"Moeglicher Fetch-/Cache-Fehler -- breche ab statt mit veraltetem Kontext zu handeln.")
            logger.error(message)
            secrets_early = load_secrets(os.path.join(os.path.dirname(__file__), '..', 'secret.json'))
            telegram_early = secrets_early.get('telegram', {})
            send_message(telegram_early.get('bot_token'), telegram_early.get('chat_id'), message)
            sys.exit(1)
        logger.info(f"  {ctx_tf}: letzte abgeschlossene Kerze <= Referenz: {ctx_ts}")

    predicted_class, confidence = predictor.predict_one(feature_row)
    from oraclebot.data.barrier_targets import BARRIER_LABELS
    logger.info(f"\nReferenzkerze: {ref_ts} | Entry: {entry_price:.2f}")
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
            f"Referenzkerze: {ref_ts}\n"
            f"Entry: {entry_price:.2f}\n"
            f"Vorhersage: {BARRIER_LABELS[predicted_class]} (Konfidenz {confidence:.1%})\n"
            f"Richtung: {dir_text}"
        )
        if signal['direction']:
            message += f"\nSL: {signal['stop_loss']:.2f}\nTP: {signal['take_profit']:.2f}"
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    if not args.force:
        mark_barrier_run_complete(now_utc, marker_path, period_hours=4)
