# scripts/check_live_signals.py
# Manueller/lokaler Live-vs-Modell-Signalvergleich (siehe analysis/live_signal_check.py) --
# dieselbe Logik, die predict_next_barrier.py taeglich um 00:00 UTC automatisch ausfuehrt, hier
# als eigenstaendiges Skript zum spontanen Nachschauen (z.B. lokal, ohne auf den naechsten
# Cron-Tick auf dem VPS zu warten).
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from oraclebot.analysis.live_signal_check import append_report_to_log, build_signal_comparison, format_telegram_report
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.utils.config import load_barrier_config, load_settings
from oraclebot.utils.exchange import Exchange

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')


def load_secrets() -> dict:
    path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=float, default=7.0,
                         help="Live-Trades der letzten N Tage vergleichen (Standard: 7).")
    parser.add_argument('--telegram', action='store_true', help="Zusammenfassung zusaetzlich per Telegram senden.")
    parser.add_argument('--log', action='store_true',
                         help="Bericht an artifacts/datasets/signal_comparison_history.jsonl anhaengen "
                              "(wie der automatische taegliche Lauf).")
    args = parser.parse_args()

    settings = load_settings()
    barrier_cfg = load_barrier_config(settings)
    safe_symbol = barrier_cfg['symbol'].replace('/', '_').replace(':', '_')
    model_path = os.path.join(ARTIFACTS_DIR, f"barrier_model_{safe_symbol}_{barrier_cfg['reference_timeframe']}.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Kein Modell gefunden: {model_path}.")
        sys.exit(1)
    predictor = BarrierPredictor.load(model_path)

    secrets = load_secrets()
    accounts = secrets.get('oraclebot', [])
    if not accounts or not accounts[0].get('apiKey'):
        logger.error("Keine 'oraclebot'-API-Keys in secret.json -- kann keine Live-Trades abrufen.")
        sys.exit(1)
    exchange = Exchange(accounts[0])

    now_utc = pd.Timestamp.now(tz='UTC')
    since_ts = now_utc - pd.Timedelta(days=args.days)
    report = build_signal_comparison(barrier_cfg, predictor, exchange, ARTIFACTS_DIR, since_ts=since_ts,
                                      now_utc=now_utc)

    print(f"\nZeitraum: seit {since_ts} ({args.days} Tage)")
    print(f"Live-Trades: {report['n_trades']}")
    if report['live_win_rate'] is not None:
        print(f"Live-Winrate: {report['live_win_rate']:.1%} ({report['n_live_wins']}/{report['n_trades']})")
    print(f"Vergleichbar mit aktuellem Modell: {report['n_comparable']}")
    print(f"Signal-Match: {report['n_match']} | Mismatch: {report['n_mismatch']}\n")

    print(f"{'Ref-Kerze':<26}{'Live-Dir':<10}{'PnL':>8}  {'Modell-Dir':<11}{'Konf.':>7}  Match?")
    print('-' * 80)
    for t in report['trades']:
        if 'error' in t:
            print(f"{t['ref_ts']:<26}{t['live_direction']:<10}{t['pnl']:>8.2f}  ({t['error']})")
            continue
        print(f"{t['ref_ts']:<26}{t['live_direction']:<10}{t['pnl']:>8.2f}  "
              f"{str(t['model_direction']):<11}{t['model_confidence']:>6.1%}  {'OK' if t['match'] else 'MISMATCH'}")

    if args.log:
        append_report_to_log(report, os.path.join(ARTIFACTS_DIR, 'signal_comparison_history.jsonl'))

    if args.telegram:
        telegram_cfg = secrets.get('telegram', {})
        from oraclebot.utils.telegram import send_message
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                     format_telegram_report(report, barrier_cfg['symbol']))
        print("\nTelegram-Nachricht gesendet.")
