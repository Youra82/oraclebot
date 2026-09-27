# scripts/retire_renko.py -- schliesst alle offenen Positionen der alten Renko-Echtzeit-Strategie und storniert
# deren Orders (Sicherheits-Stops). Wird von scripts/retire_renko.sh NACH dem Stoppen des Prozesses aufgerufen.
#   .venv/bin/python3 scripts/retire_renko.py --dry-run   # nur anzeigen
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
from oraclebot.utils.config import PROJECT_ROOT, load_settings  # noqa: E402
from oraclebot.utils.telegram import send_message  # noqa: E402

STATE_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'state')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    settings = load_settings()
    if settings['renko_breakout_settings'].get('enabled', False):
        print('renko_breakout_settings.enabled ist noch true -- erst auf false setzen (git pull / update.sh). Abbruch.')
        sys.exit(1)
    with open(os.path.join(PROJECT_ROOT, 'secret.json'), encoding='utf-8') as f:
        secrets = json.load(f)
    from oraclebot.utils.exchange import Exchange
    ex = Exchange(secrets['oraclebot'][0])
    closed = []
    for symbol in settings['renko_breakout_settings']['symbols']:
        pos = ex.fetch_open_positions_strict(symbol)
        print(f"{symbol}: {len(pos)} offene Position(en)")
        if a.dry_run:
            continue
        ex.cancel_all_orders_for_symbol(symbol)
        for _ in pos:
            ex.close_position(symbol)
            closed.append(symbol)
    if not a.dry_run:
        for name in ('renko_realtime_positions.json', 'renko_realtime_bricks.json'):
            p = os.path.join(STATE_DIR, name)
            if os.path.exists(p):
                os.replace(p, p + '.retired')
        tg = secrets.get('telegram', {})
        send_message(tg.get('bot_token'), tg.get('chat_id'),
                     f"ORACLEBOT: alte Renko-Strategie stillgelegt. Geschlossen: {', '.join(closed) or 'keine offenen Positionen'}.")
    print('fertig')


if __name__ == '__main__':
    main()
