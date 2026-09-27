#!/bin/bash
# scripts/retire_renko.sh -- legt die alte Renko-Echtzeit-Strategie still (einmalig nach dem Update auf den
# Trend-Pool ausfuehren):
#   1. Watchdog-Cronjob entfernen (sonst startet er den Prozess jede Minute neu)
#   2. laufenden Prozess beenden
#   3. offene Renko-Positionen schliessen + Orders stornieren (scripts/retire_renko.py)
cd "$(dirname "$0")/.." || exit 1
set -e

echo "1. Entferne Watchdog-Cronjob fuer watchdog_renko.sh ..."
( crontab -l 2>/dev/null | grep -v 'watchdog_renko.sh' ) | crontab - || true

PID_FILE="artifacts/state/renko_realtime.pid"
echo "2. Beende laufenden Renko-Prozess ..."
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        for i in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
        kill -0 "$PID" 2>/dev/null && kill -9 "$PID" || true
    fi
    rm -f "$PID_FILE"
fi

echo "3. Schliesse offene Renko-Positionen ..."
.venv/bin/python3 scripts/retire_renko.py

echo "Alte Renko-Strategie stillgelegt."
