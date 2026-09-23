#!/bin/bash
# scripts/watchdog_renko.sh -- startet run_renko_realtime.py neu, falls er nicht mehr laeuft.
#
# PID-Datei-basiert statt "pgrep -f <skriptname>" (Fund 2026-09-23): pgrep -f matcht sonst
# faelschlich die eigene aufrufende "sh -c"-Prozesskette, deren Kommandozeile den Pattern-Text
# selbst enthaelt (weil er als Argument in genau diesem Aufruf steckt) -- der Bot startete
# dadurch NIE, der Watchdog dachte bei jedem Tick faelschlich "laeuft schon".
cd "$(dirname "$0")/.." || exit 1

PID_FILE="artifacts/state/renko_realtime.pid"
mkdir -p artifacts/state logs

if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if [ -n "$OLD_PID" ] && kill -0 "$OLD_PID" 2>/dev/null; then
        exit 0  # laeuft noch, nichts zu tun
    fi
fi

nohup .venv/bin/python3 scripts/run_renko_realtime.py >> logs/cron_renko.log 2>&1 &
echo $! > "$PID_FILE"
