#!/bin/bash
# oraclebot - Installations-Skript (VPS)

echo "=== oraclebot Installation ==="

# Virtual Environment erstellen
python3 -m venv .venv
echo "venv erstellt."

# Packages installieren
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "Packages installiert."

# Verzeichnisse anlegen
mkdir -p logs

# Skripte ausfuehrbar machen
chmod +x *.sh

# secret.json pruefen
if [ ! -f "secret.json" ]; then
    echo "WARNUNG: secret.json fehlt! Bitte secret.json mit dem Telegram-Bot befuellen."
    echo "Vorlage: secret.json.example"
else
    echo "secret.json gefunden."
fi

echo ""
echo "=== Installation abgeschlossen ==="
echo ""
echo "Naechste Schritte:"
echo "  1. secret.json mit Bitget-API-Keys + Telegram-Bot-Token/Chat-ID befuellen (Vorlage: secret.json.example)"
echo "  2. settings.json::renko_breakout_settings pruefen (enabled bleibt false, bis bewusst aktiviert)"
echo "  3. Cronjob einrichten (alle 5 Minuten, Brick-Timeframe -- siehe README):"
echo "     crontab -e"
echo "     */5 * * * * /usr/bin/flock -n $(pwd)/oraclebot_renko.lock /bin/sh -c \"sleep 20; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 cd $(pwd) && $(pwd)/.venv/bin/python3 scripts/run_renko_breakout.py >> $(pwd)/logs/cron_renko.log 2>&1\""
echo ""
