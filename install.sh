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
echo "  1. secret.json mit Telegram-Bot-Token/Chat-ID befuellen (Vorlage: secret.json.example)"
echo "  2. settings.json pruefen (notification_settings.telegram_enabled)"
echo "  3. Cronjob einrichten (laeuft alle 15 Min wie die anderen Bots; das Skript selbst"
echo "     erkennt per Python-eigener UTC-Uhr das taegliche Fenster kurz nach 00:00 UTC und"
echo "     tut bei allen anderen Aufrufen nichts -- kein CRON_TZ noetig, siehe README):"
echo "     crontab -e"
echo "     */15 * * * * /usr/bin/flock -n $(pwd)/oraclebot.lock /bin/sh -c \"sleep 60; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 cd $(pwd) && $(pwd)/.venv/bin/python3 scripts/predict_next_candle.py >> $(pwd)/logs/cron.log 2>&1\""
echo ""
