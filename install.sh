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
echo "  2. Erster Download + Wochenauswahl:  .venv/bin/python3 scripts/trend_pool_weekly.py"
echo "  3. Kontrolle:                        .venv/bin/python3 scripts/trend_pool_live.py --dry-run"
echo "  4. Cronjob (jede Stunde in Minute 1, siehe README):"
echo "     1 * * * * cd $(pwd) && .venv/bin/python3 scripts/trend_pool_live.py >> logs/trend_pool_live.log 2>&1"
echo ""
