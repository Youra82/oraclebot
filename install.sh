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
echo "  3. Ausfuehrungsrecht fuer den Watchdog sicherstellen (chmod +x deckt nur Root-.sh-Dateien ab):"
echo "     chmod +x scripts/watchdog_renko.sh"
echo "  4. Watchdog-Cronjob einrichten (dauerhaft laufender Echtzeit-Prozess, siehe README):"
echo "     crontab -e"
echo "     * * * * * $(pwd)/scripts/watchdog_renko.sh"
echo ""
