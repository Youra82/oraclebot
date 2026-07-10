#!/bin/bash
set -e

echo "--- Sicheres Update wird ausgefuehrt ---"

# 1. Sichere secret.json
echo "1. Erstelle ein Backup von 'secret.json'..."
cp secret.json secret.json.bak

# 2. Hole neuesten Stand von GitHub
echo "2. Hole den neuesten Stand von GitHub..."
git fetch origin

# 3. Setze lokales Verzeichnis hart auf GitHub-Stand zurueck
echo "3. Setze alle Dateien auf den neuesten Stand zurueck und verwerfe lokale Aenderungen..."
git reset --hard origin/main

# 4. Stelle secret.json wieder her
echo "4. Stelle 'secret.json' aus dem Backup wieder her..."
cp secret.json.bak secret.json
rm secret.json.bak

# 5. Loesche Python-Cache
echo "5. Loesche alten Python-Cache fuer einen sauberen Neustart..."
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete

# 6. Ausfuehrungsrechte setzen
echo "6. Setze Ausfuehrungsrechte fuer alle .sh-Skripte..."
chmod +x *.sh

# 7. Dependencies aktualisieren
echo "7. Aktualisiere Python-Pakete..."
.venv/bin/pip install -r requirements.txt --quiet

echo "Update erfolgreich abgeschlossen. oraclebot ist jetzt auf dem neuesten Stand."
echo "(market_transformer_best.pt / scaler_full.pkl / tree_ensemble.pkl sind git-getrackt und wurden mit aktualisiert -- kein Neu-Training auf dem VPS noetig.)"
