#!/bin/bash
# show_results.sh — Trainings-Diagnose + Backtest-Ergebnisse fuer oraclebot anzeigen
#
# Braucht ein bereits trainiertes Modell (siehe ./run_pipeline.sh) -- der Trainings-
# Datensatz-Cache ist nicht in Git, muss also lokal einmal durchgelaufen sein.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Muss das Projekt-.venv nutzen (nicht das System-Python vom PATH) -- dort sind pandas/
# sklearn/openpyxl etc. installiert (siehe 2026-07-16-Vorfall: System-Python fuehrte zu
# "ModuleNotFoundError", weil eine fruehere Version dieses Skripts .venv nicht explizit nutzte).
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
else
    echo -e "${RED}Fehler: .venv nicht gefunden. Bitte install.sh ausfuehren.${NC}"
    exit 1
fi
export PYTHONPATH="$SCRIPT_DIR/src"

DEFAULT_CAPITAL=$("$PYTHON" -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings'].get('backtest_start_capital', 15.0))" 2>/dev/null || echo 15.0)
echo ""
read -p "Startkapital in USDT [Standard: $DEFAULT_CAPITAL aus settings.json]: " CAPITAL_INPUT
CAPITAL_INPUT="${CAPITAL_INPUT//[$'\r\n ']/}"
CAPITAL_ARG=""
if [[ "$CAPITAL_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    CAPITAL_ARG="--start-capital $CAPITAL_INPUT"
fi

echo ""
echo -e "${YELLOW}Was moechtest du sehen?${NC}"
echo "  1) Zusammenfassung        (Trainings-Diagnose + Anti-Martingale-Backtest, Konsole)"
echo "  2) Chart aktualisieren    (artifacts/charts/combined_overview.png)"
echo "  3) Excel-Export           (artifacts/charts/oraclebot_trades_<Zeitstempel>.xlsx)"
read -p "Auswahl (1-3) [Standard: 1]: " MODE
MODE="${MODE//[$'\r\n ']/}"
MODE=${MODE:-1}

if [ "$MODE" == "2" ]; then
    echo ""
    read -p "Nur Trades ab diesem Datum (JJJJ-MM-TT) [leer = kompletter Out-of-Sample-Zeitraum]: " SINCE_DATE
    SINCE_DATE="${SINCE_DATE//[$'\r\n ']/}"
    if [ -n "$SINCE_DATE" ]; then
        "$PYTHON" scripts/show_results.py --chart --since "$SINCE_DATE" $CAPITAL_ARG
    else
        "$PYTHON" scripts/show_results.py --chart $CAPITAL_ARG
    fi

elif [ "$MODE" == "3" ]; then
    echo ""
    read -p "Nur Trades ab diesem Datum (JJJJ-MM-TT) [leer = kompletter Out-of-Sample-Zeitraum]: " SINCE_DATE
    SINCE_DATE="${SINCE_DATE//[$'\r\n ']/}"
    if [ -n "$SINCE_DATE" ]; then
        "$PYTHON" scripts/show_results.py --excel --since "$SINCE_DATE" $CAPITAL_ARG
    else
        "$PYTHON" scripts/show_results.py --excel $CAPITAL_ARG
    fi

else
    "$PYTHON" scripts/show_results.py $CAPITAL_ARG
fi
