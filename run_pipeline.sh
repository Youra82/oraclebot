#!/bin/bash
# run_pipeline.sh — oraclebot Trainings-Pipeline
#
# Schritt 1: Optionen abfragen (History-Tage, frischer Datenabruf)
# Schritt 2: train_barrier_model.py → Modell trainieren + Walk-Forward-Check
# Schritt 3: show_results.py        → Zusammenfassung
#
# Anders als bei dnabot/zerobot gibt es hier KEINE Coin-/Timeframe-Auswahl -- oraclebot
# handelt bewusst nur EIN Symbol (BTC/USDT:USDT, siehe README) mit einem festen
# Multi-Timeframe-Kontext aus settings.json.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ ! -f ".venv/bin/activate" ]; then
    echo -e "${RED}Fehler: Virtuelle Umgebung nicht gefunden. Bitte install.sh ausfuehren.${NC}"
    exit 1
fi
source .venv/bin/activate
export PYTHONPATH=src
echo -e "${GREEN}✔ Virtuelle Umgebung aktiviert.${NC}"

echo ""
echo "======================================================="
echo "       oraclebot — 4h-Barriere-Strategie (BTC/USDT)"
echo "======================================================="
echo ""

# ── 1. History-Tage ──────────────────────────────────────────────────────────
DEFAULT_HISTORY_DAYS=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['history_days'])" 2>/dev/null || echo 1000)
read -p "History-Tage [Standard: $DEFAULT_HISTORY_DAYS aus settings.json]: " HISTORY_INPUT
HISTORY_INPUT="${HISTORY_INPUT//[$'\r\n ']/}"
HISTORY_ARG=""
if [[ "$HISTORY_INPUT" =~ ^[0-9]+$ ]]; then
    HISTORY_ARG="--history-days $HISTORY_INPUT"
    echo -e "${CYAN}ℹ  Ueberschriebene History: ${HISTORY_INPUT} Tage${NC}"
else
    echo -e "${GREEN}✔ Nutze history_days aus settings.json.${NC}"
fi

# ── 2. Komplett neu anfangen? ─────────────────────────────────────────────────
# Anders als bei dnabot (Genome-DB, akkumuliert Wissen ueber mehrere Laeufe) gibt es hier
# KEINE Datenbank und KEIN inkrementelles Lernen -- jeder train_barrier_model.py-Lauf trainiert
# das Modell ohnehin komplett neu (kein Warm-Start), unabhaengig vom vorherigen Modellstand.
# "Neu anfangen" bedeutet hier: Modell/Trainingsdatensatz/Diagnose loeschen -- rein
# kosmetisch/aufraeumend, aendert das Trainingsergebnis selbst NICHT. Betrifft NICHT den
# Live-Zustand (artifacts/state/anti_martingale_state.json) -- der wird hier bewusst nicht
# angefasst, auch wenn live_trading_enabled aktiv ist.
#
# Modell-Reset und OHLCV-Neuabruf sind bewusst zwei UNABHAENGIGE Fragen (nicht wie zuerst
# umgesetzt gekoppelt): ein "kompletter Neustart" soll nicht automatisch einen riskanten
# Komplett-Neuabruf aller Timeframes erzwingen. Historischer Grund (2026-07-26, mittlerweile
# behoben, siehe README-Troubleshooting): Bitgets '1M'-Endpunkt lieferte auf einem VPS
# wiederholt nur einen Bruchteil der angefragten Historie -- '1M' wird deshalb inzwischen gar
# nicht mehr direkt abgefragt, sondern aus '1d' abgeleitet (siehe data_fetch.py). Die
# Entkopplung bleibt trotzdem sinnvoll als generelle Vorsichtsmassnahme.
SYMBOL=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['symbol'])" 2>/dev/null)
REFERENCE_TF=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['reference_timeframe'])" 2>/dev/null)
SAFE_SYMBOL=$(echo "$SYMBOL" | tr '/:' '__')
CONFIG_PATH="src/oraclebot/strategy/configs/config_${SAFE_SYMBOL}_${REFERENCE_TF}.json"

echo ""
echo -e "${CYAN}ℹ  Hinweis: oraclebot hat keine Datenbank und kein inkrementelles Lernen -- jedes${NC}"
echo -e "${CYAN}   Training ist ohnehin ein kompletter Neustart. \"Loeschen\" entfernt nur lokale${NC}"
echo -e "${CYAN}   Ergebnisdateien, aendert aber nichts am eigentlichen Trainingsergebnis. Die${NC}"
echo -e "${CYAN}   Strategie-Config (min_confidence/model_max_depth/Anti-Martingale, siehe${NC}"
echo -e "${CYAN}   $CONFIG_PATH) wird MIT geloescht -- danach gelten Code-Standardwerte,${NC}"
echo -e "${CYAN}   bis ./optimize.sh erneut laeuft.${NC}"
read -p "Bisheriges Modell, Trainingsdatensatz, Diagnose UND Strategie-Config loeschen und komplett neu anfangen? (j/n) [Standard: n]: " RESET_ALL
RESET_ALL="${RESET_ALL//[$'\r\n ']/}"
if [[ "$RESET_ALL" == "j" || "$RESET_ALL" == "J" || "$RESET_ALL" == "y" || "$RESET_ALL" == "Y" ]]; then
    rm -f artifacts/datasets/barrier_model_*.pkl
    rm -f artifacts/datasets/barrier_*.jsonl
    rm -f artifacts/datasets/barrier_diagnostics_*.json
    rm -f "$CONFIG_PATH"
    echo -e "${GREEN}✔ Modell/Datensatz/Diagnose/Strategie-Config geloescht -- kompletter Neustart.${NC}"
else
    echo -e "${GREEN}✔ Bestehendes Modell/Datensatz/Config bleiben vorerst erhalten (werden gleich ueberschrieben).${NC}"
fi

echo ""
read -p "Gecachte OHLCV-Daten ignorieren und frisch abrufen? (j/n) [Standard: n]: " NO_CACHE
NO_CACHE="${NO_CACHE//[$'\r\n ']/}"
CACHE_ARG=""
if [[ "$NO_CACHE" == "j" || "$NO_CACHE" == "J" || "$NO_CACHE" == "y" || "$NO_CACHE" == "Y" ]]; then
    CACHE_ARG="--no-cache"
    echo -e "${CYAN}ℹ  Erzwinge frischen Abruf aller Timeframes (kann mehrere Minuten dauern).${NC}"
else
    echo -e "${GREEN}✔ Nutze vorhandenen Cache, wo verfuegbar.${NC}"
fi

# ── Pipeline starten ─────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Pipeline startet..."
echo "======================================================="
echo ""

echo -e "${YELLOW}[Schritt 1/2] Modell trainieren (train_barrier_model.py)...${NC}"
if ! python3 scripts/train_barrier_model.py $HISTORY_ARG $CACHE_ARG; then
    echo -e "${RED}Training fehlgeschlagen.${NC}"
    deactivate
    exit 1
fi

echo ""
echo -e "${YELLOW}[Schritt 2/2] Ergebnisse...${NC}"
python3 scripts/show_results.py

echo ""
echo "======================================================="
echo -e "  ${GREEN}Pipeline abgeschlossen!${NC}"
echo ""
echo "  Naechste Schritte:"
echo "    1. Ergebnisse/Chart/Excel ansehen: ./show_results.sh"
echo "    2. Modell committen + pushen (git-getrackt, VPS braucht kein eigenes Training):"
echo "         ./push_configs.sh"
echo "    3. Auf dem VPS aktualisieren: ./update.sh"
echo "======================================================="

deactivate
