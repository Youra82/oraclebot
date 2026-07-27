#!/bin/bash
# optimize.sh — Systematische Parametersuche fuer oraclebot (interaktiver Wrapper)
#
# Fragt Startkapital/Referenz-Timeframe/DD-Ziel/History-Tage ab, ruft dann
# optimize_barrier_model.py auf (Hebel, model_max_depth, min_confidence, Anti-Martingale --
# ausschliesslich per Walk-Forward, der finale 70/30-OOS-Split bleibt bei der Parameterwahl
# strikt unberuehrt, siehe Kommentar in optimize_barrier_model.py). Der Hebel wird seit
# 2026-07-27 selbst gesucht (Liquidations-Sicherheitsgrenze als harter Kandidaten-Filter, siehe
# margin_safety.py) statt hier interaktiv vorgegeben zu werden. Fragt am Ende, ob die
# gefundenen Werte in die Strategie-Config uebernommen werden sollen.

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
echo "       oraclebot — Parameter-Optimierung"
echo "======================================================="
echo ""
echo -e "${CYAN}ℹ  Sucht Hebel (Liquidations-sicher gefiltert), model_max_depth, min_confidence${NC}"
echo -e "${CYAN}   und Anti-Martingale-Parameter -- ausschliesslich per Walk-Forward. Der finale${NC}"
echo -e "${CYAN}   70/30-Out-of-Sample-Split wird NICHT zur Parameterwahl verwendet, nur fuer${NC}"
echo -e "${CYAN}   einen einmaligen Bestaetigungs-Bericht am Ende (strikte OOS-Disziplin).${NC}"
echo ""
echo -e "${CYAN}   NICHT gesucht (siehe README): margin_mode, live_trading_enabled,${NC}"
echo -e "${CYAN}   history_days, val_split, backtest_start_capital, taker_fee_rate_pct,${NC}"
echo -e "${CYAN}   Feature-Fenster -- entweder Strategie-Grundentscheidungen oder ein zu${NC}"
echo -e "${CYAN}   grosser Suchraum fuer die aktuelle Datenmenge (Overfitting-Risiko).${NC}"

# ── 1. Referenz-Timeframe ─────────────────────────────────────────────────────
DEFAULT_REFERENCE_TF=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['reference_timeframe'])" 2>/dev/null || echo "4h")
echo ""
read -p "Referenz-Timeframe [Standard: $DEFAULT_REFERENCE_TF aus settings.json]: " REFERENCE_TF_INPUT
REFERENCE_TF_INPUT="${REFERENCE_TF_INPUT//[$'\r\n ']/}"
REFERENCE_TF_ARG=""
if [ -n "$REFERENCE_TF_INPUT" ]; then
    REFERENCE_TF_ARG="--reference-timeframe $REFERENCE_TF_INPUT"
    echo -e "${CYAN}ℹ  Ueberschriebener Referenz-Timeframe: $REFERENCE_TF_INPUT${NC}"
else
    echo -e "${GREEN}✔ Nutze reference_timeframe aus settings.json.${NC}"
fi

# ── 2. Startkapital ────────────────────────────────────────────────────────────
DEFAULT_CAPITAL=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings'].get('backtest_start_capital', 15.0))" 2>/dev/null || echo 15.0)
echo ""
read -p "Startkapital in USDT [Standard: $DEFAULT_CAPITAL aus settings.json]: " CAPITAL_INPUT
CAPITAL_INPUT="${CAPITAL_INPUT//[$'\r\n ']/}"
CAPITAL_ARG=""
if [[ "$CAPITAL_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    CAPITAL_ARG="--start-capital $CAPITAL_INPUT"
else
    echo -e "${GREEN}✔ Nutze backtest_start_capital aus settings.json.${NC}"
fi

# ── 3. Ziel-MaxDD ──────────────────────────────────────────────────────────────
echo ""
read -p "Ziel-Obergrenze fuer MaxDD in % (90. Perzentil, Bootstrap) [Standard: 50]: " DD_INPUT
DD_INPUT="${DD_INPUT//[$'\r\n ']/}"
if ! [[ "$DD_INPUT" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then DD_INPUT=50; fi
echo -e "${CYAN}ℹ  DD-Ziel: <= ${DD_INPUT}%${NC}"

# ── 4. History-Tage ────────────────────────────────────────────────────────────
DEFAULT_HISTORY_DAYS=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['history_days'])" 2>/dev/null || echo 1000)
echo ""
read -p "History-Tage [Standard: $DEFAULT_HISTORY_DAYS aus settings.json]: " HISTORY_INPUT
HISTORY_INPUT="${HISTORY_INPUT//[$'\r\n ']/}"
HISTORY_ARG=""
if [[ "$HISTORY_INPUT" =~ ^[0-9]+$ ]]; then
    HISTORY_ARG="--history-days $HISTORY_INPUT"
else
    echo -e "${GREEN}✔ Nutze history_days aus settings.json.${NC}"
fi

# ── 5. Frischer Datenabruf? ────────────────────────────────────────────────────
echo ""
read -p "Gecachte OHLCV-Daten ignorieren und frisch abrufen? (j/n) [Standard: n]: " NO_CACHE
NO_CACHE="${NO_CACHE//[$'\r\n ']/}"
CACHE_ARG=""
if [[ "$NO_CACHE" == "j" || "$NO_CACHE" == "J" || "$NO_CACHE" == "y" || "$NO_CACHE" == "Y" ]]; then
    CACHE_ARG="--no-cache"
    echo -e "${CYAN}ℹ  Erzwinge frischen Abruf aller Timeframes (kann mehrere Minuten dauern).${NC}"
fi

# ── Optimierung starten ────────────────────────────────────────────────────────
echo ""
echo "======================================================="
echo "  Optimierung startet..."
echo "======================================================="
echo ""

# optimize_barrier_model.py fragt am Ende selbst interaktiv, ob die gefundenen Werte
# uebernommen werden sollen (kein --apply hier) -- so wird die gesamte teure Suche
# (Walk-Forward-Sweep + Anti-Martingale-Bootstrap-Grid) nur EIN Mal ausgefuehrt.
if ! python3 scripts/optimize_barrier_model.py $REFERENCE_TF_ARG $CAPITAL_ARG \
        --dd-limit "$DD_INPUT" $HISTORY_ARG $CACHE_ARG; then
    echo -e "${RED}Optimierung fehlgeschlagen.${NC}"
    deactivate
    exit 1
fi

echo ""
echo "  Naechste Schritte (falls uebernommen):"
echo "    1. Ergebnisse/Chart/Excel ansehen: ./show_results.sh"
echo "    2. Modell + Config committen + pushen: ./push_configs.sh"
echo "    3. Auf dem VPS aktualisieren: ./update.sh"

deactivate
