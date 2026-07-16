#!/bin/bash
# show_results.sh — OracleBot Ergebnisanzeige

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Muss das Projekt-.venv nutzen (nicht das System-Python vom PATH) -- dort sind pandas/torch/
# sklearn etc. installiert. System-Python fuehrte zu "ModuleNotFoundError: No module named
# 'pandas'" auf dem VPS (2026-07-16), da show_results.sh als einziges Skript nicht wie
# update.sh/run_tests.sh explizit .venv/bin/python3 nutzte.
if [ -x "$SCRIPT_DIR/.venv/bin/python3" ]; then
    PYTHON="$SCRIPT_DIR/.venv/bin/python3"
elif command -v python >/dev/null 2>&1; then
    echo -e "${YELLOW}WARNUNG: .venv nicht gefunden, verwende System-Python -- bitte install.sh ausfuehren.${NC}"
    PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
    echo -e "${YELLOW}WARNUNG: .venv nicht gefunden, verwende System-Python -- bitte install.sh ausfuehren.${NC}"
    PYTHON="python3"
else
    echo -e "${RED}FEHLER: Python nicht gefunden.${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Wähle einen Modus:${NC}"
echo "  1) Einzel-Backtest               (alle Symbole gemeinsam, Out-of-Sample)"
echo "  2) Manuelle Portfolio-Simulation (du wählst die Symbole)"
echo "  3) Automatische Portfolio-Opt.   (Bot testet Signal-Parameter, waehlt das Beste)"
echo "  4) Interaktive Charts            (echte Kerzen + Prognose-Kerzen-Overlay)"
echo ""
read -p "Auswahl (1-4) [Standard: 1]: " MODE
MODE="${MODE//[$'\r\n ']/}"
MODE="${MODE:-1}"

if [[ ! "$MODE" =~ ^[1-4]$ ]]; then
    echo -e "${RED}Ungültige Eingabe. Verwende Modus 1.${NC}"
    MODE=1
fi

# ── Modus 4: fragt intern selbst nach Symbol/Datum ───────────────────────────
if [ "$MODE" == "4" ]; then
    "$PYTHON" "$SCRIPT_DIR/scripts/interactive_chart.py"
    exit 0
fi

# ── Modus 3: Auto-Optimierung (fragt nur Kapital + Max-DD) ───────────────────
if [ "$MODE" == "3" ]; then
    echo ""
    read -p "Startkapital in USDT [Standard: 100]: " CAP
    CAP="${CAP//[$'\r\n ']/}"
    if ! [[ "$CAP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then CAP=100; fi

    read -p "Max. Drawdown in % [Standard: 30]: " MAX_DD
    MAX_DD="${MAX_DD//[$'\r\n ']/}"
    if ! [[ "$MAX_DD" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then MAX_DD=30; fi

    "$PYTHON" "$SCRIPT_DIR/scripts/backtest_signal.py" --mode 3 --capital "$CAP" --max-dd "$MAX_DD"
    exit 0
fi

# ── Modi 1-2: Startkapital abfragen ───────────────────────────────────────────
echo ""
read -p "Startkapital in USDT [Standard: 100]: " CAP
CAP="${CAP//[$'\r\n ']/}"
if ! [[ "$CAP" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then CAP=100; fi

echo ""

# ── Modus 2: braucht interaktives stdin fuer Symbol-Auswahl ──────────────────
if [ "$MODE" == "2" ]; then
    "$PYTHON" "$SCRIPT_DIR/scripts/backtest_signal.py" --mode 2 --capital "$CAP"
    exit 0
fi

# ── Modus 1 ────────────────────────────────────────────────────────────────────
"$PYTHON" "$SCRIPT_DIR/scripts/backtest_signal.py" --mode 1 --capital "$CAP"
