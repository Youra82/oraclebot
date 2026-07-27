#!/bin/bash
# push_configs.sh — trainiertes Modell + Coin/Timeframe-Config + settings.json committen/pushen
#
# Analog zu zerobot/dnabot's push_configs.sh: pusht die Config-Datei(en) im
# src/oraclebot/strategy/configs/-Verzeichnis. Anders als bei denen gibt es hier nur EIN
# Symbol/Modell (kein active_strategies-Array), daher immer genau eine Config-Datei + das
# git-getrackte Modell + settings.json (falls sich strukturelle Einstellungen geaendert haben).

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo ""
echo -e "${YELLOW}========== ORACLEBOT MODELL + CONFIG + SETTINGS PUSHEN ==========${NC}"
echo ""

SYMBOL=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['symbol'])" 2>/dev/null)
REFERENCE_TF=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['reference_timeframe'])" 2>/dev/null)
SAFE_SYMBOL=$(echo "$SYMBOL" | tr '/:' '__')
MODEL_PATH="artifacts/datasets/barrier_model_${SAFE_SYMBOL}_${REFERENCE_TF}.pkl"
DIAGNOSTICS_PATH="artifacts/datasets/barrier_diagnostics_${SAFE_SYMBOL}_${REFERENCE_TF}.json"
CONFIG_PATH="src/oraclebot/strategy/configs/config_${SAFE_SYMBOL}_${REFERENCE_TF}.json"

if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Kein trainiertes Modell gefunden: $MODEL_PATH${NC}"
    echo "   Erst ./run_pipeline.sh ausfuehren."
    exit 1
fi

# Trainings-Diagnose des zu pushenden Modells anzeigen (falls vorhanden)
if [ -f "$DIAGNOSTICS_PATH" ]; then
    echo "Trainings-Diagnose ($DIAGNOSTICS_PATH):"
    python3 -c "
import json
with open('$DIAGNOSTICS_PATH') as f:
    d = json.load(f)
print(f\"  Beispiele: {d['n_examples']} (Train={d['n_train']} Val={d['n_val']})\")
print(f\"  70/30-Split: In-Sample={d['train_accuracy']:.1%} Out-of-Sample={d['val_accuracy']:.1%}\")
print(f\"  Walk-Forward: Mittel={d['walk_forward_mean']:.1%} Worst-Case={d['walk_forward_worst_case']:.1%}\")
" 2>/dev/null || echo "  (Konnte Diagnose nicht parsen)"
else
    echo -e "${YELLOW}ℹ  Keine Diagnose-Datei gefunden ($DIAGNOSTICS_PATH) -- nicht git-getrackt, wird bei jedem Training neu erzeugt.${NC}"
fi
echo ""

# Strategie-Config anzeigen (vom Optimizer gefundene Parameter -- siehe
# src/oraclebot/utils/config.py fuer die Trennung Config-Datei/settings.json)
if [ -f "$CONFIG_PATH" ]; then
    echo "Strategie-Config ($CONFIG_PATH):"
    python3 -c "
import json
with open('$CONFIG_PATH') as f:
    c = json.load(f)
print(f\"  Hebel={c.get('leverage')}x min_confidence={c.get('min_confidence')} model_max_depth={c.get('model_max_depth')}\")
print(f\"  Anti-Martingale: base_pct={c.get('anti_martingale_base_pct')}% growth={c.get('anti_martingale_growth_factor')}x streak={c.get('anti_martingale_streak_target')}\")
" 2>/dev/null || echo "  (Konnte Config nicht parsen)"
else
    echo -e "${RED}❌ Keine Strategie-Config gefunden: $CONFIG_PATH${NC}"
    echo "   Ungewoehnlich -- sollte spaetestens seit dem Umstieg auf das Coin/Timeframe-Config-Muster existieren."
fi
echo ""

# Strukturelle settings.json-Werte anzeigen
echo "settings.json (barrier_strategy_settings, strukturell):"
python3 -c "
import json
with open('settings.json') as f:
    s = json.load(f)['barrier_strategy_settings']
print(f\"  Symbol: {s['symbol']} ({s['reference_timeframe']})\")
print(f\"  Anti-Martingale aktiv: {s['anti_martingale_enabled']} | Live-Trading: {s['live_trading_enabled']}\")
" 2>/dev/null || echo "  (Konnte settings.json nicht parsen)"
echo ""

# Änderungen stagen
git add "$MODEL_PATH" "$CONFIG_PATH" settings.json
STAGED=$(git diff --cached --name-only)

# "Keine neuen Aenderungen zum Committen" bedeutet NICHT automatisch "nichts zu pushen" -- ein
# vorheriger Lauf (oder ein zuvor fehlgeschlagener Push) kann bereits einen lokalen Commit
# hinterlassen haben, der noch nicht auf origin liegt (Bug gefunden 2026-07-27: genau das blieb
# unbemerkt lokal auf einem VPS haengen, das Skript meldete faelschlich "bereits aktuell").
git fetch origin main --quiet 2>/dev/null
UNPUSHED=$(git log origin/main..HEAD --oneline 2>/dev/null | wc -l)

if [ -z "$STAGED" ] && [ "$UNPUSHED" -eq 0 ]; then
    echo -e "${YELLOW}ℹ  Keine Änderungen — Modell/Config/settings.json bereits aktuell im Repo und gepusht.${NC}"
    exit 0
fi

if [ -n "$STAGED" ]; then
    echo "Geänderte Dateien:"
    git diff --cached --name-only | sed 's/^/  /'
    echo ""

    # Commit
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
    git commit -m "Retrain: update model + config + settings ($TIMESTAMP)"
else
    echo -e "${YELLOW}ℹ  Keine neuen Änderungen, aber $UNPUSHED lokale(r) Commit(s) noch nicht auf origin -- pushe trotzdem.${NC}"
fi

# Push (mit Rebase-Retry bei Konflikt, wie bei den anderen Bots)
echo ""
echo -e "${YELLOW}Pushe auf origin/main...${NC}"
git push origin HEAD:main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Modell + Config + settings.json erfolgreich gepusht!${NC}"
    echo "   Auf dem VPS jetzt: ./update.sh"
else
    echo ""
    echo -e "${YELLOW}Remote hat neuere Commits — führe Rebase durch...${NC}"
    git pull origin main --rebase
    if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Rebase fehlgeschlagen. Bitte manuell lösen.${NC}"
        exit 1
    fi
    git push origin HEAD:main
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✅ Modell + Config + settings.json erfolgreich gepusht!${NC}"
        echo "   Auf dem VPS jetzt: ./update.sh"
    else
        echo -e "${RED}❌ Push nach Rebase fehlgeschlagen.${NC}"
        exit 1
    fi
fi
