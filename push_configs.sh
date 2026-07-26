#!/bin/bash
# push_configs.sh — trainiertes Modell + settings.json committen und pushen
#
# Analog zu zerobot/dnabot's push_configs.sh, aber angepasst an oraclebot: kein
# "configs/"-Verzeichnis mit mehreren Symbol/Timeframe-Configs (nur EIN Symbol/Modell), daher
# werden hier direkt die zwei Artefakte gepusht, die ein ./run_pipeline.sh-Lauf veraendert:
# artifacts/datasets/barrier_model_BTC_USDT_USDT_4h.pkl (git-getrackt) + settings.json.

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
cd "$SCRIPT_DIR"

echo ""
echo -e "${YELLOW}========== ORACLEBOT MODELL + SETTINGS PUSHEN ==========${NC}"
echo ""

SYMBOL=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['symbol'])" 2>/dev/null)
REFERENCE_TF=$(python3 -c "import json; print(json.load(open('settings.json'))['barrier_strategy_settings']['reference_timeframe'])" 2>/dev/null)
SAFE_SYMBOL=$(echo "$SYMBOL" | tr '/:' '__')
MODEL_PATH="artifacts/datasets/barrier_model_${SAFE_SYMBOL}_${REFERENCE_TF}.pkl"
DIAGNOSTICS_PATH="artifacts/datasets/barrier_diagnostics_${SAFE_SYMBOL}_${REFERENCE_TF}.json"

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

# Relevante settings.json-Werte anzeigen
echo "settings.json (barrier_strategy_settings):"
python3 -c "
import json
with open('settings.json') as f:
    s = json.load(f)['barrier_strategy_settings']
print(f\"  Symbol: {s['symbol']} ({s['reference_timeframe']}) | min_confidence={s['min_confidence']}\")
print(f\"  Anti-Martingale: enabled={s['anti_martingale_enabled']} base_pct={s['anti_martingale_base_pct']}%\")
print(f\"  Live-Trading: {s['live_trading_enabled']}\")
" 2>/dev/null || echo "  (Konnte settings.json nicht parsen)"
echo ""

# Änderungen stagen
git add "$MODEL_PATH" settings.json
STAGED=$(git diff --cached --name-only)

if [ -z "$STAGED" ]; then
    echo -e "${YELLOW}ℹ  Keine Änderungen — Modell + settings.json bereits aktuell im Repo.${NC}"
    exit 0
fi

echo "Geänderte Dateien:"
git diff --cached --name-only | sed 's/^/  /'
echo ""

# Commit
TIMESTAMP=$(date '+%Y-%m-%d %H:%M')
git commit -m "Retrain: update model + settings ($TIMESTAMP)"

# Push (mit Rebase-Retry bei Konflikt, wie bei den anderen Bots)
echo ""
echo -e "${YELLOW}Pushe auf origin/main...${NC}"
git push origin HEAD:main

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Modell + settings.json erfolgreich gepusht!${NC}"
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
        echo -e "${GREEN}✅ Modell + settings.json erfolgreich gepusht!${NC}"
        echo "   Auf dem VPS jetzt: ./update.sh"
    else
        echo -e "${RED}❌ Push nach Rebase fehlgeschlagen.${NC}"
        exit 1
    fi
fi
