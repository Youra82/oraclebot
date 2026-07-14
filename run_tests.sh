#!/bin/bash
echo "--- Starte OracleBot-Sicherheitsnetz ---"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Fehler: Virtuelle Umgebung nicht gefunden. Bitte install.sh ausfuehren."
    exit 1
fi
source .venv/bin/activate

export PYTHONPATH=src

echo "Fuehre Pytest aus..."
if python3 -m pytest -v; then
    echo "Pytest erfolgreich durchgelaufen. Alle Tests bestanden."
    EXIT_CODE=0
else
    PYTEST_EXIT_CODE=$?
    if [ $PYTEST_EXIT_CODE -eq 5 ]; then
        echo "Pytest beendet: Keine Tests zum Ausfuehren gefunden."
        EXIT_CODE=0
    else
        echo "Pytest fehlgeschlagen (Exit Code: $PYTEST_EXIT_CODE)."
        EXIT_CODE=$PYTEST_EXIT_CODE
    fi
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "--- Live-Smoke-Test: Gate+Marker End-to-End (echter Datenabruf/Modell/Telegram!) ---"
    echo "Nutzt eine EIGENE Marker-Datei (smoke_test_marker.txt), NICHT die echte Produktions-"
    echo "Marker-Datei -- kann daher niemals den naechsten echten Mitternachts-Cronjob blockieren."
    SMOKE_MARKER="artifacts/datasets/smoke_test_marker.txt"
    rm -f "$SMOKE_MARKER"
    SIM_TIME="$(date -u +%Y-%m-%d) 00:05"

    echo ""
    echo "1. Lauf (sollte eine ECHTE Telegram-Nachricht senden, falls konfiguriert)..."
    if ! python3 scripts/predict_next_candle.py --simulate-now "$SIM_TIME" --marker-path "$SMOKE_MARKER"; then
        echo "Live-Smoke-Test 1. Lauf fehlgeschlagen."
        EXIT_CODE=1
    fi

    echo ""
    echo "2. Lauf, identischer simulierter Zeitpunkt (sollte uebersprungen werden, KEINE zweite Nachricht)..."
    SMOKE_OUTPUT="$(python3 scripts/predict_next_candle.py --simulate-now "$SIM_TIME" --marker-path "$SMOKE_MARKER" 2>&1)"
    echo "$SMOKE_OUTPUT"
    if echo "$SMOKE_OUTPUT" | grep -q "wurde bereits gesendet"; then
        echo "2. Lauf korrekt uebersprungen -- Doppel-Versand-Schutz bestaetigt."
    else
        echo "FEHLER: 2. Lauf haette wegen des Markers uebersprungen werden muessen, ist es aber nicht."
        EXIT_CODE=1
    fi

    rm -f "$SMOKE_MARKER"
fi

deactivate
echo ""
echo "--- Sicherheitscheck abgeschlossen ---"
exit $EXIT_CODE
