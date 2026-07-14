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

deactivate
echo "--- Sicherheitscheck abgeschlossen ---"
exit $EXIT_CODE
