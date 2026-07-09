# oraclebot — Markt-Sprachmodell

Markt-Vorhersage als Sprachmodellierung: Kerzen werden zu "Markt-Token" (Feature-Vektoren)
kodiert, mehrere Timeframes gleichzeitig als Sequenz einem Transformer gefuettert, der die
Wahrscheinlichkeitsverteilung der naechsten Kerze (Trend/Range/Close-Position/Wicks) vorhersagt.

## Status
Aktuell implementiert: Datenpipeline (Schritt 1-4 der Spezifikation).
- `src/oraclebot/data/features.py` — Kerze -> Markt-Token-Feature-Vektor
- `src/oraclebot/data/targets.py` — naechste Kerze -> kategoriale Zielvariablen
- `src/oraclebot/data/dataset.py` — Multi-Timeframe-Trainingsbeispiele ohne Zukunftsdaten

Noch offen: Tokenisierung/Clustering zu einem Markt-Vokabular, Transformer-Modell
(Multi-Timeframe Attention, stufenweiser Decoder, Beam Search).

## Tests
```
pytest tests/
```
