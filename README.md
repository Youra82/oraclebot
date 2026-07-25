# oraclebot — 1%-Barriere-Vorhersage auf 4h-Kadenz

Statt eine ganze Tageskerze vorherzusagen, beantwortet oraclebot direkt die Frage, die für
eine gehebelte Position mit symmetrischem Stop-Loss/Take-Profit tatsächlich zählt: **wird der
Preis von hier aus zuerst +1% oder -1% erreichen?** Diese Frage wird für **jede 4h-Kerze**
neu gestellt (nicht nur einmal täglich) — ein `HistGradientBoostingClassifier` auf den
technischen Features der aktuellen 4h-Kerze **plus dem jeweils aktuellsten Kontext aus fünf
weiteren Zeitebenen** (1M/1w/1d/1h/15m) sagt die Richtung vorher, eine
Anti-Martingale-Positionsgrößen-Logik verwaltet den Einsatz.

> **Disclaimer:** Diese Software ist experimentell und dient ausschließlich Forschungszwecken.
> Der Handel mit Kryptowährungen (insb. mit hohem Hebel) birgt erhebliche finanzielle Risiken.
> Nutzung auf eigene Gefahr. Live-Order-Platzierung ist implementiert und über
> `barrier_strategy_settings.live_trading_enabled` in `settings.json` steuerbar (siehe
> [Live-Trading](#live-trading-echte-order-platzierung)) — **standardmäßig `false`**. Erst auf
> `true` stellen, wenn du das Verhalten verstanden und mit echtem, für dich verkraftbarem
> Kapital getestet hast.

---

## Grundidee

```
4h-Referenzkerze ──► 21 kontinuierliche Features (Trend, Struktur, Momentum,
  (OHLCV)             Volumen, S/R-Distanz, Trendkanal, ...)
                        │
1M/1w/1d/1h/15m ──► je Zeitebene dieselben 21 Features, jeweils die letzte VOR/BEI
  (Kontext)           der Referenzkerze abgeschlossene Kerze (kein Lookahead)
                        │
                        ▼      (6 x 21 = 126 Features aneinandergehaengt)
       HistGradientBoostingClassifier (max_depth=3)
                        │
                        ▼
      Vorhersage: "hoch zuerst" oder "runter zuerst"
        (wird ausgehend vom aktuellen Schlusskurs zuerst
         +1% oder -1% erreicht?) + Konfidenz
                        │
        Konfidenz >= min_confidence?
                        │
                        ▼
     Handelssignal: long (bei "hoch zuerst") oder short,
         SL/TP symmetrisch bei +-barrier_pct% vom Entry
                        │
                        ▼
       Anti-Martingale-Positionsgröße (% vom Guthaben,
       verdoppelt sich nach Gewinnen, Reset nach Verlust
       oder nach N Gewinnen in Folge)
```

**Warum dieser Ansatz und nicht ein Modell für die nächste Tageskerze?** Frühere Version des
Projekts sagten die nächste Tageskerze kategorial vorher (Trend/Range/Docht-Form) und leiteten
daraus SL/TP ab. Rückblickende Analyse (2026-07-24) zeigte: bei symmetrischem SL=TP=1% und
100x Hebel wird an den meisten Tagen ohnehin **beide** 1%-Marken berührt — die Tages-Richtung
ist die falsche Frage. Das direkte Barriere-Ziel, ausgewertet alle 4h statt einmal täglich,
lieferte im Vergleich (identisches Symbol, identischer Testzeitraum, identisches SL/TP):

| | Tages-Modell (verworfen) | 4h-Barriere, nur 4h-Features (Zwischenstand) | 4h-Barriere + Multi-Timeframe-Kontext (aktuell) |
|---|---|---|---|
| Trades im Testzeitraum | 95 | 696 | 684 |
| Out-of-Sample-Winrate | 67.4% | 79.9% | 80.6% |
| Walk-Forward (7-8 Fenster, 2.5 Jahre) | Mittel 59.0% / Worst-Case 56.0% | Mittel 67.6% / Worst-Case 63.8% | Mittel 74.6% / Worst-Case 71.4% |

Neun unabhängige Versuche, das alte Tages-Modell zu verbessern (Ensembles, Kalibrierung, mehr
Historie, Regime-Filter, asymmetrische SL/TP, ...) blieben erfolglos, bevor die
Barriere-Neuformulierung gefunden wurde — siehe Git-Historie für die Details der einzelnen
Experimente. Der Multi-Timeframe-Kontext (2026-07-25) wurde nachtraeglich ergaenzt, nachdem
ein Walk-Forward-Vergleich (reine 4h-Features vs. 4h+1h+1d vs. alle 6 Zeitebenen) einen
durchgaengigen Gewinn in jedem einzelnen Testfenster zeigte — anders als beim alten,
deutlich kleineren Tages-Datensatz ist der Barriere-Datensatz (~5900 statt ~535 Beispiele)
groß genug, dass zusätzliche Zeitebenen nicht bloß überanpassen.

---

## Architektur

```
oraclebot/
├── scripts/
│   ├── train_barrier_model.py     # Trainiert das Barriere-Modell + Walk-Forward-Robustheitscheck
│   └── predict_next_barrier.py    # Live-Inferenz + Trading auf 4h-Kadenz
├── install.sh                     # Erstinstallation auf VPS
├── update.sh                      # Git-Update (sichert secret.json)
├── settings.json                  # Konfiguration
├── secret.json                    # Bitget-API-Keys + Telegram Bot-Token/Chat-ID (nicht in Git)
│
└── src/oraclebot/
    ├── data/
    │   ├── features.py            # Kerze -> 21 Markt-Token-Features (kausal, kein Lookahead)
    │   ├── barrier_targets.py     # Barriere-Zieldefinition + Multi-Timeframe-Trainingsbeispiel-Bau
    │   ├── dataset.py             # JSON-Lines-Persistenz fuer Trainingsbeispiele
    │   └── scaler.py              # StandardScaler-Wrapper (Modell-Input-Normalisierung)
    │
    ├── model/
    │   └── barrier_model.py       # BarrierPredictor: HistGradientBoostingClassifier-Wrapper
    │
    ├── strategy/
    │   ├── barrier_signal.py      # Vorhersage -> Handelssignal (Entry/SL/TP)
    │   ├── signal.py              # compute_position_size (risikobasierte Positionsgroesse)
    │   ├── anti_martingale.py     # Alternative Positionsgroessen-Logik (Einsatz waechst nach Gewinnen)
    │   └── live_trade.py          # Order-Orchestrierung (Entry -> SL -> TP, Trigger-Order-Cleanup)
    │
    └── utils/
        ├── data_fetch.py          # Oeffentlicher OHLCV-Download (ccxt, Bitget)
        ├── exchange.py            # Authentifizierter Bitget-Wrapper (Live-Order-Platzierung)
        ├── barrier_gate.py        # 4h-Zeitfenster + Perioden-Marker (Doppel-Versand-Schutz)
        └── telegram.py            # send_message/send_photo

artifacts/
├── datasets/
│   ├── barrier_model_BTC_USDT_USDT_4h.pkl  # Trainiertes Modell (GIT-GETRACKT, ~120KB)
│   └── *.jsonl / ohlcv_*.pkl                # Trainingsdaten-Cache (NICHT in Git, jederzeit neu baubar)
└── state/                          # Live-Zustand der Anti-Martingale-Positionsgroesse (nicht in Git)
```

---

## Wie das System funktioniert

### 1. Markt-Tokenisierung (`features.py`)

Jede Kerze wird zu 21 kontinuierlichen Features (unverändert aus früheren Projektversionen,
funktioniert timeframe-unabhängig — dieselbe Funktion läuft für die Referenzkerze UND für
jede Kontext-Zeitebene):

| Feature | Bedeutung |
|---|---|
| `return`, `body`, `upper_wick`, `lower_wick` | Rohe Kerzengeometrie, normiert auf die Kerzenrange |
| `atr_range`, `trend_state`, `momentum`, `velocity` | Volatilitäts-/Trend-/Momentum-Zustand (ATR/EMA/RSI-basiert) |
| `structure` | Marktstruktur-Score (-2..+2) aus Swing-High/Low-Vergleich (HH/HL vs. LH/LL) |
| `higher_tf_position` | Position relativ zur EMA einer groeberen Referenz (Distanz-Feature) |
| `resistance_distance`, `support_distance` | Abstand zur nächsten Widerstands-/Unterstützungszone (geclusterte Swing-Punkte) |
| `channel_position`, `channel_slope` | Position/Steigung im lokalen Trendkanal (Regression durch Swing-Highs/-Lows) |
| `volume_ratio`, `macd_hist`, `gap` | Volumen relativ zum Schnitt, MACD-Momentum, Gap zum Vorkerzen-Close |
| `dow_sin/cos`, `month_start/end` | Zyklische Kalender-Features |

**No-Lookahead-Garantie:** Jedes Feature bei Kerze `t` darf ausschließlich Daten bis
einschließlich `t` verwenden — inklusive der Swing-High/Low-Erkennung, die einen eigenen
`confirmed_at`-Mechanismus nutzt, um nicht heimlich in die Zukunft zu schauen (siehe
Kommentare in `features.py`).

### 2. Barriere-Zielvariable (`barrier_targets.py`)

Für **jede 4h-Kerze** wird geprüft: wird ausgehend vom Schlusskurs zuerst eine Bewegung von
`+barrier_pct%` oder `-barrier_pct%` erreicht (anhand der feineren 15m-Kerzen, chronologisch
strikt nach der Referenzkerze)? Konservative Konvention bei Mehrdeutigkeit (beide Barrieren in
derselben 15m-Kerze): die untere (SL) gewinnt.

```python
BARRIER_LABELS = ['down_first', 'up_first']  # 0, 1
```

`build_barrier_examples()` verknüpft pro Referenzkerze den 21er-Feature-Block der 4h-Kerze mit
je einem weiteren 21er-Block pro `context_timeframes`-Eintrag (Standard: `1M`, `1w`, `1d`,
`1h`, `15m`) zu einem flachen 126er-Trainingsbeispiel. Jeder Kontext-Block ist die jeweils
letzte VOR/BEI der Referenzkerze abgeschlossene Kerze dieser Zeitebene, angebunden per
`pd.merge_asof(direction='backward')` — no-lookahead-korrekt, kein Blick in die Zukunft.
Anders als frühere Multi-Timeframe-Fenster-Ansätze (Sequenz über mehrere Kerzen pro Zeitebene)
reicht hier je Zeitebene nur die jeweils aktuellste Kerze, da ATR-/EMA-basierte Features
Historie bereits intern verarbeiten. Beispiele, bei denen ein Kontext-Timeframe noch keine
gültige Vorgänger-Kerze hat (früher Rand der Historie), werden übersprungen.

### 3. Modell (`barrier_model.py`)

Ein einzelnes `HistGradientBoostingClassifier` (`max_depth=3`) auf dem 126-dimensionalen
Feature-Vektor (Referenzkerze + 5 Kontext-Zeitebenen). `max_depth=3` wurde per Walk-Forward-Test
gegen 2/4/5/6 validiert (fast identische Genauigkeit über alle Tiefen, 3 als Mittelweg) — bei
nur ~5-6 Tausend Trainingsbeispielen ist ein tieferer Baum kein verlässlicher Gewinn. Ein
heterogenes Ensemble (HistGBM+RandomForest+LogReg) verbesserte in einem früheren Experiment die
rohe Accuracy, aber NICHT das Handelsergebnis — deshalb bewusst bei einem einzelnen Modell
belassen. Der Multi-Timeframe-Kontext selbst (mehr Zeitebenen statt ein größeres/komplexeres
Modell) war der Hebel, der die Genauigkeit tatsächlich robust verbesserte (siehe Vergleichstabelle
oben).

### 4. Handelssignal (`barrier_signal.py`)

```
Kein Trade, wenn Konfidenz < min_confidence
    │
Richtung = long, wenn "hoch zuerst" vorhergesagt, sonst short
Entry    = aktueller Schlusskurs der Referenzkerze
SL-Abstand = TP-Abstand = barrier_pct% vom Entry (immer symmetrisch)
```

### 5. Positionsgröße: zwei Optionen

- **`signal.py: compute_position_size()`** — klassisch risikobasiert: `(Guthaben ×
  risk_per_trade_pct%) / SL-Abstand`. Wird genutzt, wenn `anti_martingale_enabled=false`.
- **`anti_martingale.py`** — Einsatz als % vom *aktuellen* Guthaben, verdoppelt sich nach
  jedem Gewinn (bis `anti_martingale_streak_target` Gewinne in Folge erreicht sind, dann
  Reset auf die Basis), fällt nach jedem Verlust sofort auf die Basis zurück — das Gegenteil
  einer klassischen (verlust-eskalierenden) Martingale. Zustandsbehaftet (`artifacts/state/`),
  da der nächste Einsatz vom Ausgang der vorherigen Position abhängt. Da `Exchange` keine
  Order-Historie abfragen kann, wird der Ausgang der letzten Position indirekt aus dem
  Guthaben-Delta erschlossen (`resolve_pending_outcome()`).

Beide Wege laufen durch dieselbe `execute_live_trade()`-Funktion (`live_trade.py`) — inklusive
eines Sicherheitsmechanismus, der beim nächsten Lauf automatisch alle verwaisten Trigger-Orders
storniert (SL/TP sind zwei unabhängige Orders, kein OCO-Verbund — greift eine, bleibt die
andere sonst als Order-Leiche stehen).

---

## Handelssignal-Beispiel (Live-Lauf)

```
Referenzkerze: 2026-07-25 08:00:00+00:00 | Entry: 64030.00
Vorhersage: down_first (Konfidenz: 68.9%)
Signal: SHORT | SL: 64670.30 | TP: 63389.70
```

---

## Live-Trading (echte Order-Platzierung)

`predict_next_barrier.py` platziert bei `barrier_strategy_settings.live_trading_enabled: true`
echte Orders auf Bitget — separat von der Telegram-Benachrichtigung, die immer unabhängig
davon läuft. Implementiert in `src/oraclebot/utils/exchange.py` (authentifizierter
ccxt-Wrapper) und `src/oraclebot/strategy/live_trade.py` (Order-Orchestrierung).

```
Jeder Cronjob-Lauf im 4h-Zeitfenster (00/04/08/12/16/20 UTC, +0-29 Min), wenn live_trading_enabled=true:

  Offene Position fuer das Symbol vorhanden?
    ├── Ja  → Kein neuer Entry (kein Stacking). Nur loggen.
    └── Nein →
           Verwaiste Trigger-Orders der letzten (bereits geschlossenen) Position stornieren
           Anti-Martingale aktiv? → Ausgang der letzten Position aus Guthaben-Delta erschliessen,
               Einsatz-Prozentsatz entsprechend anpassen
           Kein Handelssignal (min_confidence nicht erreicht)?
             ├── Ja → nichts tun
             └── Nein →
                    Echtes Guthaben abrufen (fetch_balance_usdt)
                    Hebel + Margin-Modus setzen (leverage/margin_mode aus settings.json)
                    Positionsgroesse = Anti-Martingale ODER risikobasiert, gedeckelt
                        durch verfuegbare Margin (Balance x Hebel, 1% Puffer)
                    Boersen-Minimum + Mindest-Notional (5 USDT) pruefen
                        │
                        ▼
                    1) Market-Order Entry
                        │  (tatsaechlicher Fuellpreis/-menge aus der Order-Antwort)
                        ▼
                    2) SL als reduceOnly-Trigger-Order
                        │  Fehler? → Position SOFORT per Market Order schliessen
                        │            + Telegram-Alarm -- niemals ungeschuetzt offen lassen
                        ▼
                    3) TP als reduceOnly-Trigger-Order
                        │  Fehler? → nur loggen, Position bleibt durch SL geschuetzt
                        ▼
                    Telegram-Bestaetigung mit Entry/SL/TP/Menge/Hebel
```

**Sicherheitsprinzipien** (getestet in `tests/test_live_trade.py` + echtem Live-Smoke-Test auf
Bitget, siehe `tests/test_live_workflow.py`):
- Reihenfolge ist immer Entry → SL → TP, nie umgekehrt.
- SL-Platzierung schlägt fehl → Position wird sofort geschlossen.
- SL/TP werden am **tatsächlichen Fill-Preis** verankert.
- Margin-Cap verhindert, dass die Positionsgröße bei sehr enger SL-Distanz die verfügbare
  Margin übersteigt.
- Verwaiste Trigger-Orders werden bei jedem Lauf automatisch aufgeräumt.

### Einrichtung

```bash
cp secret.json.example secret.json
nano secret.json
```

```json
{
    "oraclebot": [
        { "name": "Main-Account", "apiKey": "...", "secret": "...", "password": "..." }
    ],
    "telegram": { "bot_token": "...", "chat_id": "..." }
}
```

```bash
# Erst manuell testen (live_trading_enabled noch false), dann umstellen:
nano settings.json   # "live_trading_enabled": true
```

Ohne `oraclebot`-Keys in `secret.json` bricht `predict_next_barrier.py` bei
`live_trading_enabled=true` kontrolliert mit einer klaren Fehlermeldung ab.

---

## Konfiguration (`settings.json`)

```json
{
    "barrier_strategy_settings": {
        "symbol": "BTC/USDT:USDT",
        "reference_timeframe": "4h",
        "intraday_timeframe": "15m",
        "context_timeframes": ["1M", "1w", "1d", "1h", "15m"],
        "barrier_pct": 1.0,
        "model_max_depth": 3,
        "min_confidence": 0.60,
        "leverage": 100,
        "margin_mode": "isolated",
        "risk_per_trade_pct": 2.0,
        "anti_martingale_enabled": false,
        "anti_martingale_base_pct": 5.27,
        "anti_martingale_growth_factor": 2.0,
        "anti_martingale_streak_target": 3,
        "live_trading_enabled": false,
        "history_days": 1000,
        "val_split": 0.30,
        "num_threads": 12,
        "feature_settings": { "atr_window": 14, "ema_window": 50, "...": "..." },
        "feature_settings_by_timeframe": { "1M": { "ema_window": 6, "...": "..." }, "1w": { "ema_window": 12, "...": "..." } }
    },
    "notification_settings": {
        "telegram_enabled": true,
        "telegram_send_chart": true
    }
}
```

| Parameter | Erklärung |
|---|---|
| `reference_timeframe` / `intraday_timeframe` | 4h für Signal-Entscheidungen, 15m für die Barriere-Reihenfolgen-Bestimmung. |
| `context_timeframes` | Zusätzliche Zeitebenen, deren letzte abgeschlossene Kerze als weiterer Feature-Block angehängt wird (siehe [Barriere-Zielvariable](#2-barriere-zielvariable-barrier_targetspy)). Leer = nur 4h-Features (altes Verhalten). |
| `barrier_pct` | Symmetrischer SL/TP-Abstand in % vom Entry. Validiert bei 1.0. |
| `model_max_depth` | HistGBM-Baumtiefe. 3 validiert (siehe [Modell](#3-modell-barrier_modelpy)). |
| `min_confidence` | 0.60 validiert (684 Trades, 80.6% Winrate im Testzeitraum). Höher = weniger, aber treffsicherere Trades. |
| `anti_martingale_*` | Siehe [Positionsgröße](#5-positionsgröße-zwei-optionen). `base_pct=5.17` haelt den 90.-Perzentil-MaxDD (Bootstrap über 3000 Neuordnungen der 684 Testtrades, nicht nur den einen historisch realisierten Pfad) bei ~50% — kalibriert für 15 USDT Startkapital, 100x Hebel (rekalibriert 2026-07-25). |
| `history_days` | Wie viel Historie beim Training geladen wird (1000 = ~Bitgets 1h/15m-Datentiefen-Grenze). |
| `feature_settings_by_timeframe` | Optionale Overrides je Kontext-Timeframe (z.B. kürzere Indikator-Fenster für `1M`/`1w`, die sonst Jahre an Warmup bräuchten). |
| `live_trading_enabled` | Schaltet echte Order-Platzierung ein/aus. Standard: `false`. |
| `notification_settings.telegram_enabled` | Unabhängig von `live_trading_enabled` — Prognosen kommen auch bei deaktiviertem Live-Trading per Telegram an. |

---

## Installation (lokal — Training)

```bash
git clone https://github.com/Youra82/oraclebot.git
cd oraclebot
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp secret.json.example secret.json && nano secret.json   # Telegram (optional)
```

---

## Workflow

#### 1. Modell trainieren

```bash
PYTHONPATH=src python scripts/train_barrier_model.py
```

Lädt (gecachte) OHLCV-Daten für `reference_timeframe` + `intraday_timeframe` +
`context_timeframes` (Standard: 4h, 15m, 1M, 1w, 1d, 1h), baut die Barriere-Trainingsbeispiele
mit Multi-Timeframe-Kontext, prüft die
Robustheit über 8 chronologische Walk-Forward-Fenster, trainiert das finale Modell auf dem
offiziellen 70/30-Split, speichert:
- `artifacts/datasets/barrier_model_BTC_USDT_USDT_4h.pkl` (**git-getrackt**, ~120KB)
- `artifacts/datasets/barrier_BTC_USDT_USDT_4h.jsonl` (Trainingsdaten-Cache, nicht in Git)
- `artifacts/datasets/barrier_diagnostics_BTC_USDT_USDT_4h.json` (Kennzahlen dieses Laufs)

`--no-cache` erzwingt einen frischen OHLCV-Download. `--history-days N` überschreibt die
Konfiguration (Debugging).

#### 2. Live-Prognose (manuell testen)

```bash
PYTHONPATH=src python scripts/predict_next_barrier.py --force
```

`--force` überspringt das 4h-Zeitfenster-Gate (für manuelles Testen zu beliebiger Uhrzeit,
markiert die Periode NICHT als erledigt). Holt Marktdaten über einen inkrementellen lokalen
Cache, lädt das trainierte Modell, gibt Vorhersage + Handelssignal aus, sendet bei
`telegram_enabled=true` zusätzlich eine Telegram-Nachricht.

---

## VPS-Deployment (automatische Prognose alle 4 Stunden)

#### 1. Installation

```bash
git clone https://github.com/Youra82/oraclebot.git && cd oraclebot
./install.sh
cp secret.json.example secret.json && nano secret.json
```

#### 2. Cronjob einrichten

```bash
crontab -e
```

```cron
*/15 * * * * /usr/bin/flock -n /pfad/zu/oraclebot/oraclebot.lock /bin/sh -c "sleep 60; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 cd /pfad/zu/oraclebot && /pfad/zu/oraclebot/.venv/bin/python3 scripts/predict_next_barrier.py >> /pfad/zu/oraclebot/logs/cron.log 2>&1"
```

`predict_next_barrier.py` läuft dadurch bis zu 96×/Tag, tut aber außerhalb der 4h-Grenzen
(00/04/08/12/16/20 UTC, jeweils die ersten 30 Minuten) absichtlich **nichts** — das
Zeitfenster-Gate (`barrier_gate.py`) prüft `pd.Timestamp.now(tz='UTC')` und beendet sich sofort
(Exit-Code 0), unabhängig von der Server-/cron-Zeitzonenkonfiguration (dasselbe robuste Muster
wie bei den anderen Bots — `CRON_TZ` erwies sich auf realen VPS als nicht verlässlich
unterstützt).

#### 3. Setup verifizieren

```bash
# 1) Automatisierte Tests
.venv/bin/python3 -m pytest tests/

# 2) Gate-Verhalten manuell verifizieren -- sollte SOFORT abbrechen
#    (ausser gerade in einem der sechs 4h-Fenster)
.venv/bin/python3 scripts/predict_next_barrier.py

# 3) Den echten Vorhersage-Pfad sofort pruefen, ohne auf die naechste 4h-Grenze zu warten
.venv/bin/python3 scripts/predict_next_barrier.py --force
```

#### 4. Update auf neue Version

```bash
./update.sh
```

Sichert `secret.json` vor `git reset --hard origin/main`. Da das Modell git-getrackt ist,
bringt das Update automatisch den neuesten trainierten Stand mit — kein Training auf dem VPS
nötig.

---

## Tägliche Verwaltung & wichtige Befehle

```bash
tail -f logs/cron.log                                              # Live mitverfolgen
grep -i "ERROR" logs/cron.log                                      # Nach Fehlern suchen
cd ~/oraclebot && .venv/bin/python3 scripts/predict_next_barrier.py --force   # Manueller Testlauf
PYTHONPATH=src python -m pytest tests/                             # Tests ausfuehren
./update.sh                                                        # Bot aktualisieren
```

#### Neu trainieren (nur auf der Trainings-Maschine, nicht auf dem VPS)

```bash
PYTHONPATH=src python scripts/train_barrier_model.py
git add artifacts/datasets/barrier_model_BTC_USDT_USDT_4h.pkl settings.json
git commit -m "Retrain: ..." && git push
# Danach auf dem VPS: ./update.sh
```

---

## Wichtige Regeln & bekannte Einschränkungen

- `secret.json` ist **nicht in Git** — wird von `update.sh` gesichert/wiederhergestellt.
- `artifacts/datasets/barrier_model_BTC_USDT_USDT_4h.pkl` ist **bewusst git-getrackt** —
  einzige Voraussetzung für Inferenz auf einem schwächeren Rechner.
- **Live-Order-Platzierung ist implementiert**, aber `live_trading_enabled` ist standardmäßig
  `false`. Auf `true` stellen bedeutet echtes Geld auf echten Bitget-Orders.
- Getrennte API-Keys für oraclebot empfohlen — nicht dieselben Keys wie andere Bots auf
  demselben Symbol wiederverwenden.
- Aktuell **BTC-only** — ein zweites, unabhängig getestetes Symbol (ETH) zeigte deutlich
  schwächere Signalqualität (52.6% statt 79.9% Winrate) und wurde verworfen.
- Backtest-PnL bei mehreren hundert Trades und Anti-Martingale-Compounding wird schnell
  astronomisch groß (reines Artefakt exponentiellen Compoundings über viele Trades, ignoriert
  reale Slippage-/Liquiditäts-Grenzen) — als **relativer** Vergleich zwischen Konfigurationen
  bei gleichem Ziel-Drawdown aussagekräftig, als absolute Zahl nicht.
- Externe Datenquellen (Funding Rate, Fear & Greed Index, DXY, On-Chain-Metriken,
  News-Sentiment) wurden in einer früheren Projektversion getestet und verworfen — keine
  zeigte einen robusten Effekt.

---

## Abhängigkeiten

```
ccxt==4.3.5      # Exchange-Verbindung (Bitget)
pandas==2.3.3    # Datenverarbeitung
ta==0.11.0       # Technische Indikatoren (ATR, EMA, RSI, MACD)
numpy==2.3.5     # Array-Operationen
scikit-learn==1.8.0  # HistGradientBoostingClassifier, StandardScaler
requests         # Telegram-API
pytest           # Tests
```
