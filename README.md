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
│   ├── predict_next_barrier.py    # Live-Inferenz + Trading auf 4h-Kadenz (per Cron)
│   └── show_results.py            # Diagnose + Anti-Martingale-Backtest + Chart/Excel-Export
├── run_pipeline.sh                # Interaktiv: trainiert das Modell (ruft train_barrier_model.py)
├── show_results.sh                # Interaktiv: Zusammenfassung/Chart/Excel (ruft show_results.py)
├── push_configs.sh                # Modell + settings.json committen/pushen (mit Rebase-Retry)
├── install.sh                     # Erstinstallation auf VPS
├── update.sh                      # Git-Update (sichert secret.json)
├── run_tests.sh                   # Testsuite + Live-Smoke-Test (Gate+Marker)
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
│   ├── barrier_model_BTC_USDT_USDT_4h.pkl   # Trainiertes Modell (GIT-GETRACKT, ~120KB)
│   ├── barrier_diagnostics_*.json            # Diagnose des letzten Trainingslaufs (nicht in Git)
│   └── *.jsonl / ohlcv_*.pkl                 # Trainingsdaten-Cache (NICHT in Git, jederzeit neu baubar)
├── charts/                          # show_results.py --chart/--excel Ausgabe (nicht in Git)
│   ├── combined_overview.png        # Preis+Trades / Kapitalkurve / Serien-Chart
│   └── oraclebot_trades_*.xlsx      # Formatierter Trade-Log-Export (Excel)
└── state/                           # Live-Zustand der Anti-Martingale-Positionsgroesse (nicht in Git)
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

`secret.json` mit echten `oraclebot`-API-Keys wird bereits bei der [Installation](#installation-)
angelegt. Für echte Order-Platzierung zusätzlich:

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
        "anti_martingale_base_pct": 4.03,
        "anti_martingale_growth_factor": 2.0,
        "anti_martingale_streak_target": 3,
        "live_trading_enabled": false,
        "history_days": 1000,
        "val_split": 0.30,
        "backtest_start_capital": 15.0,
        "taker_fee_rate_pct": 0.06,
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
| `anti_martingale_*` | Siehe [Positionsgröße](#5-positionsgröße-zwei-optionen). `base_pct=4.03` haelt den 90.-Perzentil-MaxDD (Bootstrap über 3000 Neuordnungen der 684 Testtrades, **inklusive** Bitget-Taker-Gebühren) bei ~50% — kalibriert für 15 USDT Startkapital, 100x Hebel (rekalibriert 2026-07-26). Ohne Gebührenmodell hätte derselbe Bootstrap-Ansatz `base_pct=5.17` ergeben, was mit echten Gebühren aber ein 90.-Perzentil-MaxDD von ~60% statt ~50% bedeutet — Gebühren sind bei 100x Hebel keine Kleinigkeit (~12% der Brutto-PnL pro Trade). |
| `history_days` | Wie viel Historie beim Training geladen wird (1000 = ~Bitgets 1h/15m-Datentiefen-Grenze). |
| `backtest_start_capital` | Startkapital fürs `show_results.py`-Backtest (Anti-Martingale-Kapitalkurve/Chart/Excel-Export). Hat keinen Einfluss auf Live-Trading — dort zählt das echte Bitget-Guthaben. |
| `taker_fee_rate_pct` | Bitget-Taker-Gebühr pro Seite (Standard-Tier ohne VIP-Rabatt), fließt in `show_results.py`'s Backtest UND in die Anti-Martingale-Kalibrierung ein — bei 100x Hebel ~12% der Brutto-PnL pro Trade, keine Kleinigkeit (siehe [Einschränkungen](#wichtige-regeln--bekannte-einschränkungen)). |
| `feature_settings_by_timeframe` | Optionale Overrides je Kontext-Timeframe (z.B. kürzere Indikator-Fenster für `1M`/`1w`, die sonst Jahre an Warmup bräuchten). |
| `live_trading_enabled` | Schaltet echte Order-Platzierung ein/aus. Standard: `false`. |
| `notification_settings.telegram_enabled` | Unabhängig von `live_trading_enabled` — Prognosen kommen auch bei deaktiviertem Live-Trading per Telegram an. |

---

## Installation 🚀

Gilt sowohl lokal (Training, siehe [Workflow](#workflow)) als auch auf dem VPS
(Live-Deployment, siehe [VPS-Deployment](#vps-deployment-automatische-prognose-alle-4-stunden))
— dasselbe `install.sh` fuer beide.

#### 1. Projekt klonen

```bash
git clone https://github.com/Youra82/oraclebot.git
cd oraclebot
```

#### 2. Installations-Skript ausführen

```bash
chmod +x install.sh
bash ./install.sh
```

Das Skript erstellt die virtuelle Python-Umgebung, installiert alle Abhängigkeiten
(`requirements.txt`), legt `logs/` an und macht alle `.sh`-Skripte ausführbar.

#### 3. API-Keys eintragen

```bash
cp secret.json.example secret.json
nano secret.json
```

```json
{
    "oraclebot": [
        { "name": "Main-Account", "apiKey": "DEIN_API_KEY", "secret": "DEIN_SECRET", "password": "DEIN_PASSPHRASE" }
    ],
    "telegram": { "bot_token": "DEIN_BOT_TOKEN", "chat_id": "DEINE_CHAT_ID" }
}
```

`oraclebot`-Keys sind nur für [Live-Trading](#live-trading-echte-order-platzierung) nötig
(`live_trading_enabled: true`) — für reine Prognosen/Backtests reicht `telegram` (optional).
Ohne `oraclebot`-Keys bricht `predict_next_barrier.py` bei aktiviertem Live-Trading kontrolliert
mit einer klaren Fehlermeldung ab, statt undefiniert zu scheitern.

---

## Workflow

Wie bei den anderen Bots im Fleet: `./run_pipeline.sh` zum Trainieren, `./show_results.sh` zum
Ansehen der Ergebnisse. oraclebot ist aber deutlich einfacher als z.B. dnabot/zerobot — **ein**
fest konfiguriertes Symbol (BTC/USDT:USDT) mit einem festen Multi-Timeframe-Kontext, keine
Coin-/Timeframe-Auswahl, kein Genome-Discovery/Portfolio-Optimizer. Beide Skripte fragen daher
nur wenige, wirklich relevante Optionen ab.

#### 1. Modell trainieren

```bash
./run_pipeline.sh
```

Fragt optional `history_days`-Override und ob der OHLCV-Cache verworfen werden soll, ruft dann
`train_barrier_model.py` auf: lädt (gecachte) OHLCV-Daten für `reference_timeframe` +
`intraday_timeframe` + `context_timeframes` (Standard: 4h, 15m, 1M, 1w, 1d, 1h), baut die
Barriere-Trainingsbeispiele mit Multi-Timeframe-Kontext, prüft die Robustheit über 7-8
chronologische Walk-Forward-Fenster, trainiert das finale Modell auf dem offiziellen
70/30-Split, speichert:
- `artifacts/datasets/barrier_model_BTC_USDT_USDT_4h.pkl` (**git-getrackt**, ~120KB)
- `artifacts/datasets/barrier_BTC_USDT_USDT_4h.jsonl` (Trainingsdaten-Cache, nicht in Git)
- `artifacts/datasets/barrier_diagnostics_BTC_USDT_USDT_4h.json` (Kennzahlen dieses Laufs)

Zeigt am Ende automatisch die Zusammenfassung (wie `show_results.sh` Modus 1).

#### 2. Ergebnisse ansehen

```bash
./show_results.sh
```

Drei Modi:
1. **Zusammenfassung** — Trainings-Diagnose (Walk-Forward, In-Sample/Out-of-Sample) +
   vollständiger Anti-Martingale-Backtest inkl. Gebühren (siehe [Konfiguration](#konfiguration-settingsjson):
   `backtest_start_capital`/`taker_fee_rate_pct`), direkt in der Konsole.
2. **Chart aktualisieren** — `artifacts/charts/combined_overview.png` (Preis+Trades,
   Kapitalkurve, laufende Gewinn-/Verlust-Serien).
3. **Excel-Export** — `artifacts/charts/oraclebot_trades_<Zeitstempel>.xlsx`, optional gefiltert
   auf Trades ab einem Startdatum (Kapitalkurve startet dann frisch bei
   `backtest_start_capital`, nicht mit dem alten Compounding-Stand fortgesetzt).

Beide Skripte brauchen ein lokales `.venv` (`./install.sh`) und laufen NICHT auf dem VPS — das
Modell ist git-getrackt, der VPS braucht kein eigenes Training (siehe
[VPS-Deployment](#vps-deployment-automatische-prognose-alle-4-stunden)).

#### 3. Modell pushen

```bash
./push_configs.sh
```

Committet + pusht `artifacts/datasets/barrier_model_*.pkl` + `settings.json` (mit
automatischem Rebase-Retry bei Remote-Konflikten, wie bei den anderen Bots) — tut nichts, wenn
sich an beiden Dateien nichts geändert hat. Auf dem VPS danach `./update.sh`.

#### 4. Live-Prognose (manuell testen)

```bash
PYTHONPATH=src python scripts/predict_next_barrier.py --force
```

Kein interaktives `.sh`-Wrapper-Skript dafür (läuft normalerweise unbeaufsichtigt per Cron,
siehe unten) — `--force` überspringt das 4h-Zeitfenster-Gate (für manuelles Testen zu beliebiger
Uhrzeit, markiert die Periode NICHT als erledigt). Holt Marktdaten über einen inkrementellen
lokalen Cache, lädt das trainierte Modell, gibt Vorhersage + Handelssignal aus, sendet bei
`telegram_enabled=true` zusätzlich eine Telegram-Nachricht. **Achtung:** bei
`live_trading_enabled=true` platziert ein Signal ≥ `min_confidence` dabei eine ECHTE Order,
`--force` hebt nur das Zeitfenster-Gate auf, nicht das Live-Trading selbst.

---

## VPS-Deployment (automatische Prognose alle 4 Stunden)

#### 1. Installation

Wie unter [Installation](#installation-) beschrieben (`git clone` → `./install.sh` →
`secret.json` befüllen) — auf dem VPS zusätzlich mit den echten `oraclebot`-API-Keys, falls
`live_trading_enabled: true` genutzt werden soll.

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
tail -n 100 logs/cron.log                                          # Letzte Zeilen ansehen
tail -f logs/cron.log                                               # Live mitverfolgen
grep -i "ERROR" logs/cron.log                                       # Nach Fehlern suchen
crontab -l                                                          # Aktuellen Cronjob anzeigen
cd ~/oraclebot && .venv/bin/python3 scripts/predict_next_barrier.py --force   # Manueller Testlauf
PYTHONPATH=src python -m pytest tests/                              # Tests ausfuehren
./update.sh                                                         # Bot aktualisieren
./show_results.sh                                                   # Ergebnisse/Chart/Excel (nur lokal, nicht auf dem VPS)
```

#### Cronjob zeigt `FileNotFoundError` / veraltete Fehlermeldungen nach einem Update

Passiert, wenn der Cronjob noch auf einen alten, inzwischen umbenannten/gelöschten Skriptnamen
zeigt (z.B. `predict_next_candle.py` aus der alten Tages-Strategie, ersetzt durch
`predict_next_barrier.py` — genau dieser Fall trat 2026-07-26 auf einem VPS auf: `update.sh`
löschte die alte Datei, aber der Cronjob selbst wird von `update.sh` nicht angefasst). Symptom
in `logs/cron.log`: wiederholte `can't open file '.../predict_next_candle.py'`-Zeilen bei jedem
15-Minuten-Takt. Fix, ohne den ganzen Cronjob neu abzutippen (ersetzt nur den Skriptnamen,
Lock-Datei/Kommentar/Offset bleiben erhalten):

```bash
crontab -l | sed 's/predict_next_candle\.py/predict_next_barrier.py/' | crontab -
crontab -l   # zur Kontrolle: sollte jetzt predict_next_barrier.py zeigen
```

**Test, dass der Fix wirkt:** nach ca. 15-20 Minuten (ein Cron-Tick + 60s Offset)
`tail -n 20 logs/cron.log` prüfen — Erfolg ist eine Zeile mit **"Ausserhalb des
4h-Ausfuehrungsfensters..."** (aktuelle Formulierung aus `barrier_gate.py`), nicht mehr die
alte "Ausserhalb des taeglichen Ausfuehrungsfensters" oder ein `FileNotFoundError`. Den
vollständigen Lauf (mit echter Vorhersage + Telegram-Nachricht) gibt es dann beim nächsten
echten 4h-Fenster (00/04/08/12/16/20 UTC) zu sehen, am besten mit `tail -f logs/cron.log`
live mitverfolgt.

#### Neu trainieren (nur auf der Trainings-Maschine, nicht auf dem VPS)

```bash
./run_pipeline.sh
./push_configs.sh
# Danach auf dem VPS: ./update.sh
```

`push_configs.sh` (wie bei den anderen Bots) zeigt Trainings-Diagnose + relevante
`settings.json`-Werte des zu pushenden Standes an, committet `artifacts/datasets/barrier_model_*.pkl`
+ `settings.json` und pusht (mit automatischem Rebase-Retry bei Remote-Konflikten). Ohne
Änderungen an diesen beiden Dateien tut es nichts.

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
- **Gründliche Backtest-Analyse (2026-07-26) deckte zwei reale Risiken auf, die die 80.6%-
  Headline-Winrate relativiert:**
  - **Zeitliche Drift:** Winrate über den 9-Monats-Validierungszeitraum: erste Hälfte 82.2%,
    zweite Hälfte 78.9%, letzte 90 Tage 75.9%, letzte 30 Tage 72.9% — ein klarer Abwärtstrend
    genau bis zum Start des Live-Tests. Immer noch deutlich über der Gebühren-Breakeven-Grenze
    (56.0%, siehe unten), aber die realistische Erwartung fürs Live-Trading liegt eher bei
    ~73-76% als bei den über den ganzen Zeitraum gemittelten 80.6%.
  - **Überanpassungs-Lücke:** In-Sample-Winrate der tatsächlich gehandelten Signale 90.3% vs.
    Out-of-Sample 80.6% (9.7pp Lücke) — größer als die reine Klassifikations-Accuracy-Lücke aus
    dem Training (86.4%/76.8%), da die Signal-Filterung (min_confidence) den Effekt verstärkt.
  - Konfidenz-Kalibrierung ist dagegen sauber monoton (60-65%-Bucket: 62.1% realisierte
    Winrate; 95-100%-Bucket: 97.3%) und Verlust-Serien sind kurz gedeckelt (max. 3 in Folge,
    684 Trades) — die Grundarchitektur wirkt nicht kaputt, nur die Headline-Zahl optimistischer
    als der aktuelle Trend.
- **Trading-Gebühren fehlten bisher komplett im Backtest.** Bitget-Taker-Gebühr (Standard-Tier,
  0.06%/Seite) macht bei 100x Hebel ~12% der Brutto-PnL pro Trade aus — keine Kleinigkeit.
  Breakeven-Winrate MIT Gebühren (SL=TP=1%, 100x): **56.0%** statt 50% ohne Gebühren. Die
  zuletzt beobachtete reale Winrate (72.9%, letzte 30 Tage) liegt trotzdem mit 16.9pp Puffer
  komfortabel darüber. `anti_martingale_base_pct` ist seit 2026-07-26 gebührenbewusst
  kalibriert (siehe [Konfiguration](#konfiguration-settingsjson)).
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
