# oraclebot — Markt-Sprachmodell

Markt-Vorhersage als Sprachmodellierung: jede Kerze wird zu einem kontinuierlichen
**Markt-Token** (Feature-Vektor), mehrere Timeframes gleichzeitig einem **Transformer**
gefüttert, der stufenweise die Wahrscheinlichkeitsverteilung der nächsten Tageskerze
vorhersagt — analog zu GPT, das Wort für Wort vorhersagt, nur mit Kerzen-Eigenschaften
statt Wörtern.

> **Disclaimer:** Diese Software ist experimentell und dient ausschließlich Forschungszwecken.
> Der Handel mit Kryptowährungen birgt erhebliche finanzielle Risiken. Nutzung auf eigene Gefahr.
> Live-Order-Platzierung ist **nicht implementiert** — der Bot berechnet aktuell nur Signale
> und benachrichtigt per Telegram, platziert aber keine echten Trades (siehe [Wichtige Regeln](#wichtige-regeln--bekannte-einschränkungen)).

---

## Grundidee

```
Kerze (OHLCV) ──► Markt-Token (19 kontinuierliche Features: Trend, Struktur,
                   Momentum, Volumen, S/R-Distanz, Trendkanal, ...)
                        │
        6 Timeframes gleichzeitig (1M, 1w, 1d, 4h, 1h, 15m)
                        │
                        ▼
          Pro Timeframe ein eigener Transformer-Encoder
                        │
              Multi-Timeframe-Attention-Fusion
         ("welcher Timeframe ist gerade wichtig?")
                        │
                        ▼
     Stufenweiser Decoder: trend → range → close_position →
       upper_wick → lower_wick → gap_yn → inside_outside_day → high_first
                 (jeder Schritt sieht die vorherigen — wie
                  ein Sprachmodell, das Wort für Wort vorhersagt)
                        │
                        ▼
        Beam Search über alle 8 Zielvariablen gleichzeitig
                        │
                        ▼
         Rückübersetzung in konkrete Preis-Koordinaten
        (High/Body-oben/Body-unten/Low relativ zum Vortages-Close)
```

**Wichtige Korrektur (2026-07-10):** reine Transformer-Decoder-Köpfe kollabieren bei
schwach-signalhaltigen Zielgrößen zuverlässig auf "immer dieselbe Klasse" (Cross-Entropy +
Gradientenabstieg finden dort ein bequemes lokales Minimum). Ein `RandomForestClassifier`
auf denselben Features kollabiert strukturell nicht (Baum-Splits erzwingen echte
Unterscheidung) und ist bei den betroffenen Zielen sogar treffsicherer. Das System ist daher
ein **Hybrid**: 6 der 8 Zielgrößen kommen vom RandomForest-Ensemble, nur `gap_yn` und
`high_first` bleiben beim Transformer (siehe [Hybrid-Ansatz](#4-hybrid-ansatz-transformer-vs-randomforest)).

---

## Architektur

```
oraclebot/
├── scripts/
│   ├── train_transformer.py       # Haupt-Trainingslauf: Transformer + RandomForest-Ensemble
│   ├── predict_next_candle.py     # Live-Inferenz + Telegram-Benachrichtigung (--preview moeglich)
│   ├── backtest_signal.py         # Modus 1-3: Backtest / Portfolio / Auto-Optimierung
│   ├── interactive_chart.py       # Plotly-Chart: echte Kerzen + Prognose-Overlay
│   ├── measure_seed_reliability.py # Kollaps-Rate ueber mehrere Trainingslaeufe messen (Wilson-CI)
│   ├── fit_tokenizer.py           # (Legacy) K-Means-Vokabular-Exploration, kein Modell-Input mehr
│   └── build_sample_dataset.py    # (Legacy) Beispiel-Datensatz fuer fruehe Pipeline-Tests
├── install.sh                     # Erstinstallation auf VPS
├── update.sh                      # Git-Update (sichert secret.json)
├── show_results.sh                # Interaktives Menue: Backtest/Portfolio/Charts
├── settings.json                  # Konfiguration
├── secret.json                    # Telegram Bot-Token/Chat-ID (nicht in Git)
│
└── src/oraclebot/
    ├── data/
    │   ├── features.py            # Kerze -> 19 Markt-Token-Features (kausal, kein Lookahead)
    │   ├── targets.py             # Naechste Kerze -> 8 kategoriale Zielvariablen
    │   ├── dataset.py             # Multi-Timeframe-Trainingsbeispiele, No-Lookahead-Cutoff
    │   ├── scaler.py              # StandardScaler-Wrapper (Modell-Input-Normalisierung)
    │   └── tokenizer.py           # (Legacy) K-Means-Diskretisierung, nicht mehr Modell-Input
    │
    ├── model/
    │   ├── transformer.py         # MarketTransformer: Encoder + Fusion + Decoder + Beam Search
    │   ├── tree_ensemble.py       # RandomForest-Ensemble (Hybrid-Ansatz, TREE_TARGETS)
    │   ├── dataset_torch.py       # PyTorch Dataset/Collate fuer variable Timeframe-Fenster
    │   └── reconstruct.py         # Kategoriale Vorhersage -> konkrete Preis-Koordinaten
    │
    ├── strategy/
    │   └── signal.py              # Vorhersage -> Handelssignal (Entry/SL/TP/Positionsgroesse)
    │
    └── utils/
        ├── data_fetch.py          # Oeffentlicher OHLCV-Download (ccxt, Bitget)
        ├── telegram.py            # send_message/send_photo
        └── chart_png.py           # Statisches Vorhersage-Chart (matplotlib) fuer Telegram

artifacts/
├── datasets/
│   ├── market_transformer_best.pt # Trainierter Checkpoint (GIT-GETRACKT, VPS braucht kein Training)
│   ├── scaler_full.pkl            # Gefitteter FeatureScaler (GIT-GETRACKT)
│   ├── tree_ensemble.pkl          # RandomForest-Ensemble (GIT-GETRACKT)
│   └── *.jsonl / ohlcv_*.pkl      # Trainingsdaten-Cache (NICHT in Git, jederzeit neu baubar)
├── charts/                        # Generierte Chart-HTML/PNG (nicht in Git)
└── results/                       # seed_reliability*.json Studien (Kollaps-Rate-Messungen)
```

---

## Wie das System funktioniert

### 1. Markt-Tokenisierung (`features.py`)

Jede Kerze wird zu 19 kontinuierlichen Features:

| Feature | Bedeutung |
|---|---|
| `return`, `body`, `upper_wick`, `lower_wick` | Rohe Kerzengeometrie, normiert auf die Kerzenrange |
| `atr_range`, `trend_state`, `momentum`, `velocity` | Volatilitäts-/Trend-/Momentum-Zustand (ATR/EMA/RSI-basiert) |
| `structure` | Marktstruktur-Score (-2..+2) aus Swing-High/Low-Vergleich (HH/HL vs. LH/LL) |
| `resistance_distance`, `support_distance` | Abstand zur nächsten Widerstands-/Unterstützungszone (geclusterte Swing-Punkte) |
| `channel_position`, `channel_slope` | Position/Steigung im lokalen Trendkanal (Regression durch Swing-Highs/-Lows) |
| `volume_ratio`, `macd_hist`, `gap` | Volumen relativ zum Schnitt, MACD-Momentum, Gap zum Vortages-Close |
| `dow_sin/cos`, `month_start/end` | Zyklische Kalender-Features |

**No-Lookahead-Garantie (zentrales Designprinzip):** Jedes Feature bei Kerze `t` darf
ausschließlich Daten bis einschließlich `t` verwenden. Das ist nicht trivial bei
Swing-High/Low-Erkennung: ob Kerze `t` ein lokales Extremum ist, lässt sich naiv erst
beurteilen, wenn man die folgenden Kerzen kennt (`_compute_swings()` nutzt dafür
`confirmed_at` — ein Swing-Punkt zählt erst ab dem Zeitpunkt, ab dem er real bekannt
gewesen wäre, nicht ab seinem eigenen Zeitstempel). Ein zentraler Bug hier (centered
`rolling()`-Fenster ohne diese Korrektur) wurde am 2026-07-10 gefunden und behoben — die
letzten paar Kerzen vor jedem Trainings-Cutoff hatten zuvor Wissen über die Zukunft
enthalten. Regressionstest: `test_swing_features_are_causal_no_lookahead` in
`tests/test_features.py` (Feature bei `t` darf sich nicht ändern, wenn man Kerzen nach `t`
abschneidet).

### 2. Zielvariablen (`targets.py`)

Die nächste Kerze wird in 8 kategoriale Ziele zerlegt statt als rohe OHLC-Regression:

| Ziel | Klassen | Bedeutung |
|---|---|---|
| `trend` | bearish / bullish | Richtung (binär, keine Neutral-Zone — siehe unten) |
| `range` | 0-0.5 / 0.5-1 / 1-2 / >2 × ATR | Volle High-Low-Spanne relativ zur Volatilität |
| `close_position` | lower / middle / upper third | Wo der Close innerhalb der Kerzenrange liegt |
| `upper_wick`, `lower_wick` | small / medium / large | Docht-Anteil relativ zur Range |
| `gap_yn` | no_gap / gap | Öffnet die Kerze deutlich abseits vom Vortages-Close? |
| `inside_outside_day` | normal / inside / outside | Liegt die Kerze innerhalb/außerhalb der vorherigen? |
| `high_first` | low_first / high_first | Wurde Tagestief oder Tageshoch zuerst erreicht (aus 4h-Kerzen bestimmt) |

**Binär statt 3-Klassen bei `trend`:** ein sklearn-Baseline-Test zeigte mit einer expliziten
Neutral-Zone nur +4–7pp über Zufalls-Baseline, ohne Neutral-Zone (binär) +13.6pp (BTC) — die
Zone kostete an der Klassengrenze echtes Signal, statt es abzubilden.

### 3. Modell-Architektur (`transformer.py`)

Drei separate Bausteine mit unterschiedlicher Tiefe — kein einheitlicher "ein Transformer
macht alles". `d_model=256`, `nhead=8` (8×32-dim Heads), `dim_feedforward=1024` (Standard:
das Übliche 4×d_model-Verhältnis), `dropout=0.1`, Standard-PyTorch **Post-LN**
(`norm_first=False`, wie im Original-Paper Vaswani et al. 2017 — nicht Pre-LN wie bei GPT-2+),
ReLU-Aktivierung (PyTorch-Default).

#### 3a. Pro-Timeframe-Encoder (`TimeframeEncoder`) — 6× unabhängige Instanzen

```
  Feature-Fenster (seq_len_tf, 19)              z.B. 15m: 500 Kerzen x 19 Features
            │
            ▼
  nn.Linear(19 → 256)                           kontinuierliche Projektion, KEINE
            │                                   Diskretisierung (anders als der
            ▼                                   verworfene K-Means-Tokenizer)
  [CLS] + Feature-Sequenz  (seq_len_tf+1, 256)   gelerntes CLS-Token vorangestellt (BERT-Stil)
            │
            ▼
  + sinusförmige Positional Encoding             fest, nicht gelernt (Vaswani et al.),
            │                                    kein RoPE/ALiBi
            ▼
  ┌────────────────────────────────────────┐
  │  nn.TransformerEncoderLayer  × 4        │  Post-LN, pro Schicht:
  │                                         │
  │   x ──► Multi-Head Self-Attention ──►   │   8 Heads x 32 dim
  │           Add & LayerNorm ──►           │
  │           Feed-Forward 256→1024→256 ──► │   ReLU dazwischen
  │           Add & LayerNorm               │
  └────────────────────────────────────────┘
            │
            ▼
  CLS-Token-Zustand herausgreifen  (1, 256)      = Kontextvektor DIESES Timeframes
```

Läuft **6× komplett unabhängig** (eigene Gewichte je Timeframe: 1M/1w/1d/4h/1h/15m, kein
Weight-Sharing) — jede Zeitebene hat andere Dynamik, ein 15m-Encoder soll nicht dieselben
Muster suchen wie ein 1M-Encoder.

#### 3b. Multi-Timeframe-Fusion (`TimeframeFusion`) — genau 1×, nicht gestapelt

```
  6 Kontextvektoren (1M,1w,1d,4h,1h,15m)   je (1, 256)
            │
            ▼
  + gelernte Timeframe-Embeddings           Segment-Embedding-artig: "das hier ist
    (1 pro Timeframe, 256-dim)              der 1d-Kontext", "das hier ist 15m", ...
            │
            ▼
  [neues CLS] + 6 Timeframe-Tokens  (7, 256)
            │
            ▼
  1x nn.MultiheadAttention (Self-Attention, nhead=8)     -- NUR EINE Schicht, nicht gestapelt
            │
            ▼
  Add & LayerNorm (nur auf das CLS-Token)
            │
            ▼
  Fusionierter Kontext (1, 256)  +  Attention-Gewichte des CLS auf jeden Timeframe
                                     (= exakt die "Timeframe-Gewichte" im Live-Output)
```

#### 3c. Stufenweiser Decoder (`StepwiseDecoder`) — kein "echter" Transformer-Decoder

Wichtige Klarstellung: **keine** maskierte Self-Attention wie bei GPT. Acht sequentielle,
unabhängige MLP-Köpfe, Konditionierung auf vorherige Schritte rein **additiv** über
Klassen-Embeddings — kein Attention-Mechanismus im Decoder selbst:

```
  Fusionierter Kontext (1, 256)
       │
       ▼
  ┌─ Schritt 1: trend ────────────────────┐
  │  Linear(256→256) → ReLU → Dropout     │
  │       → Linear(256→2)  [Logits]       │
  │  gewählte Klasse → Embedding(256) ────┼──┐
  └────────────────────────────────────────┘  │  additiv zum Kontext
       Kontext + Klassen-Embedding  ◄──────────┘
       │
       ▼
  ┌─ Schritt 2: range ────────────────────┐   eigener Kopf, gleiche Struktur,
  │  ... (Kardinalität 4 statt 2)         │   sieht bereits die trend-Entscheidung
  └────────────────────────────────────────┘
       │
       ▼
      ... close_position → upper_wick → lower_wick →
          gap_yn → inside_outside_day → high_first
       │
       ▼
  8 Klassen-Vorhersagen, jede kennt alle vorherigen Schritte
```

Beim Training: Teacher Forcing (die echte Klasse wird als "gewählt" durchgereicht). Bei der
Inferenz: **Beam Search** (`predict_beam`, `beam_width=5`) sucht über alle 8 Schritte
gleichzeitig nach der wahrscheinlichsten Gesamtsequenz, statt pro Schritt gierig die
Einzel-Top-1-Klasse zu nehmen.

### 4. Hybrid-Ansatz: Transformer vs. RandomForest

```python
TREE_TARGETS = ['trend', 'range', 'close_position', 'upper_wick', 'lower_wick', 'inside_outside_day']
# Nur gap_yn und high_first bleiben beim Transformer-Decoder.
```

Grund: bei schwachem Signal findet Cross-Entropy + Gradientenabstieg zuverlässig das lokale
Minimum "immer dieselbe Klasse vorhersagen" — in einer n=15-Studie (Wilson-CI [54.8%, 93.0%])
kollabierte der Transformer bei ~80% der Trainingsläufe. `RandomForestClassifier` auf
denselben Features kollabiert strukturell nicht (Baum-Splits erzwingen echte Unterscheidung)
und war deterministisch reproduzierbar (132/132 identische Vorhersagen bei fixem
`random_state`). **Wichtige Lehre dabei:** Accuracy allein erkennt Kollaps nicht zuverlässig
— `range` sah mit 44–59% Accuracy brauchbar aus, war aber zu 131/132 Beispielen auf einen
einzigen Bucket kollabiert (die zweithäufigste echte Klasse). Nur ein direkter
Verteilungs-Check der Vorhersagen deckte das auf.

### 5. Preis-Rekonstruktion (`reconstruct.py`)

Zwei Rekonstruktionsfunktionen für zwei unterschiedliche Zwecke:

- **`reconstruct_candle()`** — nutzt alle 5 geometrischen Ziele (trend/range/close_position/wicks) für volle Docht-Geometrie. Täuscht dabei Präzision vor, die das Modell nicht hat — wird nur noch für Debug-Ausgaben verwendet, **nicht** für SL/TP oder Chart-Overlay.
- **`reconstruct_simple_candle()`** — nutzt nur `trend`+`range` (die einzigen mit tatsächlich validierter Vorhersagekraft), zeigt die gesamte Bewegung als Body ohne Docht. Wird für Chart-Overlay und Telegram-Charts verwendet.

Zwei getrennte Kalibrierungs-Konstanten, die leicht verwechselbar sind:
```python
RANGE_BUCKET_VALUES      = [0.25, 0.75, 1.5, 2.5]           # volle High-Low-Spanne (inkl. Docht) -- fuer signal.py SL/TP
RANGE_BUCKET_MOVE_VALUES = [0.0818, 0.2902, 0.638, 1.8113]  # NUR Open->Close-Bewegung -- fuer reconstruct_simple_candle()
```
Eine Verwechslung dieser beiden (Range als Body-Bewegung verwendet) führte am 2026-07-10 zu
2–4× zu großen Prognose-Kerzen im Chart-Overlay — durch einen direkten Größenvergleich mit
echten Kerzen entdeckt und behoben (Median-Größenverhältnis 2.24× → 0.90× vor dem
Lookahead-Fix, aktuell ~0.39× danach — siehe [Bekannte Einschränkungen](#wichtige-regeln--bekannte-einschränkungen)).

---

## Beispiel-Output (Live-Prognose)

```
Vorhersage fuer die Tageskerze am 2026-07-11:
  trend               : bullish (Kategorie 1)
  range               : 0-0.5atr (Kategorie 0)
  close_position      : upper_third (Kategorie 2)
  upper_wick          : medium (Kategorie 1)
  lower_wick          : large (Kategorie 2)
  gap_yn              : no_gap (Kategorie 0)
  inside_outside_day  : inside_day (Kategorie 1)
  high_first          : low_first (Kategorie 0)

  Timeframe-Gewichte: {'1M': 0.208, '1w': 0.156, '1d': 0.166, '4h': 0.186, '1h': 0.043, '15m': 0.183}

Rekonstruierte Preis-Koordinaten (Anker: Close 2026-07-10=63858.80, ATR=2158.53):
  High (Docht oben):    64155.60
  Body oben:            64020.69
  Body unten:           63858.80
  Low (Docht unten):    63615.97

Handelssignal (Trend-Konfidenz=52.3%):
  Richtung:     LONG
  Entry:        63858.80
  Stop-Loss:    63588.98  (Abstand 269.82)
  Take-Profit:  64263.52  (Abstand 404.72)
  Positionsgroesse bei 1000 USDT Beispiel-Balance: 0.037062 BTC
```

---

## Handelssignal & Position Sizing (`signal.py`)

```
Kein Trade, wenn Trend-Konfidenz < min_trend_confidence
    │
Richtung = long, wenn trend==bullish, sonst short
Entry    = letzter bekannter Schlusskurs (prev_close)
SL-Abstand = sl_range_fraction × RANGE_BUCKET_VALUES[range] × ATR
TP-Abstand = risk_reward × SL-Abstand
    │
Positionsgroesse = (Balance × risk_per_trade_pct%) / SL-Abstand
                    (risikobasiert, NICHT volles Kapital -- wie bei mbot/dnabot)
```

SL/TP hängen bewusst **nicht** an der vollen Docht-Geometrie, sondern nur an `trend` und
`range` — dieselben zwei Ziele, die als Einzige durchgehend über Zufalls-Baseline lagen.

**Ehrlicher Hinweis zu `min_trend_confidence`:** die "Konfidenz" ist die
RandomForest-Wahrscheinlichkeit der vorhergesagten Klasse (`argmax`-Wahrscheinlichkeit), die
bei einem binären Ziel immer ≥ 50% ist. Der aktuelle Schwellwert (`0.35`) filtert dadurch
faktisch **nichts** heraus — im letzten Mode-3-Backtest wurden alle 159/160 Beispiele
gehandelt (0 "Kein Trade"). Der Parameter ist als Filter vorbereitet, aber bei binärem `trend`
aktuell wirkungslos; für eine echte Konfidenz-Filterung müsste er gegen eine höhere Schwelle
wie 0.55–0.65 getestet werden (siehe `backtest_signal.py --mode 3`).

---

## Konfiguration (`settings.json`)

```json
{
    "dataset_settings": {
        "symbols": ["BTC/USDT:USDT"],
        "reference_timeframe": "1d",
        "window_sizes": { "1M": 12, "1w": 52, "1d": 200, "4h": 300, "1h": 400, "15m": 500 }
    },
    "model_settings": {
        "timeframes": ["1M", "1w", "1d", "4h", "1h", "15m"],
        "d_model": 256, "nhead": 8, "num_encoder_layers": 4,
        "dim_feedforward": 1024, "dropout": 0.1, "beam_width": 5
    },
    "training_settings": {
        "history_days": 1000,
        "batch_size": 16,
        "learning_rate": 0.0001,
        "epochs": 50,
        "val_split": 0.30,
        "early_stopping_patience": 12,
        "boosted_targets": ["trend"],
        "diversity_weight": 2.0
    },
    "strategy_settings": {
        "min_trend_confidence": 0.35,
        "sl_range_fraction": 0.5,
        "risk_reward": 1.5,
        "risk_per_trade_pct": 1.0,
        "live_trading_enabled": false
    },
    "notification_settings": {
        "telegram_enabled": true,
        "telegram_send_chart": true
    }
}
```

| Parameter | Erklärung |
|---|---|
| `dataset_settings.symbols` | Aktuell BTC-only. Multi-Symbol ist im Code vorbereitet (`dataset.py` iteriert pro Symbol), aber nicht getestet. |
| `dataset_settings.window_sizes` | Anzahl Kerzen im Eingabefenster je Timeframe (Sliding Window für den Encoder). |
| `training_settings.history_days` | 1000 = Obergrenze von Bitgets 1h/15m-Datentiefe (~1040 Tage). Mehr History half deutlich (Trend-Accuracy 53.8%→60.6%) — aber erst, nachdem der Hybrid-Ansatz das Transformer-Kollaps-Problem behoben hatte. |
| `training_settings.boosted_targets` / `diversity_weight` | Transformer-seitige Gegenmaßnahmen gegen Kollaps (stärkere Kopf-Initialisierung + Vorhersage-Streuungs-Regularisierer) — nur noch für `trend`/`gap_yn`/`high_first` relevant, da die anderen Ziele vom RandomForest kommen. |
| `strategy_settings.min_trend_confidence` | Siehe Hinweis oben — aktuell quasi wirkungslos bei binärem `trend`. |
| `strategy_settings.live_trading_enabled` | Schaltet **nur** eine Warnung in `predict_next_candle.py` — Order-Platzierung ist nicht implementiert, unabhängig vom Wert. |
| `notification_settings.telegram_enabled` | Unabhängig von `live_trading_enabled` — Prognosen kommen auch bei deaktiviertem Live-Trading per Telegram an. |
| `notification_settings.telegram_send_chart` | `true` = Chart-PNG mit Bildunterschrift; `false` = nur Text (`send_message`). |

---

## Installation (lokal — Training)

#### 1. Projekt klonen

```bash
git clone https://github.com/Youra82/oraclebot.git
cd oraclebot
```

#### 2. Abhängigkeiten installieren

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

#### 3. Telegram konfigurieren (optional, für Benachrichtigungen)

```bash
cp secret.json.example secret.json
nano secret.json
```

```json
{
    "telegram": {
        "bot_token": "DEIN_BOT_TOKEN",
        "chat_id": "DEINE_CHAT_ID"
    }
}
```

---

## Workflow

#### 1. Modell trainieren

```bash
PYTHONPATH=src python scripts/train_transformer.py
```

Lädt (gecachte) OHLCV-Daten für alle 6 Timeframes, trainiert den Transformer mit Early
Stopping, trainiert danach das RandomForest-Ensemble auf denselben Features, speichert:
- `artifacts/datasets/market_transformer_best.pt` (bester Checkpoint nach OOS-Genauigkeit)
- `artifacts/datasets/scaler_full.pkl`
- `artifacts/datasets/tree_ensemble.pkl`

Alle drei sind **git-getrackt** — kein erneutes Training auf dem VPS nötig, `update.sh`
bringt den trainierten Stand automatisch mit.

`--no-cache` erzwingt einen frischen OHLCV-Download statt der gecachten `.pkl`-Dateien.
`--max-examples N` begrenzt die Trainingsmenge (Debugging).

#### 2. Ergebnisse analysieren

```bash
./show_results.sh
```

| Modus | Funktion |
|---|---|
| **1) Einzel-Backtest** | Simuliert das Handelssignal auf allen Out-of-Sample-Beispielen (kombiniert über Symbole). |
| **2) Manuelle Portfolio-Simulation** | Du wählst Symbole aus, kombinierte Kapital-Simulation. |
| **3) Automatische Optimierung** | Grid-Search über `min_trend_confidence × risk_reward` (12 Kombinationen), wählt das beste Ergebnis innerhalb Max-Drawdown-Limit und Mindest-Trade-Anzahl (`--min-trades`, Standard 15 — verhindert, dass ein Ergebnis mit z.B. 2 Trades als "bestes" gewählt wird). |
| **4) Interaktive Charts** | Plotly-HTML: echte Kerzen + gestrichelte Prognose-Kerzen-Boxen überlagert, Titel zeigt Trend-Trefferquote. |

Direkter Aufruf ohne Menü:
```bash
PYTHONPATH=src python scripts/backtest_signal.py --mode 3 --capital 100 --max-dd 30 --min-trades 15
PYTHONPATH=src python scripts/interactive_chart.py --symbol "BTC/USDT:USDT" --last-days 160
```

#### 3. Beste Parameter übernehmen

Modus 3 zeigt das beste `min_trend_confidence`/`risk_reward`-Paar — manuell in
`strategy_settings` in `settings.json` eintragen.

#### 4. Live-Prognose (manuell testen)

```bash
PYTHONPATH=src python scripts/predict_next_candle.py
```

Holt frische Marktdaten (kein Cache), lädt den trainierten Checkpoint, gibt Vorhersage +
Preis-Koordinaten + Handelssignal aus, sendet bei `telegram_enabled=true` zusätzlich Chart +
Text per Telegram.

```bash
# Vorschau: behandelt die noch laufende Tageskerze als abgeschlossen,
# prognostiziert bereits jetzt die danach folgende Kerze (kein Warten auf 00:00 UTC)
PYTHONPATH=src python scripts/predict_next_candle.py --preview
```

---

## VPS-Deployment (tägliche automatische Prognose)

#### 1. Installation

```bash
git clone https://github.com/Youra82/oraclebot.git && cd oraclebot
./install.sh
cp secret.json.example secret.json && nano secret.json   # Telegram-Zugangsdaten
```

#### 2. Cronjob einrichten

Bitgets Tageskerze schließt um **00:00 UTC** — der Cronjob läuft kurz danach, damit die
Prognose garantiert die nächste (noch nicht begonnene) Kerze trifft, nicht die gerade
abgeschlossene:

```bash
crontab -e
```

```cron
TZ=UTC
5 0 * * * /usr/bin/flock -n /pfad/zu/oraclebot/oraclebot.lock /bin/sh -c "cd /pfad/zu/oraclebot && .venv/bin/python3 scripts/predict_next_candle.py >> /pfad/zu/oraclebot/logs/cron.log 2>&1"
```

`TZ=UTC` stellt sicher, dass die Zeitangabe UTC-basiert ausgewertet wird, unabhängig von der
Server-Lokalzeit. `flock` verhindert überlappende Läufe.

#### 3. Update auf neue Version

```bash
./update.sh
```

Sichert `secret.json` vor `git reset --hard origin/main`, aktualisiert Dependencies. Da
Modell-Checkpoint + Scaler + RandomForest-Ensemble git-getrackt sind, bringt das Update
automatisch den neuesten trainierten Stand mit — kein GPU/Training auf dem VPS nötig, reine
CPU-Inferenz (`torch.load(..., map_location='cpu')`).

---

## Tägliche Verwaltung & wichtige Befehle

#### Logs ansehen

```bash
tail -f logs/cron.log              # Live mitverfolgen
grep -i "ERROR" logs/cron.log      # Nach Fehlern suchen
tail -n 200 logs/cron.log          # Letzte 200 Zeilen
```

#### Manueller Prognose-Lauf (Test)

```bash
cd ~/oraclebot && .venv/bin/python3 scripts/predict_next_candle.py
```

#### Sofort-Vorschau ohne auf 00:00 UTC zu warten

```bash
cd ~/oraclebot && .venv/bin/python3 scripts/predict_next_candle.py --preview
```

#### Neu trainieren (nur auf der Trainings-Maschine, nicht auf dem VPS)

```bash
PYTHONPATH=src python scripts/train_transformer.py
git add artifacts/datasets/market_transformer_best.pt artifacts/datasets/scaler_full.pkl artifacts/datasets/tree_ensemble.pkl settings.json
git commit -m "Retrain: ..." && git push
# Danach auf dem VPS: ./update.sh
```

#### Kollaps-Stabilität messen

```bash
PYTHONPATH=src python scripts/measure_seed_reliability.py --n-runs 5 --tag mein_test
```

Trainiert mehrfach neu, prüft nach jedem Lauf die Vorhersage-Diversität (nicht nur Accuracy!)
und berechnet ein Wilson-Score-Konfidenzintervall der Kollaps-Rate.

#### Tests ausführen

```bash
PYTHONPATH=src python -m pytest tests/
```

69 Tests über Feature-Engineering (inkl. Kausalitäts-Regressionstest), Zielvariablen,
Rekonstruktion, Signal-Logik, Transformer-Architektur, RandomForest-Ensemble,
Telegram-Versand. Vor jedem Live-Deployment ausführen.

#### Bot aktualisieren (VPS)

```bash
./update.sh
```

---

## Wichtige Regeln & bekannte Einschränkungen

- `secret.json` ist **nicht in Git** — wird von `update.sh` gesichert/wiederhergestellt.
- `artifacts/datasets/market_transformer_best.pt`, `scaler_full.pkl`, `tree_ensemble.pkl`
  sind **bewusst git-getrackt** (anders als sonst übliche Binärdateien) — sie sind die
  einzige Voraussetzung für Inferenz auf einem schwächeren/GPU-losen Rechner.
- **Live-Order-Platzierung ist nicht implementiert.** `live_trading_enabled=true` löst nur
  eine Warnung aus, es wird nie eine echte Order gesendet. Der Bot ist aktuell ein
  Prognose-/Signal-System, kein automatisierter Trader.
- `min_trend_confidence` ist bei binärem `trend`-Ziel aktuell praktisch wirkungslos (siehe
  [Handelssignal](#handelssignal--position-sizing-signalpy)) — nicht mit einem echten Filter verwechseln.
- Aktuell **BTC-only** (`dataset_settings.symbols`). Multi-Symbol-Code existiert, ist aber
  nicht mit mehreren Symbolen getestet.
- Die vorhergesagte Kerzengröße ist nach dem Lookahead-Fix (2026-07-10) ehrlich, aber
  tendenziell zu klein (Median ~0.39× der echten Kerzengröße) — das Modell kann nicht mehr
  "vorausschauen", ob ein Tag ungewöhnlich volatil wird, und bleibt konservativ. Das ist
  erwartetes Verhalten, kein Bug.
- Backtest-PnL ist auf dem chronologischen 70/30-Split (ein fester ~160-Tage-OOS-Zeitraum),
  **kein Walk-Forward über mehrere Zeitfenster**. Ob die Performance in anderen Marktphasen
  (z.B. 2023) genauso hält, ist ungeprüft.
- Vier externe Datenquellen wurden getestet und verworfen (Funding Rate, Fear & Greed Index,
  Long/Short-Ratio, DXY) — keine zeigte einen robusten Effekt nach Split-Ratio-/Seed-Robustheitsprüfung.
- `fit_tokenizer.py`, `tokenizer.py`, `build_sample_dataset.py` sind Altlasten der
  ursprünglichen K-Means-Diskretisierungs-Idee — nicht Teil der aktuellen Produktions-Pipeline
  (der Transformer bekommt kontinuierliche Features direkt, siehe [Architektur](#3-modell-architektur-transformerpy)).

---

## Abhängigkeiten

```
ccxt==4.3.5      # Exchange-Verbindung (Bitget, oeffentliche OHLCV-Endpunkte)
pandas==2.1.3    # Datenverarbeitung
ta==0.11.0       # Technische Indikatoren (ATR, EMA, RSI, MACD)
numpy            # Array-Operationen
scikit-learn     # RandomForestClassifier, StandardScaler
torch            # Transformer-Modell
requests         # Telegram-API
matplotlib       # Statisches Vorhersage-Chart fuer Telegram
pytest           # Tests
```
