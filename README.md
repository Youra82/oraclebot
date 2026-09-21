# oraclebot — Portfolio-Renko-Breakout-Strategie

## Grundidee

Portfolio-weite Renko-Breakout-Strategie auf sieben Altcoins (NEAR, DOT, SOL, ADA, AVAX, SUI,
XRP). Baut Entropy-Adaptive-Renko-Bricks (EAR, portiert aus `zerobot`) aus 5-Minuten-Kerzen —
Bricks entstehen preisbasiert, nicht zeitbasiert, Brick-Größe passt sich dynamisch an die aktuelle
Marktentropie an. Kein Vorhersage-Modell:

- **Entry**: ein Ausbruch aus einer Seitwärtsphase — `breakout_run` Bricks am Stück in dieselbe
  Richtung, direkt nach `horizontal_lookback` Bricks gemischter Richtung davor.
- **Exit**: der erste vollständig ausgebildete Gegen-Brick. Kein festes Take-Profit — die
  Brick-Struktur selbst definiert das Ende des Trades.
- **Portfolio-Arbitrierung**: nur EINE offene Position gleichzeitig über alle 7 Coins. Erstes
  frisches Signal gewinnt, alle anderen werden in derselben Runde verworfen (nicht nachgeholt).
- **Positionsgröße**: Anti-Martingale (Paroli) — Einsatz wächst nach Gewinnserien, fällt sofort
  auf die Basis zurück nach jedem Verlust.

Validiert über 70/30 In-Sample/Out-of-Sample-Split, mit realistischen Kosten (Taker-Gebühren,
Slippage, echte Bitget-Funding-Historie) und einer Liquidationsprüfung anhand der tatsächlichen
5m-Kursbewegung (nicht nur der Brick-Schlusskurse) gegen die echten, symbolspezifischen
Bitget-Wartungsmargen.

---

## Architektur

```
scripts/run_renko_breakout.py          Live-Cron (alle 5 Minuten)
src/oraclebot/
├── data/ear_bricks.py                 EAR-Brick-Konstruktion (Shannon-Entropie, inkrementell fortsetzbar)
├── strategy/
│   ├── horizontal_breakout_signal.py  Entry/Exit-Logik auf einer Brick-Kette
│   ├── renko_portfolio_state.py       Live-Zustand je Symbol (persistierte Brick-Kette, kein Neuaufbau)
│   ├── renko_live_trade.py            Order-Ausführung (Entry + Sicherheits-Stop, Exit)
│   └── anti_martingale.py             Positionsgrößen-Logik (Paroli)
├── analysis/
│   ├── renko_live_signal_check.py     Täglicher Live-vs-Backtest-Konsistenzcheck
│   └── renko_chart.py                 Interaktive Brick-Chart-Illustration (Plotly)
└── utils/
    ├── exchange.py                    Bitget-Wrapper (ccxt)
    ├── data_fetch.py                  OHLCV-Fetch inkl. Live-Cache
    ├── periodic_gate.py               Zeitfenster+Marker-Gate (fuer den taeglichen Konsistenzcheck)
    ├── telegram.py                    Benachrichtigungen
    └── config.py                      settings.json laden
```

### Warum die Brick-Kette live nicht einfach neu aufgebaut wird

Direkt aus einem dokumentierten `zerobot`-Vorfall übernommen: dort baute der Live-Betrieb die
EAR-Brick-Kette über ein rollierendes Fenster, der Backtest dagegen über eine durchgehende Kette
— beide wichen strukturell voneinander ab. `renko_portfolio_state.py` setzt die Kette stattdessen
inkrementell fort (persistierter Kerzen-Puffer + `precomputed_H_roll` für die Entropie-Glättung
an der Nahtstelle) — getestet gegen einen kompletten Neuaufbau (`test_incremental_batches_match_one_shot_build`).
Der tägliche Konsistenzcheck (`analysis/renko_live_signal_check.py`) baut zusätzlich regelmäßig
frisch nach und vergleicht gegen den Live-Zustand, als laufende Absicherung gegen genau diese
Fehlerklasse.

---

## Aktuelle Konfiguration (Stand 2026-09-21)

Siehe `_note`-Felder in `settings.json::renko_breakout_settings` für die volle Herleitung jeder
Zahl. Kurzfassung:

| Parameter | Wert | Begründung |
|---|---|---|
| Hebel | 20x | Bei 40x lag die reale Liquidationsdistanz (mit den echten, coin-spezifischen Bitget-Wartungsmargen statt einer BTC-Annahme) nur bei ~1.78-2.04% — zu nah am historisch schlechtesten beobachteten Kursausschlag (1.37%). 20x gibt ~3x Puffer. |
| Sicherheits-Stop | 3.0% | Backstop bei Prozessausfall (kein reguläres Exit-Mittel — der reguläre Exit ist der Gegen-Brick). Muss klar unter der Liquidationsdistanz liegen, sonst wirkungslos. |
| Einsatz-Basis (Anti-Martingale) | 2.0% | Vom User bewusst gewählter, aggressivster der 3 getesteten Kandidaten (0.5/1.0/2.0%). |
| `base_pct_brick` | pro Coin kalibriert | Einheitliche Brick-Größe ließ volatilere Coins (NEAR) ~2.75x so oft Bricks bilden wie ruhigere (XRP), was die Portfolio-Arbitrierung stark NEAR-lastig machte. Kalibriert auf ~65 Bricks/Tag je Coin (Bisektion nur auf In-Sample). Bekannte Grenze: die tatsächlich ARBITRIERTE Trade-Verteilung bleibt trotzdem leicht NEAR-lastig (korrelierte marktweite Vola-Schübe, nicht mehr über Brick-Größe lösbar). |

---

## Installation 🚀

```bash
git clone https://github.com/Youra82/oraclebot.git
cd oraclebot
chmod +x install.sh
bash ./install.sh
```

Erstellt die virtuelle Python-Umgebung, installiert `requirements.txt`, legt `logs/` an, macht
alle `.sh`-Skripte ausführbar.

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

`oraclebot`-Keys nur für echtes Live-Trading nötig (`renko_breakout_settings.enabled: true`) —
für Illustrationen/Charts reicht `telegram` (optional).

---

## VPS-Deployment (aktiv, `enabled: true`)

Läuft seit 2026-09-21 scharf. `renko_breakout_settings.enabled` ist der globale Kill-Switch.

#### Cronjob (alle 5 Minuten, Brick-Timeframe)

```cron
*/5 * * * * /usr/bin/flock -n /pfad/zu/oraclebot/oraclebot_renko.lock /bin/sh -c "sleep 20; OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 cd /pfad/zu/oraclebot && /pfad/zu/oraclebot/.venv/bin/python3 scripts/run_renko_breakout.py >> /pfad/zu/oraclebot/logs/cron_renko.log 2>&1"
```

#### Setup verifizieren / überwachen

```bash
.venv/bin/python3 -m pytest tests/                    # komplette Testsuite
.venv/bin/python3 scripts/run_renko_breakout.py --dry-run   # Brick-Ketten fortsetzen + Signale loggen, KEINE Orders
tail -f logs/cron_renko.log                            # live mitverfolgen
```

#### Manuell starten (ohne auf den nächsten Cron-Tick zu warten)

```bash
.venv/bin/python3 scripts/run_renko_breakout.py
```

Kein Zeitfenster-Gate — läuft bei jedem Aufruf sofort durch alle 7 Coins. Platziert bei
`enabled: true` echte Orders, falls gerade ein Signal ansteht. Der allererste Lauf ist ein "Cold
Start": pro Coin werden erst 3 Tage Warm-up-Historie geholt, bevor Live-Signale entstehen können.

#### Strategie pausieren (Kill-Switch)

`renko_breakout_settings.enabled` auf `false`, committen, pushen, `./update.sh` auf dem VPS. Der
Cron beendet sich dann bei jedem Tick sofort ohne Seiteneffekte. Eine bereits offene Position wird
dadurch NICHT automatisch geschlossen — ggf. manuell auf Bitget schließen.

#### Update auf neue Version

```bash
./update.sh
```

Sichert `secret.json` vor `git reset --hard origin/main`, stellt es danach wieder her.

---

## Illustration eines Trades ansehen

```bash
PYTHONPATH=src python3 -c "
from oraclebot.analysis.renko_chart import generate_renko_chart
from oraclebot.utils.data_fetch import fetch_all_timeframes

symbol = 'NEAR/USDT:USDT'
ohlcv = fetch_all_timeframes(symbol, ['5m'], 14, cache_dir='artifacts/datasets', use_cache=True)
generate_renko_chart(ohlcv['5m'], symbol, base_pct=0.00278, k_entropy=0.7, h_window=15,
                      horizontal_lookback=6, breakout_run=2, start_capital=100.0,
                      out_path='artifacts/charts/illustration.html')
"
```

Zeigt Bricks als Candlestick, die Seitwärtsphase vor jedem Entry grau schattiert, Entry/Exit-
Marker und eine illustrative Kapitalkurve (ohne Hebel/Gebühren — reine Nachvollziehbarkeit der
Signal-Logik, keine Performance-Zahl). `base_pct` je Coin siehe
`settings.json::renko_breakout_settings.base_pct_brick_by_symbol`.

---

## Tägliche Verwaltung & wichtige Befehle

```bash
tail -f logs/cron_renko.log                                         # Live mitverfolgen
grep -i "ERROR" logs/cron_renko.log                                 # Nach Fehlern suchen
crontab -l                                                           # Aktuellen Cronjob anzeigen
cat artifacts/state/renko_breakout_portfolio.json                   # Aktuell offene Position (falls vorhanden)
cat artifacts/state/renko_anti_martingale_state.json                # Aktueller Einsatz-Prozentsatz
PYTHONPATH=src python -m pytest tests/                              # Tests ausfuehren
./update.sh                                                         # Bot aktualisieren
```

#### `update.sh` fragt bei `git fetch`/`git reset --hard` nach Benutzername/Passwort

Passiert, wenn die Remote-URL des lokalen Repos auf HTTPS steht
(`https://github.com/Youra82/oraclebot.git`) statt auf SSH. Fix (im `oraclebot`-Verzeichnis auf
dem betroffenen Rechner/VPS):

```bash
git remote set-url origin git@github.com:Youra82/oraclebot.git
git fetch origin   # sollte jetzt ohne Passwort funktionieren
```

Fragt danach immer noch nach Passwort/Passphrase, kurz prüfen, ob der SSH-Key überhaupt aktiv ist:

```bash
ssh -T git@github.com
# Erwartete Ausgabe: "Hi Youra82! You've successfully authenticated..."
```

Kommt stattdessen ein Fehler, ist der SSH-Key auf diesem Rechner nicht im ssh-agent geladen oder
nicht bei GitHub hinterlegt.

---

## Wichtige Regeln & bekannte Einschränkungen

- **Hebel/Sicherheits-Stop müssen gegen die ECHTEN, symbolspezifischen Bitget-Wartungsmargen
  kalibriert werden, nicht gegen eine pauschale Annahme.** Fund 2026-09-21: eine erste
  Kalibrierung nutzte fälschlich BTCs Wartungsmarge (0.40%) für alle 7 Altcoins — die echten
  Sätze liegen bei 0.66% (NEAR/DOT/ADA/AVAX/SUI) bzw. 0.50% (SOL), abgefragt über
  `publicMixGetV2MixMarketQueryPositionLever`. Damit war die reale Liquidationsdistanz bei 40x
  deutlich enger als angenommen. Vor jeder künftigen Hebel-Änderung: echte Sätze neu abfragen,
  nicht die Werte aus dieser Tabelle als dauerhaft gültig annehmen (Bitget kann sie ändern).
- **Backtests ohne Liquidationsprüfung sind für gehebelte Renko-Strategien irreführend.** Ein
  Trade, der laut Brick-Logik als kleiner Verlust endet, kann auf dem Weg dahin einen größeren
  Kursausschlag gehabt haben, der bei hohem Hebel längst zur Liquidation geführt hätte. Jeder
  Realismus-Backtest hier prüft deshalb den maximalen adversen Ausschlag (MAE) anhand echter
  5m-High/Low-Kerzen gegen die Liquidationsdistanz, nicht nur den Brick-definierten Exit-Preis.
- **Portfolio-Arbitrierung ("ein Slot, erstes Signal gewinnt") verzerrt die Coin-Verteilung
  stärker, als reine Brick-Raten-Kalibrierung ausgleichen kann** — korrelierte, marktweite
  Volatilitätsschübe lassen einzelne Coins (hier: NEAR) trotz ausgeglichener Kandidaten-Signalrate
  überproportional oft gewinnen. Nur über eine andere Arbitrierungs-Regel lösbar (noch nicht
  umgesetzt).
- `secret.json` ist **nicht in Git** — wird von `update.sh` gesichert/wiederhergestellt.
- Backtest-PnL bei mehreren hundert Trades und Anti-Martingale-Compounding wird schnell
  astronomisch groß (reines Artefakt exponentiellen Compoundings über viele Trades) — als
  **relativer** Vergleich zwischen Konfigurationen aussagekräftig, als absolute Zahl nicht.
- Bitgets `fetch_funding_rate_history` ignoriert den `since`-Parameter vollständig — die
  Rohschnittstelle (`publicMixGetV2MixMarketHistoryFundRate` mit `pageNo`-Paginierung) liefert
  serverseitig maximal ~270 Datensätze (~89 Tage) zurück, unabhängig von der angefragten Tiefe.

---

## Abhängigkeiten

```
ccxt==4.3.5      # Exchange-Verbindung (Bitget)
pandas==2.3.3    # Datenverarbeitung
numpy==2.3.5     # Array-Operationen
requests         # Telegram-API
pytest           # Tests
plotly>=6.0.0    # Interaktive HTML-Charts
openpyxl>=3.1.0  # Excel-Export
```
