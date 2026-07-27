# src/oraclebot/utils/margin_safety.py
# Liefert die Hebel-Obergrenze, ab der Bitgets eigene Zwangsliquidation VOR unserem eigenen
# Stop-Loss greifen kann -- NUR als Kandidaten-Filter fuer optimize_barrier_model.py's
# Hebel-Suche gedacht (siehe LEVERAGE_CANDIDATES dort), NICHT fuer eine versteckte
# Laufzeit-Anpassung in live_trade.py/evaluation.py. Der Hebel, den der Optimizer findet und in
# die Coin/Timeframe-Strategie-Config schreibt, wird live UND im Backtest exakt so verwendet, wie
# er dort steht -- keine zusaetzliche, fuer den Nutzer unsichtbare Korrektur (Nutzer-Feedback
# 2026-07-27: "wir pfuschen nicht durch die gegend rum. es wird so live getradet wie es
# optimiert und gebacktestet wurde").
#
# Fund (2026-07-27): ein reales Live-Trade-Log zeigte eine "Long liquidation" (Bitgets eigene
# Zwangsschliessung, nicht unser SL-Trigger) statt des erwarteten SL-Treffers bei ~1% Abstand.
# Bitgets Liquidationspreis haengt vom tatsaechlich allozierten Margin-zu-Notional-Verhaeltnis ab
# (Liquidationspreis = [Margin+Offset-Groesse*Entry*Richtung] / [Groesse*(MMR+Gebuehr-Richtung)],
# siehe https://www.bitget.com/support/articles/12560603808759) -- bei sehr hohem Hebel (z.B.
# 100x, Margin = Notional/100) ist die Preisdistanz bis zur Liquidation rechnerisch KLEINER als
# der volle 1/Hebel-Abstand, weil Bitget schon bei Erreichen der Maintenance-Margin-Schwelle
# (nicht erst bei Margin=0) zwangsschliesst. Ein zu hoher Hebel kann also mit dem eigenen SL um
# dieselbe Preisbewegung "rennen" und verlieren.

# Maintenance Margin Rate fuer BTCUSDT-Perpetual, Positionswert 0-200.000 USDT (deckt die
# Positionsgroessen dieses Bots bei weitem ab). Quelle: Bitget-Ankuendigung zu Hebel-/Margin-Tier-
# Anpassungen fuer BTCUSDT/ETHUSDT (12560603834416, Stand 2026-07-27). Aendert sich Bitgets Tier-
# Struktur oder wird ein anderes Symbol/Positionswert-Tier relevant, muss dieser Wert aktualisiert
# werden.
BITGET_MMR_PCT = 0.40

# Nur ein Teil (nicht 100%) der theoretisch sicheren Hebelgrenze als Obergrenze fuer die
# Optimizer-Kandidaten zulassen -- Puffer fuer Unsicherheiten, die die vereinfachte Formel nicht
# abbildet (Bitgets undokumentierter "Offset"-Term, Funding-Gebuehren waehrend der Haltedauer,
# Slippage beim Market-Entry).
SAFETY_BUFFER = 0.7


def compute_max_safe_leverage(barrier_pct: float, taker_fee_pct: float, mmr_pct: float = BITGET_MMR_PCT,
                               safety_buffer: float = SAFETY_BUFFER) -> float:
    """Theoretische Hebel-Obergrenze, bei der Bitgets eigene Liquidation (basierend auf der
    Maintenance-Margin-Rate) garantiert erst NACH der konfigurierten SL-Distanz (barrier_pct)
    ausgeloest wird.

    Aus Bitgets Liquidationspreis-Formel (Long, isoliert, Offset=0) folgt fuer die relative
    Preisdistanz bis zur Liquidation: 1 - (1 - 1/Hebel) / (1 - MMR - Gebuehr). Aufgeloest nach dem
    Hebel, bei dem diese Distanz genau `barrier_pct` betraegt, ergibt die Obergrenze;
    `safety_buffer` reduziert sie zusaetzlich als Sicherheitsmarge.

    Nur zum FILTERN von Hebel-Kandidaten in optimize_barrier_model.py gedacht -- kein Aufrufer
    in live_trade.py/evaluation.py darf diese Funktion nutzen, um den konfigurierten Hebel zur
    Laufzeit zu veraendern."""
    mmr = mmr_pct / 100.0
    fee = taker_fee_pct / 100.0
    barrier = barrier_pct / 100.0
    return (1.0 / (1.0 - (1.0 - barrier) * (1.0 - mmr - fee))) * safety_buffer
