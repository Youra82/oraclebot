# src/oraclebot/data/ear_bricks.py
# Entropy-Adaptive-Renko-Brick-Konstruktion -- portiert aus zerobot/src/zerobot/strategy/
# ear_engine.py (dort seit Juni 2026 live, siehe [[research_zerobot_live_vs_backtest_2026_08]]
# fuer die dortige Live-vs-Backtest-Absicherung). Reine Datenstruktur-Funktion, KEIN Signal --
# das neue Ausbruch-aus-Seitwaertsphase-Signal (strategy/horizontal_breakout_signal.py) baut
# darauf auf, verwendet aber eine andere Signallogik als zerobots eigener Entropy-Squeeze.
#
# Renko-Grundprinzip: Bricks entstehen NICHT zeitbasiert (anders als OHLCV-Kerzen), sondern
# preisbasiert -- ein neuer Brick bildet sich erst, wenn der Preis sich um mindestens `brick_size`
# vom letzten Brick-Schluss bewegt hat. Ein Richtungswechsel braucht das DOPPELTE der Brick-Groesse
# (Standard-Renko-Konvention) -- das macht Renko von Natur aus resistent gegen kleines Rauschen.
# Brick-Groesse ist adaptiv: klein im ruhigen Trend, gross im chaotischen Markt (Shannon-Entropie
# der Kerzengeometrie als Chaos-Mass).
import numpy as np
import pandas as pd


def candle_entropy(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Shannon-Entropie der Kerzengeometrie (Position des Schlusses in der Kerzenrange als
    Wahrscheinlichkeitsverteilung ueber 'Richtung zum Hoch' vs. 'Richtung zum Tief') -- H->0 bei
    einer klar gerichteten Kerze (Schluss nahe einem Extrem), H->1 bei einer unentschlossenen
    Kerze (Schluss mittig). Vektorisiert, identisch zu zerobots skalarer Referenzimplementierung."""
    hl = high - low
    eps = 1e-10
    safe_hl = np.where(hl < 1e-12, 1.0, hl)
    pb = np.clip((close - low) / safe_hl, eps, 1 - eps)
    ps = np.clip((high - close) / safe_hl, eps, 1 - eps)
    s = pb + ps
    pb = pb / s
    ps = ps / s
    ent = -pb * np.log2(pb) - ps * np.log2(ps)
    return np.where(hl < 1e-12, 1.0, ent)


def build_ear_bricks(df: pd.DataFrame, base_pct: float = 0.004, k_entropy: float = 0.7,
                      h_window: int = 15, init_close: float = None, init_direction: str = None,
                      precomputed_H_roll=None) -> list:
    """Baut Entropy-Adaptive-Renko-Bricks aus einem OHLCV-DataFrame (DatetimeIndex, aufsteigend).

    Args:
        base_pct: Basis-Brick-Groesse als Anteil des letzten Brick-Schlusses (0.004 = 0.4%).
        k_entropy: wie stark die geglaettete Entropie die Brick-Groesse vergroessert
            (brick_size = letzter_close * base_pct * (1 + k_entropy * H_geglaettet)).
        h_window: Glaettungsfenster (rollierender Mittelwert) fuer die Entropie.
        init_close/init_direction: optionaler persistierter Zustand fuer eine inkrementelle
            Fortsetzung einer bereits laufenden Kette (Live-Betrieb) -- ohne Angabe startet die
            Kette beim ersten Schlusskurs (pfadabhaengig, aber deterministisch reproduzierbar).
        precomputed_H_roll: optionales, vorberechnetes geglaettetes Entropie-Array (gleiche
            Laenge wie `df`) -- fuer eine inkrementelle Fortsetzung, bei der `df` NUR die neuen
            Kerzen enthaelt (nicht auch schon verarbeitete Puffer-Kerzen): das rollierende Mittel
            braucht sonst an der Nahtstelle `h_window` Kerzen Kontext VOR der ersten neuen Kerze,
            die man aber nicht nochmal durch die Brick-Konstruktion laufen lassen darf (das
            wuerde bereits erzeugte Bricks doppelt bauen). Dasselbe Muster wie in zerobots
            ear_engine.py. Ohne dieses Argument wird H_roll aus `df` selbst berechnet (korrekt
            fuer einen durchgaengigen Voll-Aufbau, wie er im Backtest immer vorliegt).

    Returns:
        Liste von Dicts {candle_idx, ts (Zeitstempel der ausloesenden Kerze), direction
        ('up'/'down'), close (Brick-Schluss), H}, chronologisch.
    """
    n = len(df)
    min_n = 1 if init_close is not None else 2
    if n < min_n:
        return []

    closes = df['close'].to_numpy()
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    opens = df['open'].to_numpy()
    index = df.index

    if precomputed_H_roll is not None:
        H_roll = np.asarray(precomputed_H_roll)
    else:
        H_raw = candle_entropy(opens, highs, lows, closes)
        H_roll = pd.Series(H_raw).rolling(h_window, min_periods=1).mean().to_numpy()

    bricks = []
    lc = init_close if init_close is not None else closes[0]
    direction = init_direction
    start_i = 0 if init_close is not None else 1

    for i in range(start_i, n):
        H = float(H_roll[i])
        bs = lc * base_pct * (1.0 + k_entropy * H)
        price = closes[i]

        if direction is None:
            if price >= lc + bs:
                direction = 'up'
            elif price <= lc - bs:
                direction = 'down'
            else:
                continue

        if direction == 'up':
            while price >= lc + bs:
                nc = lc + bs
                bricks.append({'candle_idx': i, 'ts': index[i], 'direction': 'up', 'close': nc, 'H': H})
                lc = nc
                bs = lc * base_pct * (1.0 + k_entropy * H)
            if price <= lc - 2 * bs:
                direction = 'down'
                while price <= lc - bs:
                    nc = lc - bs
                    bricks.append({'candle_idx': i, 'ts': index[i], 'direction': 'down', 'close': nc, 'H': H})
                    lc = nc
                    bs = lc * base_pct * (1.0 + k_entropy * H)
        else:
            while price <= lc - bs:
                nc = lc - bs
                bricks.append({'candle_idx': i, 'ts': index[i], 'direction': 'down', 'close': nc, 'H': H})
                lc = nc
                bs = lc * base_pct * (1.0 + k_entropy * H)
            if price >= lc + 2 * bs:
                direction = 'up'
                while price >= lc + bs:
                    nc = lc + bs
                    bricks.append({'candle_idx': i, 'ts': index[i], 'direction': 'up', 'close': nc, 'H': H})
                    lc = nc
                    bs = lc * base_pct * (1.0 + k_entropy * H)

    return bricks
