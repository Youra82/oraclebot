# src/oraclebot/data/features.py
# Markt-Token: wandelt eine Kerze (+ Kontext) in den Feature-Vektor der "Marktsprache" um.
import numpy as np
import pandas as pd
import ta

FEATURE_NAMES = [
    'return',          # (Close_t - Close_t-1) / Close_t-1
    'body',            # |Close-Open| / (High-Low)
    'upper_wick',      # (High - max(Open,Close)) / (High-Low)
    'lower_wick',      # (min(Open,Close) - Low) / (High-Low)
    'atr_range',        # (High-Low) / ATR_n  -- normierte Tagesrange
    'trend_state',      # (Close - EMA50) / ATR_n, kontinuierlich
    'structure',        # Marktstruktur-Score aus Swing HH/HL vs LH/LL, -2..+2
    'momentum',          # RSI-Zustand: (RSI - 50) / 50, kontinuierlich in [-1, 1]
    'velocity',          # (Close_t - Close_t-n) / (n * ATR_n) -- ATR-normierte Geschwindigkeit
    'volume_ratio',      # Volume_t / Durchschnitt(Volume, n)
    'higher_tf_position',  # (Close - Low_htf) / (High_htf - Low_htf) über Rolling-Window
    'macd_hist',           # (MACD-Linie - Signal-Linie) / ATR_n -- Momentum-Beschleunigung
    'resistance_distance',  # (naechste Widerstandszone oberhalb Close - Close) / ATR_n
    'support_distance',      # (Close - naechste Unterstuetzungszone unterhalb Close) / ATR_n
    'channel_position',       # (Close - untere Trendlinie) / (obere - untere Trendlinie); <0 oder >1 = Ausbruch
    'channel_slope',          # Steigung des lokalen Trendkanals (Swing-Highs/-Lows-Regression) / ATR_n
    'gap',                    # (Open_t - Close_t-1) / ATR_n
    'dow_sin', 'dow_cos',   # zyklische Wochentags-Codierung
    'month_start', 'month_end',  # binäre Zeit-Flags
]

REGIME_LABELS = ['trend_low_vol', 'range_high_vol', 'panic', 'range_low_vol', 'trend_high_vol']


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    return ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=window
    ).average_true_range()


def _compute_swings(df: pd.DataFrame, swing_window: int = 5) -> tuple:
    """Findet Swing-Highs/-Lows (lokale Extrema über `swing_window` Kerzen auf beiden Seiten).

    Wird sowohl fuer den Marktstruktur-Score als auch fuer Support/Resistance-Distanzen
    genutzt, damit die Swing-Erkennung nicht doppelt berechnet wird.

    Lookahead-Fix (2026-07-10): Ob Kerze i ein lokales Extremum ist, laesst sich erst
    beurteilen, wenn die folgenden `swing_window` Kerzen abgeschlossen sind -- man braucht
    die Zukunft, um zu wissen, dass i ein Hoch/Tief war. Das centered rolling() liefert also
    is_swing_high/-low technisch korrekt fuer Kerze i, aber real-time waere dieses Wissen erst
    `swing_window` Kerzen spaeter verfuegbar. Ohne Korrektur wuerden die letzten paar Kerzen vor
    einem Vorhersage-Cutoff (build_training_examples() in dataset.py) bereits Informationen ueber
    noch gar nicht abgeschlossene, teils in der Zukunft liegende Kerzen enthalten -- ein echtes
    Lookahead-Leck, das genau die juengsten und damit wichtigsten Feature-Zeilen betraf.
    Fix: jeder Swing-Punkt behaelt seinen echten Zeitstempel (fuer Geometrie/Trendlinien in
    _chart_technical_features), bekommt aber zusaetzlich `confirmed_at` = Zeitstempel
    `swing_window` Kerzen spaeter. Aufrufer duerfen einen Swing-Punkt nur verwenden, wenn
    `confirmed_at <= ts` (nicht der eigene Zeitstempel) -- das entspricht exakt dem Wissensstand
    eines Live-Bots zum Zeitpunkt ts.
    """
    high, low = df['high'], df['low']
    is_swing_high = high == high.rolling(2 * swing_window + 1, center=True).max()
    is_swing_low = low == low.rolling(2 * swing_window + 1, center=True).min()
    confirmed_at_full = pd.Series(df.index, index=df.index).shift(-swing_window)

    swing_highs = pd.DataFrame({
        'price': high[is_swing_high], 'confirmed_at': confirmed_at_full[is_swing_high],
    }).dropna()
    swing_lows = pd.DataFrame({
        'price': low[is_swing_low], 'confirmed_at': confirmed_at_full[is_swing_low],
    }).dropna()
    return swing_highs, swing_lows


def _market_structure_score(swing_highs: pd.DataFrame, swing_lows: pd.DataFrame, index: pd.Index,
                             lookback: int = 20) -> pd.Series:
    """Swing-High/Low-basierter Trendstruktur-Score in {-2,-1,0,+1,+2}.

    Vergleich der letzten zwei Swing-Highs/-Lows in `lookback` Kerzen ergibt HH/HL bzw. LH/LL.
    Filterung ueber `confirmed_at <= ts` (nicht den eigenen Zeitstempel), siehe _compute_swings().
    """
    scores = pd.Series(0, index=index, dtype=float)

    for i, ts in enumerate(index):
        window_start = index[max(0, i - lookback)]
        recent_highs = swing_highs[(swing_highs['confirmed_at'] >= window_start) & (swing_highs['confirmed_at'] <= ts)]
        recent_lows = swing_lows[(swing_lows['confirmed_at'] >= window_start) & (swing_lows['confirmed_at'] <= ts)]

        higher_high = len(recent_highs) >= 2 and recent_highs['price'].iloc[-1] > recent_highs['price'].iloc[-2]
        higher_low = len(recent_lows) >= 2 and recent_lows['price'].iloc[-1] > recent_lows['price'].iloc[-2]
        lower_high = len(recent_highs) >= 2 and recent_highs['price'].iloc[-1] < recent_highs['price'].iloc[-2]
        lower_low = len(recent_lows) >= 2 and recent_lows['price'].iloc[-1] < recent_lows['price'].iloc[-2]

        if higher_high and higher_low:
            scores.iloc[i] = 2
        elif higher_high or higher_low:
            scores.iloc[i] = 1
        elif lower_high and lower_low:
            scores.iloc[i] = -2
        elif lower_high or lower_low:
            scores.iloc[i] = -1
        else:
            scores.iloc[i] = 0

    return scores


def _higher_tf_position(df: pd.DataFrame, window: int) -> pd.Series:
    roll_high = df['high'].rolling(window).max()
    roll_low = df['low'].rolling(window).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    return (df['close'] - roll_low) / rng


def _cluster_levels(prices: np.ndarray, tolerance: float) -> list:
    """Gruppiert nahe beieinanderliegende Swing-Preise zu Zonen (wie die per Hand gezeichneten

    Widerstands-/Unterstuetzungs-Baender im Chart, statt einzelner Punktlevel). Punkte, die
    hoechstens `tolerance` auseinanderliegen, werden zu einer Zone zusammengefasst (Zonen-Level
    = Mittelwert). Gibt eine Liste der Zonen-Level zurueck.
    """
    if len(prices) == 0:
        return []
    prices = np.sort(prices)
    clusters = [[prices[0]]]
    for p in prices[1:]:
        if p - clusters[-1][-1] <= tolerance:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [float(np.mean(c)) for c in clusters]


def _chart_technical_features(df: pd.DataFrame, swing_highs: pd.DataFrame, swing_lows: pd.DataFrame,
                               atr: pd.Series, lookback: int = 20, zone_tolerance_atr: float = 0.5,
                               no_level_atr: float = 3.0) -> tuple:
    """Berechnet zwei klassische chart-technische Konzepte aus Swing-Highs/-Lows:

    1) Support-/Resistance-ZONEN (nahe Swing-Punkte werden geclustert statt als
       Einzellevel behandelt -- entspricht den von Hand gezeichneten Widerstandsbaendern).
    2) Lokaler Trendkanal: lineare Regression durch die Swing-Highs (obere Trendlinie) und
       durch die Swing-Lows (untere Trendlinie) im Lookback-Fenster. `channel_position`
       (Close relativ zu den beiden Linien) verlaesst [0,1], sobald der Kanal durchbrochen
       wird -- das entspricht direkt "Trendlinie gebrochen = lokale Richtungsaenderung".

    Gibt (resistance_dist, support_dist, channel_position, channel_slope, no_level_atr) zurueck,
    jeweils in ATR-Einheiten wo zutreffend (Division durch ATR erfolgt beim Aufrufer).
    """
    close = df['close']
    resistance = pd.Series(np.nan, index=df.index)
    support = pd.Series(np.nan, index=df.index)
    channel_position = pd.Series(np.nan, index=df.index)
    channel_slope = pd.Series(np.nan, index=df.index)

    for i, ts in enumerate(df.index):
        window_start = df.index[max(0, i - lookback)]
        # Filterung ueber confirmed_at (nicht den eigenen Swing-Zeitstempel), siehe _compute_swings().
        recent_highs = swing_highs[(swing_highs['confirmed_at'] >= window_start) & (swing_highs['confirmed_at'] <= ts)]
        recent_lows = swing_lows[(swing_lows['confirmed_at'] >= window_start) & (swing_lows['confirmed_at'] <= ts)]
        c = close.iloc[i]
        a = atr.iloc[i]
        tolerance = zone_tolerance_atr * a if pd.notna(a) and a > 0 else 0.0

        # 1) Zonen statt Einzelpunkte
        resistance_zones = [z for z in _cluster_levels(recent_highs['price'].values, tolerance) if z > c]
        support_zones = [z for z in _cluster_levels(recent_lows['price'].values, tolerance) if z < c]
        resistance.iloc[i] = (min(resistance_zones) - c) if resistance_zones else np.nan
        support.iloc[i] = (c - max(support_zones)) if support_zones else np.nan

        # 2) Lokaler Trendkanal aus Swing-Highs (oben) / Swing-Lows (unten)
        # x-Koordinaten nutzen den ECHTEN Swing-Zeitstempel (recent_highs.index), nicht
        # confirmed_at -- die Trendlinie muss geometrisch durch die tatsaechlichen Extrempunkte
        # verlaufen, nur die Zulassung zum Zeitpunkt ts ist ueber confirmed_at gegated.
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            x_highs = np.array([df.index.get_loc(t) for t in recent_highs.index], dtype=float)
            x_lows = np.array([df.index.get_loc(t) for t in recent_lows.index], dtype=float)
            upper_slope, upper_intercept = np.polyfit(x_highs, recent_highs['price'].values, 1)
            lower_slope, lower_intercept = np.polyfit(x_lows, recent_lows['price'].values, 1)
            upper_val = upper_slope * i + upper_intercept
            lower_val = lower_slope * i + lower_intercept
            channel_width = upper_val - lower_val
            channel_position.iloc[i] = (c - lower_val) / channel_width if channel_width > 0 else 0.5
            channel_slope.iloc[i] = (upper_slope + lower_slope) / 2.0

    return resistance, support, channel_position, channel_slope, no_level_atr


def classify_regime(row: pd.Series) -> str:
    """Ordnet einen Feature-Vektor einem der 5 Marktregime zu (regelbasiert, Platzhalter für spätere Clustering-Ablösung)."""
    high_vol = row['atr_range'] > 1.3
    low_vol = row['atr_range'] < 0.7
    trending = abs(row['structure']) >= 1

    if high_vol and abs(row['return']) > 0.03:
        return 'panic'
    if trending and high_vol:
        return 'trend_high_vol'
    if trending and low_vol:
        return 'trend_low_vol'
    if low_vol:
        return 'range_low_vol'
    return 'range_high_vol'


def compute_features(df: pd.DataFrame, atr_window: int = 14, ema_window: int = 50,
                      volume_window: int = 20, velocity_window: int = 10,
                      structure_swing_window: int = 5, structure_lookback: int = 20,
                      higher_tf_window: int = 20, rsi_window: int = 14,
                      macd_fast: int = 12, macd_slow: int = 26, macd_signal_window: int = 9,
                      sr_lookback: int = 20, zone_tolerance_atr: float = 0.5) -> pd.DataFrame:
    """Berechnet den Markt-Token-Feature-Vektor für jede Kerze eines OHLCV-DataFrames.

    Args:
        df: DataFrame mit Spalten [open, high, low, close, volume], DatetimeIndex (UTC, aufsteigend sortiert).
        rsi_window: Fenster fuer RSI (separat parametrierbar, da 14 Kerzen auf Monats-/Wochenbasis
            Jahre an Warmup braeuchten -- fuer 1M/1w sollte ein kleineres Fenster uebergeben werden).
        macd_fast/macd_slow/macd_signal_window: MACD-Perioden (Standard 12/26/9 wie ueblich;
            fuer 1M/1w sollten deutlich kleinere Werte uebergeben werden, siehe feature_settings_by_timeframe).
        zone_tolerance_atr: Swing-Punkte innerhalb dieses ATR-Vielfachen werden zu einer
            Support-/Resistance-Zone zusammengefasst (siehe _cluster_levels).
        sr_lookback: Fenster (Anzahl Kerzen), in dem nach Swing-Highs/-Lows fuer die
            Support/Resistance-Distanz gesucht wird.

    Returns:
        DataFrame mit Spalten FEATURE_NAMES + 'regime' (NaN-Zeilen aus Warmup-Fenstern entfernt).
        Leer, wenn df kuerzer als die benoetigte Warmup-Historie ist (die `ta`-Bibliothek
        wirft sonst IndexError statt NaN zu produzieren).
    """
    df = df.copy()
    min_len = max(atr_window, ema_window, volume_window, velocity_window, rsi_window,
                  macd_slow + macd_signal_window) + 1
    if len(df) < min_len:
        return pd.DataFrame(columns=FEATURE_NAMES + ['regime'])

    atr = _atr(df, atr_window)
    ema = ta.trend.EMAIndicator(close=df['close'], window=ema_window).ema_indicator()
    rsi = ta.momentum.RSIIndicator(close=df['close'], window=rsi_window).rsi()
    macd_ind = ta.trend.MACD(close=df['close'], window_fast=macd_fast, window_slow=macd_slow,
                              window_sign=macd_signal_window)
    macd_hist = macd_ind.macd() - macd_ind.macd_signal()
    swing_highs, swing_lows = _compute_swings(df, structure_swing_window)
    resistance_dist, support_dist, channel_position, channel_slope, no_level_atr = _chart_technical_features(
        df, swing_highs, swing_lows, atr, sr_lookback, zone_tolerance_atr)
    hl_range = (df['high'] - df['low']).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)
    out['return'] = df['close'].pct_change()
    out['body'] = (df['close'] - df['open']).abs() / hl_range
    out['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / hl_range
    out['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / hl_range
    out['atr_range'] = hl_range / atr.replace(0, np.nan)
    out['trend_state'] = (df['close'] - ema) / atr.replace(0, np.nan)
    out['structure'] = _market_structure_score(swing_highs, swing_lows, df.index, structure_lookback)
    out['momentum'] = (rsi - 50.0) / 50.0
    out['velocity'] = (df['close'] - df['close'].shift(velocity_window)) / (velocity_window * atr.replace(0, np.nan))
    out['volume_ratio'] = df['volume'] / df['volume'].rolling(volume_window).mean()
    out['higher_tf_position'] = _higher_tf_position(df, higher_tf_window)
    out['macd_hist'] = macd_hist / atr.replace(0, np.nan)
    out['resistance_distance'] = (resistance_dist / atr.replace(0, np.nan)).fillna(no_level_atr)
    out['support_distance'] = (support_dist / atr.replace(0, np.nan)).fillna(no_level_atr)
    out['channel_position'] = channel_position.fillna(0.5)
    out['channel_slope'] = (channel_slope / atr.replace(0, np.nan)).fillna(0.0)
    out['gap'] = (df['open'] - df['close'].shift(1)) / atr.replace(0, np.nan)

    dow = df.index.dayofweek
    out['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    out['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    out['month_start'] = (df.index.day <= 3).astype(float)
    out['month_end'] = (df.index.day >= df.index.days_in_month - 2).astype(float)

    out = out.dropna(subset=[c for c in FEATURE_NAMES if c not in ('dow_sin', 'dow_cos', 'month_start', 'month_end')])
    out['regime'] = out.apply(classify_regime, axis=1)
    return out
