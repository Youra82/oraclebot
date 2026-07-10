# src/oraclebot/data/targets.py
# Zielvariablen: zerlegt die naechste Kerze in kategoriale Ziele statt roher OHLC-Werte.
import numpy as np
import pandas as pd
import ta

TREND_LABELS = ['bearish', 'bullish']                      # 0, 1 -- kein Neutral-Bucket mehr,
# siehe compute_targets(): eine explizite Neutral-Zone hat in Tests (sklearn-Baselines auf
# denselben Features, 2026-07-09) das lernbare Signal stark reduziert (2-Klassen-Vorsprung
# ueber Zufalls-Baseline deutlich groesser als bei 3 Klassen mit Neutral-Zone).
RANGE_LABELS = ['0-0.5atr', '0.5-1atr', '1-2atr', '>2atr']  # 0..3
CLOSE_POS_LABELS = ['lower_third', 'middle_third', 'upper_third']  # 0..2
WICK_LABELS = ['small', 'medium', 'large']                  # 0..2
GAP_LABELS = ['no_gap', 'gap']                              # 0..1
INSIDE_OUTSIDE_LABELS = ['normal', 'inside_day', 'outside_day']  # 0..2
HIGH_FIRST_LABELS = ['low_first', 'high_first']              # 0..1

# Die ersten 5 sind die geometrischen Zielgroessen, aus denen reconstruct.py Preis-Koordinaten
# baut. Die letzten 3 sind zusaetzliche Markt-Struktur-Groessen (Vorschlag aus externer
# Code-Analyse: "Marktgrammatik" wie Gap-Verhalten, Inside/Outside Day, Reihenfolge der Extreme).
TARGET_NAMES = ['trend', 'range', 'close_position', 'upper_wick', 'lower_wick',
                'gap_yn', 'inside_outside_day', 'high_first']


def _bin_range(atr_ratio: pd.Series) -> pd.Series:
    return pd.cut(atr_ratio, bins=[-np.inf, 0.5, 1.0, 2.0, np.inf], labels=[0, 1, 2, 3]).astype('Int64')


def _bin_close_position(cp: pd.Series) -> pd.Series:
    return pd.cut(cp, bins=[-np.inf, 1 / 3, 2 / 3, np.inf], labels=[0, 1, 2]).astype('Int64')


def _bin_wick(wick_ratio: pd.Series, small_max: float = 0.15, medium_max: float = 0.35) -> pd.Series:
    return pd.cut(wick_ratio, bins=[-np.inf, small_max, medium_max, np.inf], labels=[0, 1, 2]).astype('Int64')


def _compute_high_first(daily_df: pd.DataFrame, intraday_df: pd.DataFrame) -> pd.Series:
    """Fuer jede Tageskerze: wurde das Tageshoch oder das Tagestief zuerst erreicht?

    Bestimmt anhand der Intraday-Kerzen (z.B. 4h), welche zuerst das Tageshoch bzw. Tagestief
    beruehrt -- reine Marktstruktur-Information, die aus der Tageskerze selbst nicht
    hervorgeht. 1 = Hoch zuerst, 0 = Tief zuerst. NaN, wenn zu wenige Intraday-Kerzen fuer
    den Tag gefunden werden oder Hoch/Tief in derselben Intraday-Kerze liegen (Reihenfolge
    nicht bestimmbar).
    """
    result = pd.Series(np.nan, index=daily_df.index)
    idx = intraday_df.index
    for i, day_start in enumerate(daily_df.index):
        day_end = day_start + pd.Timedelta(days=1)
        start_pos = idx.searchsorted(day_start, side='left')
        end_pos = idx.searchsorted(day_end, side='left')
        day_bars = intraday_df.iloc[start_pos:end_pos]
        if len(day_bars) < 2:
            continue
        high_idx = day_bars['high'].idxmax()
        low_idx = day_bars['low'].idxmin()
        if high_idx == low_idx:
            continue
        result.iloc[i] = 1.0 if high_idx < low_idx else 0.0
    return result


def compute_targets(df: pd.DataFrame, intraday_df: pd.DataFrame, atr_window: int = 14,
                     gap_atr_threshold: float = 0.1) -> pd.DataFrame:
    """Berechnet die Zielvariablen fuer JEDE Kerze aus ihren eigenen OHLC-Werten.

    Wichtig: Diese Funktion beschreibt die Kerze bei Index i selbst (ihr realisiertes
    Ergebnis), nicht eine Vorhersage. Beim Dataset-Bau wird target(t+1) an input(t)
    angehaengt -- so bleibt die Zukunftsdaten-Regel (Punkt 7 der Spezifikation) explizit
    an einer Stelle sichtbar statt in dieser Funktion versteckt.

    Args:
        df: OHLCV-DataFrame mit DatetimeIndex (Referenz-Timeframe, i.d.R. Tageskerzen).
        intraday_df: OHLCV-DataFrame eines feineren Timeframes (z.B. 4h) -- noetig um
            `high_first` zu bestimmen (welches Extrem innerhalb der Kerze zuerst erreicht wurde;
            aus der Kerze selbst nicht ablesbar).
        atr_window: Fenster fuer die ATR-Berechnung (muss zum Feature-ATR passen).
        gap_atr_threshold: `gap_yn`=1 (Gap), wenn |Open_t - Close_t-1| > gap_atr_threshold * ATR.

    Returns:
        DataFrame mit TARGET_NAMES (kategoriale Int-Codes) + rohen Kontinuierlichen Werten
        ('close_position_raw', 'range_atr_raw') fuer Debugging/Analyse.
    """
    df = df.copy()
    if len(df) < atr_window + 1:
        return pd.DataFrame(columns=TARGET_NAMES + ['range_atr_raw', 'close_position_raw'])

    atr = ta.volatility.AverageTrueRange(
        high=df['high'], low=df['low'], close=df['close'], window=atr_window
    ).average_true_range()
    hl_range = (df['high'] - df['low']).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)

    # Binaer (hoch/runter), KEINE Neutral-Zone mehr: ein sklearn-Baseline-Test (2026-07-09,
    # gleiche Features) zeigte einen Vorsprung von +13.6pp (BTC) / +6pp (ETH, RandomForest)
    # ueber Zufalls-Baseline bei binaerer Formulierung, gegenueber nur +4-7pp mit der alten
    # 3-Klassen-Version mit Neutral-Zone -- die Zone hat an der Klassengrenze echtes Signal
    # gekostet, statt es abzubilden.
    ret = df['close'].pct_change()
    trend = pd.Series(0, index=df.index, dtype='Int64')  # bearish
    trend[ret > 0] = 1  # bullish
    out['trend'] = trend

    out['range_atr_raw'] = hl_range / atr.replace(0, np.nan)
    out['range'] = _bin_range(out['range_atr_raw'])

    cp = (df['close'] - df['low']) / hl_range
    out['close_position_raw'] = cp
    out['close_position'] = _bin_close_position(cp)

    upper_wick = (df['high'] - df[['open', 'close']].max(axis=1)) / hl_range
    lower_wick = (df[['open', 'close']].min(axis=1) - df['low']) / hl_range
    out['upper_wick'] = _bin_wick(upper_wick)
    out['lower_wick'] = _bin_wick(lower_wick)

    gap_atr_raw = (df['open'] - df['close'].shift(1)) / atr.replace(0, np.nan)
    out['gap_yn'] = (gap_atr_raw.abs() > gap_atr_threshold).astype('Int64')

    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    inside = (df['high'] <= prev_high) & (df['low'] >= prev_low)
    outside = (df['high'] >= prev_high) & (df['low'] <= prev_low)
    inside_outside = pd.Series(0, index=df.index, dtype='Int64')  # normal
    inside_outside[inside] = 1
    inside_outside[outside] = 2
    out['inside_outside_day'] = inside_outside

    out['high_first'] = _compute_high_first(df, intraday_df).astype('Int64')

    return out.dropna(subset=TARGET_NAMES)
