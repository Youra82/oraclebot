# src/oraclebot/analysis/hourly_volatility.py
# Historisches stuendliches Volatilitaets-Profil (0-23 UTC): wie viel einer typischen
# Tagesspanne sich erfahrungsgemaess in welcher Stunde realisiert. Rein statistisch -- KEIN
# Pfad-Vorhersagemodell, sagt nicht den tatsaechlichen Stundenverlauf voraus, nur die
# historisch typische Groessenordnung samt Tag-zu-Tag-Streuung.
#
# Braucht mehrjaehrige Historie, die bei der Live-Inferenz auf dem VPS nicht vorliegt (dort
# wird nur ein kleines rollierendes Fenster pro Timeframe gefetcht, siehe
# predict_next_candle.required_fetch_limit()). Wird deshalb einmalig waehrend des Trainings
# berechnet (train_transformer.py, dort liegt die volle Historie vor) und als kleine,
# git-getrackte JSON-Datei gespeichert -- nicht bei jeder taeglichen Live-Prognose neu
# berechnet.
import json

import pandas as pd


def compute_hourly_volatility_profile(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Pro UTC-Stunde (0-23): mittlerer und Streuungs-Anteil der stuendlichen |Close-Open|-
    Bewegung an der Tagesspanne (High-Low) DESSELBEN Kalendertages, ueber die gesamte Historie
    von `hourly_df` gemittelt.

    Args:
        hourly_df: 1h-OHLCV-DataFrame, DatetimeIndex (UTC).

    Returns:
        DataFrame, Index=Stunde (0-23), Spalten 'mean', 'std', 'count'. `count` ist die Anzahl
        historischer Tage, aus denen diese Stunde gemittelt wurde.
    """
    df = hourly_df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour

    daily_range = df.groupby('date', group_keys=False).apply(
        lambda g: g['high'].max() - g['low'].min(), include_groups=False)
    daily_range = daily_range[daily_range > 0]

    df['hour_move'] = (df['close'] - df['open']).abs()
    df['day_range'] = df['date'].map(daily_range)
    df = df.dropna(subset=['day_range'])
    df['hour_frac'] = df['hour_move'] / df['day_range']

    profile = df.groupby('hour')['hour_frac'].agg(['mean', 'std', 'count'])
    return profile.reindex(range(24))


def save_profile(profile: pd.DataFrame, path: str) -> None:
    data = {
        str(hour): {'mean': float(row['mean']), 'std': float(row['std']), 'count': int(row['count'])}
        for hour, row in profile.iterrows() if pd.notna(row['mean'])
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_profile(path: str) -> dict:
    """Laedt das gespeicherte Profil als {stunde (int): {'mean':.., 'std':.., 'count':..}}."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(hour): values for hour, values in raw.items()}


def compute_hourly_path_profile(hourly_df: pd.DataFrame) -> dict:
    """Getrennt fuer historische baerische (trend=0) und bullische (trend=1) Tage: mittlere und
    Streuungs-Position des STUNDEN-CLOSE innerhalb der jeweiligen Tagesspanne (0=Tagestief,
    1=Tageshoch), pro Stunde (0-23).

    Zeigt den TYPISCHEN Pfad, den Tage mit dieser Trendrichtung historisch genommen haben --
    KEINE Vorhersage des exakten Pfads fuer einen bestimmten Tag, nur ein historisches Muster,
    mit dem sich eine Tages-Prognose (Trend + Range) plausibel stundenweise "auffuellen" laesst.

    Args:
        hourly_df: 1h-OHLCV-DataFrame, DatetimeIndex (UTC).

    Returns:
        {trend (0/1): {stunde (0-23): {'mean':, 'std':, 'count':}}} -- trend-Codierung wie
        targets.TREND_LABELS (0=bearish, 1=bullish).
    """
    df = hourly_df.copy()
    df['date'] = df.index.date
    df['hour'] = df.index.hour

    daily = df.groupby('date').agg(open=('open', 'first'), high=('high', 'max'),
                                    low=('low', 'min'), close=('close', 'last'))
    daily['trend'] = (daily['close'] > daily['open']).astype(int)
    daily['range'] = daily['high'] - daily['low']
    daily = daily[daily['range'] > 0]

    df = df.join(daily[['low', 'range', 'trend']], on='date', rsuffix='_day')
    df = df.dropna(subset=['range'])
    df['frac'] = (df['close'] - df['low_day']) / df['range']

    result = {}
    for trend_value, group in df.groupby('trend'):
        stats = group.groupby('hour')['frac'].agg(['mean', 'std', 'count']).reindex(range(24))
        result[int(trend_value)] = {
            int(hour): {'mean': float(row['mean']), 'std': float(row['std']), 'count': int(row['count'])}
            for hour, row in stats.iterrows() if pd.notna(row['mean'])
        }
    return result


def save_path_profile(profile: dict, path: str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({str(trend): {str(h): v for h, v in hours.items()} for trend, hours in profile.items()},
                   f, indent=2)


def load_path_profile(path: str) -> dict:
    """Laedt das gespeicherte Pfad-Profil als {trend (int): {stunde (int): {...}}}."""
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    return {int(trend): {int(h): v for h, v in hours.items()} for trend, hours in raw.items()}
