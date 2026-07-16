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
