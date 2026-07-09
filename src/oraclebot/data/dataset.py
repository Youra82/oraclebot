# src/oraclebot/data/dataset.py
# Baut Multi-Timeframe-Trainingsbeispiele: X_t = [Kerzen mehrerer Timeframes bis inkl. t] -> Y_(t+1)
import json
import logging

import pandas as pd

from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.data.targets import TARGET_NAMES, compute_targets

logger = logging.getLogger(__name__)

# Default-Fenstergroessen (Anzahl Kerzen) je Timeframe, siehe Spezifikation Punkt 3.
DEFAULT_WINDOW_SIZES = {
    '1M': 12,
    '1w': 52,
    '1d': 200,
    '4h': 500,
    '1h': 1000,
    '15m': 2000,
}


def build_training_examples(ohlcv_by_timeframe: dict, reference_timeframe: str = '1d',
                             intraday_timeframe: str = '4h', window_sizes: dict = None,
                             feature_kwargs: dict = None, feature_kwargs_by_timeframe: dict = None,
                             target_kwargs: dict = None) -> list:
    """Baut Trainingsbeispiele nach der Spezifikation:

    Input X_t: fuer jeden Timeframe die letzten `window_sizes[tf]` Markt-Token
               mit Zeitstempel < Beginn der Ziel-Kerze (keine Zukunftsdaten, Punkt 7).
    Output Y_(t+1): kategoriale Zielvariablen der naechsten Kerze des Referenz-Timeframes
               (trend, range, close_position, upper_wick, lower_wick, gap_yn,
               inside_outside_day, high_first), Punkt 4.

    Args:
        ohlcv_by_timeframe: dict {timeframe: OHLCV-DataFrame mit DatetimeIndex, aufsteigend sortiert}.
        reference_timeframe: Timeframe, dessen naechste Kerze vorhergesagt werden soll (Standard: Tageskerze).
        intraday_timeframe: feinerer Timeframe, aus dem `high_first` bestimmt wird (siehe targets.py).
        window_sizes: dict {timeframe: Anzahl Kerzen im Eingabefenster}. Default: DEFAULT_WINDOW_SIZES.
        feature_kwargs: an compute_features() durchgereichte Parameter, fuer alle Timeframes gleich.
        feature_kwargs_by_timeframe: optionale Overrides je Timeframe (ueberschreibt feature_kwargs
            fuer den jeweiligen Timeframe). Wichtig fuer 1M/1w: die Standard-Indikatorfenster
            (EMA-50, ATR-14 Kerzen) sind auf Tages-/Intraday-Skala kalibriert -- auf Monatskerzen
            angewandt braeuchten sie >4 Jahre Historie allein fuers Warmup.
        target_kwargs: an compute_targets() durchgereichte Parameter.

    Returns:
        Liste von Dicts im Format {date, <timeframe>: [[feature...], ...], target: {...}}.
        Beispiele, fuer die ein Timeframe nicht genug historische Kerzen hat, werden uebersprungen.
    """
    window_sizes = window_sizes or DEFAULT_WINDOW_SIZES
    feature_kwargs = feature_kwargs or {}
    feature_kwargs_by_timeframe = feature_kwargs_by_timeframe or {}
    target_kwargs = target_kwargs or {}

    if reference_timeframe not in ohlcv_by_timeframe:
        raise ValueError(f"reference_timeframe '{reference_timeframe}' fehlt in ohlcv_by_timeframe")
    if intraday_timeframe not in ohlcv_by_timeframe:
        raise ValueError(f"intraday_timeframe '{intraday_timeframe}' fehlt in ohlcv_by_timeframe")

    feature_frames = {
        tf: compute_features(df, **{**feature_kwargs, **feature_kwargs_by_timeframe.get(tf, {})})
        for tf, df in ohlcv_by_timeframe.items()
    }
    target_frame = compute_targets(
        ohlcv_by_timeframe[reference_timeframe], ohlcv_by_timeframe[intraday_timeframe], **target_kwargs)

    ref_index = feature_frames[reference_timeframe].index
    examples = []
    skipped_insufficient_history = 0

    for i in range(len(ref_index) - 1):
        t, t_next = ref_index[i], ref_index[i + 1]
        if t_next not in target_frame.index:
            continue

        example = {'date': t_next.isoformat(), 'reference_time': t.isoformat()}
        complete = True

        for tf, window in window_sizes.items():
            ff = feature_frames.get(tf)
            if ff is None:
                complete = False
                break
            # Cutoff bei t_next (Oeffnungszeit der Ziel-Kerze) = Schlusszeit der Referenz-Kerze t.
            # So sind ausschliesslich Kerzen zugelassen, die vollstaendig VOR der Vorhersage abgeschlossen sind.
            cutoff_pos = ff.index.searchsorted(t_next, side='left')
            window_slice = ff.iloc[max(0, cutoff_pos - window):cutoff_pos]
            if len(window_slice) < window:
                complete = False
                break
            example[tf] = window_slice[FEATURE_NAMES].values.tolist()

        if not complete:
            skipped_insufficient_history += 1
            continue

        example['target'] = target_frame.loc[t_next, TARGET_NAMES].astype(int).to_dict()
        examples.append(example)

    if skipped_insufficient_history:
        logger.info(f"{skipped_insufficient_history} Beispiele wegen unzureichender Historie uebersprungen.")
    logger.info(f"{len(examples)} Trainingsbeispiele gebaut (Referenz-Timeframe: {reference_timeframe}).")
    return examples


def save_dataset_jsonl(examples: list, path: str):
    """Speichert Trainingsbeispiele als JSON-Lines-Datei (ein Beispiel pro Zeile)."""
    with open(path, 'w', encoding='utf-8') as f:
        for example in examples:
            f.write(json.dumps(example) + '\n')
    logger.info(f"{len(examples)} Beispiele gespeichert: {path}")


def load_dataset_jsonl(path: str) -> list:
    """Laedt Trainingsbeispiele aus einer JSON-Lines-Datei."""
    examples = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples
