# src/oraclebot/data/barrier_targets.py
# Alternative Zielformulierung zu targets.py: statt die naechste TAGESKERZE zu klassifizieren,
# wird fuer JEDE Referenzkerze (Standard: 4h) direkt vorhergesagt, ob eine Kursbewegung von
# `barrier_pct`% zuerst nach oben oder nach unten erreicht wird -- das ist exakt die Frage, die
# fuer eine symmetrische SL=TP-Strategie zaehlt (siehe Recherche 2026-07-24), statt sie ueber den
# Umweg "wird morgen bullisch oder baerisch" indirekt zu erschliessen.
#
# Validiert (8-Fenster-Walk-Forward ueber 2.5 Jahre BTC/USDT:USDT, 4h-Referenz/15m-Aufloesung,
# depth=3 HistGradientBoosting auf den reinen 4h-Features): 62.0-71.2% Accuracy je Fenster,
# Mittel 67.5%, Standardabweichung nur 3.0pp -- deutlich robuster als targets.py's taegliches
# trend-Ziel (Mittel 59.0%, Worst-Case 56.0% bei nur 3 Fenstern). Ausserdem ~7x mehr
# Handelsgelegenheiten (4h- statt Tages-Kadenz).
import pandas as pd

BARRIER_LABELS = ['down_first', 'up_first']  # 0, 1


def compute_barrier_labels(reference_df: pd.DataFrame, intraday_df: pd.DataFrame,
                            barrier_pct: float = 1.0) -> pd.DataFrame:
    """Fuer jede Kerze in `reference_df`: wird ausgehend vom Schlusskurs zuerst eine Bewegung von
    +barrier_pct% oder -barrier_pct% erreicht (anhand der feineren `intraday_df`-Kerzen,
    chronologisch STRIKT NACH der Referenzkerze -- kein Blick in die eigene Kerze)?

    Args:
        reference_df: OHLCV-DataFrame der Referenz-Kerzen (Standard: 4h), DatetimeIndex.
        intraday_df: feinere OHLCV-Kerzen (Standard: 15m) fuer die Reihenfolge-Bestimmung.
        barrier_pct: symmetrischer Abstand in Prozent vom Schlusskurs.

    Returns:
        DataFrame (Index = reference_df-Zeitstempel) mit 'entry', 'label' (0=runter zuerst,
        1=hoch zuerst), 'exit_time'. Zeilen ohne bestimmbares Ergebnis (z.B. am Ende der
        verfuegbaren Historie, wo weder Barriere je erreicht wird) werden weggelassen.
    """
    records = []
    for ts, row in reference_df.iterrows():
        entry = float(row['close'])
        up_level = entry * (1 + barrier_pct / 100.0)
        down_level = entry * (1 - barrier_pct / 100.0)
        future_bars = intraday_df[intraday_df.index > ts]
        label, exit_time = None, None
        for fts, bar in future_bars.iterrows():
            if bar['low'] <= down_level:
                label, exit_time = 0, fts
                break
            if bar['high'] >= up_level:
                label, exit_time = 1, fts
                break
        if label is None:
            continue
        records.append({'ts': ts, 'entry': entry, 'label': label, 'exit_time': exit_time})

    if not records:
        return pd.DataFrame(columns=['entry', 'label', 'exit_time'])
    return pd.DataFrame(records).set_index('ts')


def build_barrier_examples(reference_df: pd.DataFrame, intraday_df: pd.DataFrame,
                            feature_kwargs: dict = None, barrier_pct: float = 1.0) -> list:
    """Baut flache Trainingsbeispiele: pro Referenzkerze ein Feature-Vektor + Barriere-Label.

    Anders als dataset.py's build_training_examples (mehrere Timeframes, Fenster-Historie) --
    hier reicht der Feature-Vektor DER Referenzkerze selbst (ATR-/EMA-basierte Features
    verarbeiten Historie bereits intern), keine Sequenz noetig.

    Returns:
        Liste von Dicts: {date, reference_time, entry, exit_time, features (Liste), target}.
    """
    from oraclebot.data.features import FEATURE_NAMES, compute_features

    feature_kwargs = feature_kwargs or {}
    feat = compute_features(reference_df, **feature_kwargs)
    labels = compute_barrier_labels(reference_df, intraday_df, barrier_pct=barrier_pct)

    joined_index = feat.index.intersection(labels.index)
    examples = []
    for ts in sorted(joined_index):
        examples.append({
            'date': ts.isoformat(),
            'reference_time': ts.isoformat(),
            'entry': float(labels.loc[ts, 'entry']),
            'exit_time': labels.loc[ts, 'exit_time'].isoformat(),
            'features': feat.loc[ts, FEATURE_NAMES].tolist(),
            'target': int(labels.loc[ts, 'label']),
        })
    return examples
