# src/oraclebot/utils/training_history.py
# Protokolliert jeden train_barrier_model.py-Lauf (Zeitstempel + genutzte Tuning-Parameter +
# Out-of-Sample-Ergebnis) und warnt, wenn mehrere Laeufe mit UNTERSCHIEDLICHEN Parametern kurz
# hintereinander passieren -- ein Indiz fuer manuelles Parameter-Tuning gegen denselben
# Out-of-Sample-Holdout ("Mensch-im-Loop-Overfitting", siehe README). Anders als z.B. probebots
# Optuna-Trial-Akkumulation gibt es hier keinen Suchprozess, der sich technisch "sperren" liesse
# (train_barrier_model.py ist ein einzelner deterministischer Fit, random_state=0) -- deshalb
# bewusst als Warnung statt harter Sperre umgesetzt: reines Retraining auf neuen Daten mit
# UNVERAENDERTEN Parametern loest keine Warnung aus, das ist normales/gesundes Verhalten.
import json
import os
from datetime import datetime, timedelta, timezone

TUNABLE_KEYS = ['min_confidence', 'model_max_depth', 'barrier_pct', 'context_timeframes']
WARNING_WINDOW_HOURS = 24
MIN_RUNS_FOR_WARNING = 3


def _tuning_signature(entry: dict) -> tuple:
    """Vergleichbarer Fingerabdruck der Tuning-relevanten Parameter eines Laufs (Listen wie
    context_timeframes werden zu Tupeln, damit sie hashbar/vergleichbar sind)."""
    return tuple(
        tuple(entry.get(k)) if isinstance(entry.get(k), list) else entry.get(k)
        for k in TUNABLE_KEYS
    )


def append_entry(history_path: str, barrier_cfg: dict, val_accuracy: float,
                  walk_forward_mean: float, walk_forward_worst_case: float):
    """Haengt einen Trainingslauf an die Verlaufs-Datei an (JSON-Lines, append-only)."""
    entry = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'val_accuracy': val_accuracy,
        'walk_forward_mean': walk_forward_mean,
        'walk_forward_worst_case': walk_forward_worst_case,
    }
    for k in TUNABLE_KEYS:
        entry[k] = barrier_cfg.get(k)

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def load_history(history_path: str) -> list:
    if not os.path.exists(history_path):
        return []
    entries = []
    with open(history_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def check_overfitting_risk(history_path: str, window_hours: int = WARNING_WINDOW_HOURS,
                            min_runs: int = MIN_RUNS_FOR_WARNING) -> str:
    """Gibt eine Warnmeldung zurueck, wenn >= min_runs Laeufe innerhalb von window_hours Stunden
    UNTERSCHIEDLICHE Tuning-Parameter genutzt haben (Hinweis auf manuelles Parameter-Tuning
    gegen denselben Out-of-Sample-Holdout) -- sonst None. Muss NACH append_entry() fuer den
    aktuellen Lauf aufgerufen werden, damit dieser Lauf selbst mitgezaehlt wird."""
    entries = load_history(history_path)
    if len(entries) < min_runs:
        return None

    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    recent = []
    for e in entries:
        try:
            ts = datetime.fromisoformat(e['timestamp'])
        except (KeyError, ValueError, TypeError):
            continue
        if ts >= cutoff:
            recent.append(e)

    if len(recent) < min_runs:
        return None

    signatures = {_tuning_signature(e) for e in recent}
    if len(signatures) < 2:
        return None  # gleiche Parameter wiederholt -- normales Retraining, keine Warnung

    return (
        f"WARNUNG: {len(recent)} Trainingslaeufe in den letzten {window_hours}h mit "
        f"{len(signatures)} unterschiedlichen Parameter-Kombinationen ({', '.join(TUNABLE_KEYS)}).\n"
        f"Das kann ein Zeichen von manuellem Parameter-Tuning gegen denselben Out-of-Sample-\n"
        f"Holdout sein (Mensch-im-Loop-Overfitting) -- der Val-Split ist dann nicht mehr wirklich\n"
        f"ungesehen. Siehe README ('Wichtige Regeln'). Kein Blocker, nur ein Hinweis."
    )
