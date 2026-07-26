# src/oraclebot/utils/progress.py
# Gemeinsamer Ein-Zeilen-Fortschrittsbalken + laufender Timer, fuer alle Stellen im Projekt, die
# potenziell lange ohne sichtbare Ausgabe laufen (OHLCV-Fetch in data_fetch.py, Label-Berechnung
# in barrier_targets.py). Ausgelagert statt pro Modul dupliziert, damit beide Stellen exakt
# gleich aussehen/funktionieren.
import sys
import time


def render_progress(prefix: str, current: int, total: int, start_time: float, width: int = 30):
    """Ueberschreibt sich selbst per \\r (kein Zeilen-Spam) + zeigt einen laufenden mm:ss-Timer.
    Nur wenn stdout ein echtes Terminal ist (interaktiver Lauf) -- bei Cron-Ausfuehrung (Output
    nach logs/cron.log umgeleitet) wird NICHTS geschrieben, damit das Log sauber bleibt
    (isatty()==False dort)."""
    if not sys.stdout.isatty():
        return
    pct = min(current / total, 1.0) if total else 1.0
    filled = int(width * pct)
    bar = '#' * filled + '-' * (width - filled)
    elapsed = int(time.time() - start_time)
    mins, secs = divmod(elapsed, 60)
    sys.stdout.write(f"\r  [{bar}] {prefix}: {current}/{total} ({pct * 100:.0f}%) | {mins:02d}:{secs:02d}   ")
    sys.stdout.flush()


def finish_progress():
    """Schliesst die per \\r ueberschriebene Fortschrittszeile mit einem Zeilenumbruch ab,
    damit nachfolgende logger.info()-Ausgaben nicht an dieselbe Zeile angehaengt werden."""
    if sys.stdout.isatty():
        sys.stdout.write('\n')
        sys.stdout.flush()
