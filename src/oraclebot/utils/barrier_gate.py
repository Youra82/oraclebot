# src/oraclebot/utils/barrier_gate.py
# Wie daily_gate.py, aber fuer die 4h-Kadenz des Barriere-Modells: statt einmal pro Tag um
# Mitternacht soll predict_next_barrier.py bei JEDER 4h-Grenze (00/04/08/12/16/20 UTC) laufen.
import os

import pandas as pd


def check_barrier_gate(now_utc: pd.Timestamp, marker_path: str, period_hours: int = 4):
    """Prueft Zeitfenster + Perioden-Marker (analog zu daily_gate.check_daily_gate, nur fuer
    4h- statt Tages-Perioden). Zeitfenster ist die ersten 30 Minuten jeder `period_hours`-Grenze,
    Dedup ueber einen Marker mit dem ISO-Zeitstempel der aktuellen Periode (nicht nur dem Datum).

    Returns:
        (should_run, skip_reason): `skip_reason` ist None wenn should_run True ist.
    """
    if not (now_utc.hour % period_hours == 0 and now_utc.minute < 30):
        return False, (
            f"Ausserhalb des {period_hours}h-Ausfuehrungsfensters (Stunde muss durch "
            f"{period_hours} teilbar sein, erste 30 Minuten), aktuell {now_utc.strftime('%H:%M')} UTC. "
            f"Ueberspringe (kein Fehler) -- laeuft bei der naechsten Periodengrenze erneut.")

    period_start = now_utc.floor(f'{period_hours}h')
    period_str = period_start.isoformat()
    if os.path.exists(marker_path):
        with open(marker_path, 'r', encoding='utf-8') as f:
            last_run_period = f.read().strip()
        if last_run_period == period_str:
            return False, (
                f"Aktuelle Periode ({period_str}) wurde bereits verarbeitet (siehe {marker_path}). "
                f"Ueberspringe (kein Fehler), verhindert Doppel-Ausfuehrung durch mehrere Cron-Ticks "
                f"im selben Zeitfenster.")

    return True, None


def mark_barrier_run_complete(now_utc: pd.Timestamp, marker_path: str, period_hours: int = 4) -> None:
    """Traegt die aktuelle 4h-Periode als 'bereits verarbeitet' ein."""
    period_start = now_utc.floor(f'{period_hours}h')
    with open(marker_path, 'w', encoding='utf-8') as f:
        f.write(period_start.isoformat())
