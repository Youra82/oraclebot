# src/oraclebot/utils/daily_gate.py
# Entscheidet, ob der taegliche Live-Prognose-Lauf (predict_next_candle.py) gerade tatsaechlich
# etwas tun soll -- als reine, von der Systemuhr entkoppelte Funktion, damit sie ohne Warten
# auf die echte Mitternacht per Test abgesichert werden kann.
import os

import pandas as pd


def check_daily_gate(now_utc: pd.Timestamp, marker_path: str):
    """Prueft Zeitfenster + Tages-Marker.

    Das Zeitfenster (00:00-00:29 UTC) ist absichtlich 30 Minuten breit (Puffer gegen verzoegerte
    Cron-Ticks), deckt bei einem */15-Cronjob aber ZWEI Ticks ab (:00 und :15), nicht nur einen.
    Ohne den Marker wuerde die volle Prognose- und Telegram-Logik daher zweimal pro Nacht laufen,
    mit leicht unterschiedlichen Zahlen (beobachtet 2026-07-14: zwei Nachrichten um 00:01 und
    00:16 UTC mit unterschiedlicher Konfidenz/SL/TP fuer dieselbe Zielkerze).

    Returns:
        (should_run, skip_reason): `skip_reason` ist None wenn should_run True ist, sonst ein
        Log-Text, der erklaert, warum uebersprungen wird.
    """
    if not (now_utc.hour == 0 and now_utc.minute < 30):
        return False, (
            f"Ausserhalb des taeglichen Ausfuehrungsfensters (00:00-00:29 UTC), aktuell "
            f"{now_utc.strftime('%H:%M')} UTC. Ueberspringe (kein Fehler) -- laeuft beim "
            f"naechsten passenden Cron-Intervall erneut.")

    today_str = now_utc.strftime('%Y-%m-%d')
    if os.path.exists(marker_path):
        with open(marker_path, 'r', encoding='utf-8') as f:
            last_run_date = f.read().strip()
        if last_run_date == today_str:
            return False, (
                f"Heutige Prognose ({today_str}) wurde bereits gesendet (siehe {marker_path}). "
                f"Ueberspringe (kein Fehler), verhindert Doppel-Versand durch mehrere Cron-Ticks "
                f"im 00:00-00:29-Fenster.")

    return True, None


def mark_daily_run_complete(now_utc: pd.Timestamp, marker_path: str) -> None:
    """Traegt den aktuellen UTC-Tag als 'heute schon gesendet' ein."""
    with open(marker_path, 'w', encoding='utf-8') as f:
        f.write(now_utc.strftime('%Y-%m-%d'))
