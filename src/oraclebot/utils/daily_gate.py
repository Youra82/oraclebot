# src/oraclebot/utils/daily_gate.py
# Entscheidet, ob der taegliche Live-Prognose-Lauf (predict_next_candle.py) gerade tatsaechlich
# etwas tun soll -- als reine, von der Systemuhr entkoppelte Funktion, damit sie ohne Warten
# auf die echte Mitternacht per Test abgesichert werden kann.
import os
from zoneinfo import ZoneInfo

import pandas as pd

NOTIFICATION_TIMEZONE = ZoneInfo('Europe/Berlin')
WINDOW_MINUTES = 30
DEFAULT_NOTIFICATION_TIME_LOCAL = "02:05"

# Sicherheitspuffer fuer die Cache-/Fetch-Fehler-Erkennung in predict_next_candle.py: die
# Tageskerze braucht 24h, dazu kommt der Abstand von UTC-Mitternacht bis zur konfigurierten
# Uhrzeit -- alles darueber hinaus ist ein Hinweis auf einen echten Fetch-Fehler, kein normaler
# Tagesablauf. Grosszuegig bemessen (nicht knapp am erwarteten Wert), damit ein leicht
# verzoegerter Cron-Tick nicht faelschlich als Fehler gilt.
STALENESS_SAFETY_BUFFER = pd.Timedelta(hours=6)


def _target_local_datetime(now_utc: pd.Timestamp, notification_time_local: str) -> pd.Timestamp:
    target_hour, target_minute = (int(part) for part in notification_time_local.split(':'))
    now_local = now_utc.tz_convert(NOTIFICATION_TIMEZONE)
    return now_local.normalize() + pd.Timedelta(hours=target_hour, minutes=target_minute)


def check_daily_gate(now_utc: pd.Timestamp, marker_path: str,
                      notification_time_local: str = DEFAULT_NOTIFICATION_TIME_LOCAL):
    """Prueft, ob JETZT der (in settings.json konfigurierte) taegliche Prognose-Zeitpunkt in
    deutscher Ortszeit ist, und ob heute schon gesendet wurde.

    `notification_time_local` ("HH:MM") ist frei waehlbar (settings.json,
    notification_settings.notification_time_local) -- keine eingebaute Untergrenze. Wird eine
    Uhrzeit gewaehlt, zu der die UTC-Tageskerze (schliesst immer um 00:00 UTC) noch nicht
    abgeschlossen ist, rechnet die Prognose einfach mit der zuletzt tatsaechlich abgeschlossenen
    Kerze -- das ist kein Fehler, nur eine andere Zielkerze als bei einer spaeteren Uhrzeit.

    Das Zeitfenster ist absichtlich 30 Minuten breit (Puffer gegen verzoegerte Cron-Ticks),
    deckt bei einem */15-Cronjob aber ZWEI Ticks ab, nicht nur einen. Ohne den Marker wuerde die
    volle Prognose- und Telegram-Logik daher zweimal pro Tag laufen, mit leicht unterschiedlichen
    Zahlen (beobachtet 2026-07-14: zwei Nachrichten um 00:01 und 00:16 UTC mit unterschiedlicher
    Konfidenz/SL/TP fuer dieselbe Zielkerze).

    Args:
        now_utc: aktuelle (oder simulierte) Zeit, tz-aware UTC.
        marker_path: Pfad der Tages-Marker-Datei.
        notification_time_local: "HH:MM" in deutscher Ortszeit (Europe/Berlin, DST-bewusst).

    Returns:
        (should_run, skip_reason): `skip_reason` ist None wenn should_run True ist, sonst ein
        Log-Text, der erklaert, warum uebersprungen wird.
    """
    target_local = _target_local_datetime(now_utc, notification_time_local)
    now_local = now_utc.tz_convert(NOTIFICATION_TIMEZONE)
    window_end = target_local + pd.Timedelta(minutes=WINDOW_MINUTES)

    if now_local < target_local:
        return False, (
            f"Noch nicht so weit: taegliche Prognose ist auf {notification_time_local} "
            f"deutscher Zeit eingestellt, aktuell {now_local.strftime('%H:%M %Z')}. "
            f"Ueberspringe (kein Fehler).")
    if now_local >= window_end:
        return False, (
            f"Ausserhalb des taeglichen Ausfuehrungsfensters ({notification_time_local}-"
            f"{window_end.strftime('%H:%M')} deutscher Zeit), aktuell "
            f"{now_local.strftime('%H:%M %Z')}. Ueberspringe (kein Fehler) -- laeuft morgen zur "
            f"konfigurierten Zeit erneut.")

    today_str = now_utc.strftime('%Y-%m-%d')
    if os.path.exists(marker_path):
        with open(marker_path, 'r', encoding='utf-8') as f:
            last_run_date = f.read().strip()
        if last_run_date == today_str:
            return False, (
                f"Heutige Prognose ({today_str}) wurde bereits gesendet (siehe {marker_path}). "
                f"Ueberspringe (kein Fehler), verhindert Doppel-Versand durch mehrere Cron-Ticks "
                f"im Zeitfenster.")

    return True, None


def mark_daily_run_complete(now_utc: pd.Timestamp, marker_path: str) -> None:
    """Traegt den aktuellen UTC-Tag als 'heute schon gesendet' ein."""
    with open(marker_path, 'w', encoding='utf-8') as f:
        f.write(now_utc.strftime('%Y-%m-%d'))


def max_expected_staleness(now_utc: pd.Timestamp, notification_time_local: str) -> pd.Timedelta:
    """Wie 'alt' die letzte abgeschlossene Tageskerze bei einem normalen Lauf zur konfigurierten
    `notification_time_local` hoechstens sein sollte -- 24h Kerzendauer + Abstand von UTC-
    Mitternacht bis zur konfigurierten Uhrzeit (DST-bewusst ueber `now_utc`s Kalendertag) +
    Sicherheitspuffer. Ersetzt eine vorher fest verdrahtete 30h-Schwelle, die nur fuer eine feste
    "kurz nach Mitternacht"-Ausfuehrung stimmte und bei spaeteren `notification_time_local`-Werten
    jeden Tag faelschlich als Cache-Fehler durchgeschlagen waere."""
    target_local = _target_local_datetime(now_utc, notification_time_local)
    target_utc = target_local.tz_convert('UTC')
    utc_midnight_today = now_utc.normalize()
    offset_from_utc_midnight = target_utc - utc_midnight_today
    if offset_from_utc_midnight < pd.Timedelta(0):
        offset_from_utc_midnight += pd.Timedelta(days=1)
    return pd.Timedelta(hours=24) + offset_from_utc_midnight + STALENESS_SAFETY_BUFFER
