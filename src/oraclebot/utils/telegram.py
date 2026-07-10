# src/oraclebot/utils/telegram.py
# Gleiche Konvention wie mbot/dnabot: bot_token/chat_id kommen aus secret.json,
# unabhaengig von live_trading_enabled -- Prognose-Benachrichtigung und Order-Ausfuehrung
# sind zwei getrennte Schalter (siehe notification_settings/strategy_settings in settings.json).
import logging

import requests

logger = logging.getLogger(__name__)


def send_message(bot_token: str, chat_id: str, message: str):
    """Sendet eine Textnachricht an einen Telegram-Chat (MarkdownV2, escaped)."""
    if not bot_token or not chat_id:
        logger.warning("Telegram Bot-Token oder Chat-ID nicht konfiguriert. Nachricht nicht gesendet.")
        return

    escape_chars = r'_*[]()~`>#+-=|{}.!'
    escaped = message
    for char in escape_chars:
        escaped = escaped.replace(char, f'\\{char}')

    api_url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {'chat_id': chat_id, 'text': escaped, 'parse_mode': 'MarkdownV2'}

    try:
        response = requests.post(api_url, data=payload, timeout=10)
        response.raise_for_status()
        logger.info("Telegram-Nachricht erfolgreich gesendet.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Fehler beim Senden der Telegram-Nachricht: {e}")
