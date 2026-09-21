# src/oraclebot/utils/config.py
# Laedt settings.json.
import json
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')


def load_settings(settings_path: str = None) -> dict:
    settings_path = settings_path or os.path.join(PROJECT_ROOT, 'settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        return json.load(f)
