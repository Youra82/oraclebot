# src/oraclebot/utils/config.py
# Laedt settings.json (strukturelle Einstellungen: Symbol, Timeframes, Feature-Fenster, Hebel,
# Live-Trading-Schalter, ...) UND die aktive Coin/Timeframe-Strategie-Config (vom Optimizer
# gefundene Parameter: min_confidence, model_max_depth, Anti-Martingale-Werte) und mischt sie zu
# einem einzigen barrier_strategy_settings-Dict. Analog zu dnabot/zerobots
# configs/config_<symbol>_<timeframe>.json-Muster, aber deutlich einfacher (nur EIN Symbol/
# Timeframe aktiv, kein active_strategies-Array noetig).
#
# Bewusste Trennung: NUR die Parameter, die scripts/optimize_barrier_model.py tatsaechlich
# systematisch sucht, leben in der Config-Datei. Alles andere (Symbol, Kontext-Timeframes,
# Feature-Fenster, Hebel, Margin-Modus, Live-Trading-Schalter, Startkapital, Gebuehren-Annahme)
# bleibt in settings.json -- das sind entweder Strategie-Grundentscheidungen oder
# Methodik-/Betriebsparameter, keine per Backtest optimierbaren Werte (siehe README).
import json
import os

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'src', 'oraclebot', 'strategy', 'configs')

# Diese Schluessel werden von der Coin/Timeframe-Config ueberschrieben, falls vorhanden --
# alles andere kommt ausschliesslich aus settings.json.
STRATEGY_CONFIG_KEYS = [
    'min_confidence', 'model_max_depth',
    'anti_martingale_base_pct', 'anti_martingale_growth_factor', 'anti_martingale_streak_target',
]


def config_filename(symbol: str, reference_timeframe: str) -> str:
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    return f"config_{safe_symbol}_{reference_timeframe}.json"


def load_settings(settings_path: str = None) -> dict:
    settings_path = settings_path or os.path.join(PROJECT_ROOT, 'settings.json')
    with open(settings_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_strategy_config(symbol: str, reference_timeframe: str) -> dict:
    """Laedt die Coin/Timeframe-Config-Datei (ohne '_meta'), oder ein leeres Dict, falls sie
    (noch) nicht existiert -- z.B. ganz frisches Setup, bevor optimize_barrier_model.py je
    gelaufen ist. settings.json's eigene Standardwerte bleiben dann massgeblich."""
    config_path = os.path.join(CONFIGS_DIR, config_filename(symbol, reference_timeframe))
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    cfg.pop('_meta', None)
    return cfg


def load_barrier_config(settings: dict = None, symbol: str = None, reference_timeframe: str = None) -> dict:
    """Gibt das gemischte barrier_strategy_settings-Dict zurueck: settings.json als Basis,
    STRATEGY_CONFIG_KEYS werden von der aktiven Coin/Timeframe-Config ueberschrieben (falls
    vorhanden). Alle bestehenden Skripte lesen weiterhin aus einem einzigen flachen Dict -- nur
    die Ladelogik hat sich geaendert, keine Aufrufer-seitigen Zugriffsmuster.

    `symbol`/`reference_timeframe` optional ueberschreibbar (z.B. train_barrier_model.py's
    `--symbol`/`--reference-timeframe`-CLI-Flags) -- die Config-Datei wird dann fuer die
    UEBERSCHRIEBENE Kombination geladen, nicht fuer die in settings.json hinterlegte."""
    settings = settings or load_settings()
    barrier_cfg = dict(settings['barrier_strategy_settings'])
    if symbol:
        barrier_cfg['symbol'] = symbol
    if reference_timeframe:
        barrier_cfg['reference_timeframe'] = reference_timeframe
    resolved_symbol = barrier_cfg.get('symbol', 'BTC/USDT:USDT')
    resolved_reference_tf = barrier_cfg.get('reference_timeframe', '4h')
    strategy_cfg = load_strategy_config(resolved_symbol, resolved_reference_tf)
    barrier_cfg.update(strategy_cfg)
    return barrier_cfg


def save_strategy_config(symbol: str, reference_timeframe: str, values: dict, meta: dict) -> str:
    """Schreibt die Coin/Timeframe-Config-Datei (vom Optimizer aufgerufen). `values` darf NUR
    STRATEGY_CONFIG_KEYS enthalten -- alles andere gehoert in settings.json, nicht hierher."""
    unexpected = set(values) - set(STRATEGY_CONFIG_KEYS)
    if unexpected:
        raise ValueError(f"save_strategy_config: unerwartete Schluessel {unexpected} -- "
                          f"nur {STRATEGY_CONFIG_KEYS} gehoeren in die Coin/Timeframe-Config.")
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, config_filename(symbol, reference_timeframe))
    payload = dict(values)
    payload['_meta'] = meta
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return config_path
