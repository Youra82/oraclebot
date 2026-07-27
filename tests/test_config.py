import json

import pytest

from oraclebot.utils import config as config_module
from oraclebot.utils.config import (
    config_filename, load_barrier_config, load_settings, load_strategy_config, save_strategy_config,
)


@pytest.fixture
def isolated_configs_dir(tmp_path, monkeypatch):
    """save_strategy_config/load_strategy_config schreiben/lesen ueber das Modul-Level
    CONFIGS_DIR -- fuer Tests auf ein temporaeres Verzeichnis umbiegen, damit die echte
    config_BTC_USDT_USDT_4h.json (aktiv fuer den Live-Bot) nie angefasst wird."""
    monkeypatch.setattr(config_module, 'CONFIGS_DIR', str(tmp_path))
    return tmp_path


def test_config_filename_replaces_slash_and_colon():
    assert config_filename('BTC/USDT:USDT', '4h') == 'config_BTC_USDT_USDT_4h.json'


def test_load_strategy_config_returns_empty_dict_if_file_missing(isolated_configs_dir):
    assert load_strategy_config('ETH/USDT:USDT', '1h') == {}


def test_save_then_load_strategy_config_roundtrip(isolated_configs_dir):
    values = {'min_confidence': 0.65, 'model_max_depth': 4}
    save_strategy_config('ETH/USDT:USDT', '1h', values, meta={'note': 'test'})
    loaded = load_strategy_config('ETH/USDT:USDT', '1h')
    assert loaded == values  # '_meta' muss entfernt sein


def test_save_strategy_config_rejects_unexpected_keys(isolated_configs_dir):
    with pytest.raises(ValueError):
        save_strategy_config('ETH/USDT:USDT', '1h', {'risk_per_trade_pct': 2.0}, meta={})


def test_save_strategy_config_writes_meta_block(isolated_configs_dir):
    path = save_strategy_config('ETH/USDT:USDT', '1h', {'min_confidence': 0.6}, meta={'source': 'unit-test'})
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    assert raw['_meta'] == {'source': 'unit-test'}


def test_load_barrier_config_merges_settings_with_strategy_config(isolated_configs_dir):
    settings = {
        'barrier_strategy_settings': {
            'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h',
            'leverage': 100, 'min_confidence': 0.60,
        }
    }
    save_strategy_config('BTC/USDT:USDT', '4h', {'min_confidence': 0.75}, meta={})
    merged = load_barrier_config(settings)
    assert merged['leverage'] == 100  # unveraendert aus settings.json
    assert merged['min_confidence'] == 0.75  # von der Strategie-Config ueberschrieben


def test_load_barrier_config_falls_back_to_settings_when_no_strategy_config_exists(isolated_configs_dir):
    settings = {
        'barrier_strategy_settings': {
            'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h', 'min_confidence': 0.60,
        }
    }
    merged = load_barrier_config(settings)
    assert merged['min_confidence'] == 0.60


def test_load_barrier_config_respects_symbol_and_reference_timeframe_override(isolated_configs_dir):
    settings = {
        'barrier_strategy_settings': {
            'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h', 'min_confidence': 0.60,
        }
    }
    save_strategy_config('ETH/USDT:USDT', '1h', {'min_confidence': 0.80}, meta={})
    merged = load_barrier_config(settings, symbol='ETH/USDT:USDT', reference_timeframe='1h')
    assert merged['symbol'] == 'ETH/USDT:USDT'
    assert merged['reference_timeframe'] == '1h'
    assert merged['min_confidence'] == 0.80


def test_load_settings_reads_real_settings_json():
    settings = load_settings()
    assert 'barrier_strategy_settings' in settings
    assert 'symbol' in settings['barrier_strategy_settings']
