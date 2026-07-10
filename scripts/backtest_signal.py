# scripts/backtest_signal.py
# Walk-Forward-Backtest des Handelssignals auf den Out-of-Sample-Validierungsbeispielen.
# Modus 1: Einzel-Backtest (alle Symbole kombiniert)
# Modus 2: Manuelle Portfolio-Simulation (User waehlt Symbole)
# Modus 3: Automatische Optimierung (Grid-Search ueber Strategie-Parameter)
import argparse
import itertools
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.WARNING, format='%(message)s')  # ruhig: nur Backtest-Output soll erscheinen

import pandas as pd
import torch

from oraclebot.data.dataset import load_dataset_jsonl
from oraclebot.data.features import FEATURE_NAMES
from oraclebot.data.scaler import FeatureScaler
from oraclebot.model.transformer import MarketTransformer
from oraclebot.strategy.backtest import run_signal_backtest
from oraclebot.utils.data_fetch import fetch_all_timeframes

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')


def load_settings() -> dict:
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_model_and_scaler(settings: dict, device):
    model_cfg = settings['model_settings']
    ds_cfg = settings['dataset_settings']
    model = MarketTransformer(
        n_features=len(FEATURE_NAMES), timeframes=model_cfg['timeframes'], window_sizes=ds_cfg['window_sizes'],
        d_model=model_cfg['d_model'], nhead=model_cfg['nhead'], num_encoder_layers=model_cfg['num_encoder_layers'],
        dim_feedforward=model_cfg['dim_feedforward'], dropout=model_cfg['dropout'],
    ).to(device)
    checkpoint_path = os.path.join(ARTIFACTS_DIR, 'market_transformer_best.pt')
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(f"Kein Checkpoint gefunden: {checkpoint_path}. Erst train_transformer.py ausfuehren.")
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    scaler_path = os.path.join(ARTIFACTS_DIR, 'scaler_full.pkl')
    scaler = FeatureScaler.load(scaler_path)
    return model, scaler


def load_val_examples_by_symbol(dataset_path: str, val_split: float) -> dict:
    """Laedt die gepoolte JSONL und rekonstruiert denselben chronologischen Pro-Symbol-Split

    wie beim Training (siehe train_transformer.py) -- nur der Validierungsanteil, damit der
    Backtest auf echten Out-of-Sample-Daten laeuft, nicht auf auswendig gelernten.
    """
    examples = load_dataset_jsonl(dataset_path)
    by_symbol = {}
    for ex in examples:
        by_symbol.setdefault(ex['symbol'], []).append(ex)

    val_by_symbol = {}
    for symbol, exs in by_symbol.items():
        n_val = max(1, int(len(exs) * val_split))
        val_by_symbol[symbol] = exs[-n_val:]
    return val_by_symbol


def load_ohlcv_by_symbol(symbols: list, timeframes: list, history_days: int) -> dict:
    ohlcv_by_symbol = {}
    for symbol in symbols:
        ohlcv_by_symbol[symbol] = fetch_all_timeframes(symbol, timeframes, history_days, cache_dir=ARTIFACTS_DIR, use_cache=True)
    return ohlcv_by_symbol


def print_results_table(rows: list, title: str):
    df = pd.DataFrame(rows)
    pd.set_option('display.width', 1200)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(f"\n{BOLD}{'=' * 90}{NC}")
    print(f"{BOLD}{title}{NC}")
    print(f"{BOLD}{'=' * 90}{NC}")
    print(df.to_string(index=False))
    print(f"{BOLD}{'=' * 90}{NC}")


def result_row(label: str, result: dict) -> dict:
    color = GREEN if result['total_pnl_pct'] > 0 and result['max_drawdown_pct'] < 30 else \
        (YELLOW if result['total_pnl_pct'] > -10 else RED)
    return {
        'Portfolio': label,
        'Trades': result['trades_count'],
        'Kein Trade': result['skipped_no_trade'],
        'Win Rate %': round(result['win_rate'], 1),
        'PnL %': round(result['total_pnl_pct'], 2),
        'Max DD %': round(result['max_drawdown_pct'], 2),
        'Endkapital': round(result['end_capital'], 2),
    }


def run_mode_1(settings: dict, start_capital: float):
    """Einzel-Backtest: alle Symbole gemeinsam als ein Portfolio."""
    print(f"{CYAN}--- OracleBot Signal-Backtest (Einzel-Modus) ---{NC}")
    device = torch.device('cpu')
    model, scaler = load_model_and_scaler(settings, device)

    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']
    strategy_cfg = settings['strategy_settings']

    symbols = ds_cfg.get('symbols', ['BTC/USDT:USDT'])
    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in symbols)
    dataset_path = os.path.join(ARTIFACTS_DIR, f"{symbols_tag}_full.jsonl")
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Kein Datensatz gefunden: {dataset_path}. Erst train_transformer.py ausfuehren.")

    val_by_symbol = load_val_examples_by_symbol(dataset_path, train_cfg['val_split'])
    ohlcv_by_symbol = load_ohlcv_by_symbol(symbols, model_cfg['timeframes'], train_cfg['history_days'])

    all_val_examples = [ex for exs in val_by_symbol.values() for ex in exs]
    strategy_cfg = {**strategy_cfg, 'beam_width': model_cfg['beam_width']}
    result = run_signal_backtest(all_val_examples, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
                                  strategy_cfg, start_capital=start_capital)

    rows = [result_row(f"Alle ({', '.join(symbols)})", result)]
    print_results_table(rows, "OracleBot Signal-Backtest (Out-of-Sample)")
    return result


def run_mode_2(settings: dict, start_capital: float):
    """Manuelle Portfolio-Simulation: User waehlt Symbole aus."""
    print(f"{CYAN}--- OracleBot Signal-Backtest (Manuelle Symbol-Auswahl) ---{NC}")
    device = torch.device('cpu')
    model, scaler = load_model_and_scaler(settings, device)

    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']
    strategy_cfg = {**settings['strategy_settings'], 'beam_width': model_cfg['beam_width']}

    all_symbols = ds_cfg.get('symbols', ['BTC/USDT:USDT'])
    print(f"\n{BOLD}Verfuegbare Symbole:{NC}")
    for idx, s in enumerate(all_symbols, 1):
        print(f"  {idx}) {s}")
    raw = input(f"\nAuswahl (z.B. '1' oder '1,2') [Standard: alle]: ").strip()
    if raw:
        indices = [int(p) - 1 for p in raw.replace(',', ' ').split() if p.strip().lstrip('-').isdigit()]
        selected = [all_symbols[i] for i in indices if 0 <= i < len(all_symbols)]
    else:
        selected = all_symbols
    if not selected:
        print(f"{RED}Keine gueltigen Symbole ausgewaehlt.{NC}")
        return None

    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in all_symbols)
    dataset_path = os.path.join(ARTIFACTS_DIR, f"{symbols_tag}_full.jsonl")
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Kein Datensatz gefunden: {dataset_path}. Erst train_transformer.py ausfuehren.")

    val_by_symbol = load_val_examples_by_symbol(dataset_path, train_cfg['val_split'])
    ohlcv_by_symbol = load_ohlcv_by_symbol(selected, model_cfg['timeframes'], train_cfg['history_days'])

    rows = []
    for symbol in selected:
        examples = val_by_symbol.get(symbol, [])
        if not examples:
            continue
        result = run_signal_backtest(examples, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
                                      strategy_cfg, start_capital=start_capital)
        rows.append(result_row(symbol, result))

    combined_examples = [ex for s in selected for ex in val_by_symbol.get(s, [])]
    if len(selected) > 1 and combined_examples:
        combined_result = run_signal_backtest(combined_examples, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
                                               strategy_cfg, start_capital=start_capital)
        rows.append(result_row(f"Kombiniert ({len(selected)} Symbole)", combined_result))

    print_results_table(rows, "OracleBot Signal-Backtest (Manuelle Auswahl)")
    return rows


def run_mode_3(settings: dict, start_capital: float, max_dd_limit: float, min_trades: int = 15):
    """Automatische Optimierung: Grid-Search ueber min_trend_confidence / risk_reward.

    `min_trades`: Kombinationen mit weniger Trades werden nicht als "bestes Ergebnis"
    zugelassen, auch wenn ihr PnL/MaxDD gut aussieht -- sonst gewinnt zuverlaessig die
    Kombination, die praktisch nie handelt (0 Trades = 0% Verlust = "bestes" Ergebnis rein
    rechnerisch, aber nutzlos). Lehre aus pbot (2026-07-10): dessen Optuna-Optimierung prunt
    Trials mit zu wenigen Trades aus genau diesem Grund (dort min_trades=15, hier uebernommen).
    """
    print(f"{CYAN}--- OracleBot Signal-Backtest (Automatische Parameter-Optimierung) ---{NC}")
    device = torch.device('cpu')
    model, scaler = load_model_and_scaler(settings, device)

    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']

    symbols = ds_cfg.get('symbols', ['BTC/USDT:USDT'])
    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in symbols)
    dataset_path = os.path.join(ARTIFACTS_DIR, f"{symbols_tag}_full.jsonl")
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Kein Datensatz gefunden: {dataset_path}. Erst train_transformer.py ausfuehren.")

    val_by_symbol = load_val_examples_by_symbol(dataset_path, train_cfg['val_split'])
    ohlcv_by_symbol = load_ohlcv_by_symbol(symbols, model_cfg['timeframes'], train_cfg['history_days'])
    all_val_examples = [ex for exs in val_by_symbol.values() for ex in exs]

    confidence_grid = [0.35, 0.40, 0.45, 0.50]
    risk_reward_grid = [1.5, 2.0, 3.0]

    print(f"\nTeste {len(confidence_grid) * len(risk_reward_grid)} Parameter-Kombinationen "
          f"(min_trend_confidence x risk_reward)...")

    rows = []
    best = None
    for min_conf, rr in itertools.product(confidence_grid, risk_reward_grid):
        strategy_cfg = {
            'min_trend_confidence': min_conf, 'sl_range_fraction': settings['strategy_settings']['sl_range_fraction'],
            'risk_reward': rr, 'risk_per_trade_pct': settings['strategy_settings']['risk_per_trade_pct'],
            'beam_width': model_cfg['beam_width'],
        }
        result = run_signal_backtest(all_val_examples, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
                                      strategy_cfg, start_capital=start_capital)
        label = f"conf={min_conf:.2f} rr={rr:.1f}"
        rows.append(result_row(label, result))

        if result['max_drawdown_pct'] <= max_dd_limit and result['trades_count'] >= min_trades:
            if best is None or result['total_pnl_pct'] > best[1]['total_pnl_pct']:
                best = (strategy_cfg, result)

    rows.sort(key=lambda r: r['PnL %'], reverse=True)
    print_results_table(rows, f"OracleBot Parameter-Optimierung (Max-DD-Limit: {max_dd_limit}%, Min-Trades: {min_trades})")

    if best:
        cfg, result = best
        print(f"\n{GREEN}Bestes Ergebnis innerhalb Max-DD-Limit und Min-Trades:{NC} min_trend_confidence={cfg['min_trend_confidence']}, "
              f"risk_reward={cfg['risk_reward']} -> PnL={result['total_pnl_pct']:.2f}%, MaxDD={result['max_drawdown_pct']:.2f}%, "
              f"Trades={result['trades_count']}")
    else:
        print(f"\n{YELLOW}Keine Kombination blieb innerhalb von Max-DD-Limit und Min-Trades "
              f"(Max-DD<={max_dd_limit}%, Trades>={min_trades}).{NC}")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=int, choices=[1, 2, 3], default=1)
    parser.add_argument('--capital', type=float, default=100.0)
    parser.add_argument('--max-dd', type=float, default=30.0, help='Nur fuer Modus 3')
    parser.add_argument('--min-trades', type=int, default=15, help='Nur fuer Modus 3')
    args = parser.parse_args()

    settings = load_settings()
    if args.mode == 1:
        run_mode_1(settings, args.capital)
    elif args.mode == 2:
        run_mode_2(settings, args.capital)
    elif args.mode == 3:
        run_mode_3(settings, args.capital, args.max_dd, args.min_trades)
