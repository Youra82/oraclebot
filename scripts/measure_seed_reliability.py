# scripts/measure_seed_reliability.py
# Misst, wie zuverlaessig das aktuelle Trainings-Rezept (settings.json) den Trend-Kollaps
# vermeidet -- statt der bisherigen Anekdote ("hat schon oefter geklappt") eine echte Rate:
# N Trainingslaeufe mit identischer Konfiguration (nur der Zufalls-Seed unterscheidet sich,
# da kein fester Seed gesetzt wird), pro Lauf wird geprueft ob die trend-Vorhersage auf dem
# OOS-Validierungsset genuin divers ist oder auf eine Klasse kollabiert.
#
# WICHTIG: ueberschreibt waehrend der Messung wiederholt market_transformer_best.pt/
# scaler_full.pkl (train_transformer.py schreibt immer an denselben Pfad) -- der aktuell
# committete/funktionierende Checkpoint wird vorher gesichert und am Ende wiederhergestellt,
# damit die Messung den Produktions-Stand nicht dauerhaft veraendert.
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')
CHECKPOINT_PATH = os.path.join(ARTIFACTS_DIR, 'market_transformer_best.pt')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler_full.pkl')
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'artifacts', 'results', 'seed_reliability.json')

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'


def _diagnose_checkpoint() -> dict:
    """Prueft das GERADE trainierte Checkpoint auf trend-Diversitaet -- laedt frisch von

    Platte (nicht aus dem Trainingsprozess uebernommen), damit die Messung unabhaengig vom
    Trainingslauf selbst ist.
    """
    import torch
    from collections import Counter
    from backtest_signal import load_settings, load_model_and_scaler, load_val_examples_by_symbol, load_ohlcv_by_symbol

    settings = load_settings()
    device = torch.device('cpu')
    model, scaler = load_model_and_scaler(settings, device)
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']
    ds_cfg = settings['dataset_settings']
    symbol = ds_cfg['symbols'][0]
    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in ds_cfg['symbols'])
    dataset_path = os.path.join(ARTIFACTS_DIR, f"{symbols_tag}_full.jsonl")

    from oraclebot.strategy.backtest import predict_for_examples

    val_by_symbol = load_val_examples_by_symbol(dataset_path, train_cfg['val_split'])
    examples = val_by_symbol[symbol]
    ohlcv_by_symbol = load_ohlcv_by_symbol([symbol], model_cfg['timeframes'], train_cfg['history_days'])
    preds = predict_for_examples(examples, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
                                  beam_width=model_cfg['beam_width'], device=device)
    trend_counts = Counter(p['prediction']['trend'] for p in preds)
    correct = sum(1 for p, ex in zip(preds, examples) if p['prediction']['trend'] == ex['target']['trend'])
    n_classes_used = len(trend_counts)
    accuracy = correct / len(preds) if preds else 0.0
    return {
        'collapsed': n_classes_used < 2,
        'trend_distribution': dict(trend_counts),
        'trend_accuracy': accuracy,
        'n_examples': len(preds),
    }


def _parse_best_epoch(stdout: str) -> dict:
    best_epoch, best_val_acc, verdict = None, None, None
    for line in stdout.splitlines():
        match = re.search(r'Bestes Ergebnis:\s*Epoche\s*(\d+)\s*mit\s*([\d.]+)%', line)
        if match:
            best_epoch = int(match.group(1))
            best_val_acc = float(match.group(2))
        if '-> Robust' in line or '-> Grenzwertig' in line or '-> Overfitted' in line:
            verdict = line.strip().split('->')[-1].strip()
    return {'best_epoch': best_epoch, 'best_val_acc': best_val_acc, 'verdict': verdict}


def main(n_runs: int, symbol: str, history_days: int, tag: str = ''):
    from backtest_signal import load_settings
    cfg_snapshot = load_settings()['training_settings']
    results_path = RESULTS_PATH if not tag else RESULTS_PATH.replace('.json', f'_{tag}.json')
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"{RED}Kein bestehender Checkpoint zum Sichern gefunden -- Messung wuerde den "
              f"aktuellen Produktions-Stand nicht wiederherstellen koennen. Abgebrochen.{NC}")
        return

    backup_ckpt = CHECKPOINT_PATH + '.reliability_backup'
    backup_scaler = SCALER_PATH + '.reliability_backup'
    shutil.copy(CHECKPOINT_PATH, backup_ckpt)
    shutil.copy(SCALER_PATH, backup_scaler)
    print(f"{CYAN}Produktions-Checkpoint gesichert. Starte {n_runs} Trainingslaeufe...{NC}\n")

    results = []
    try:
        for run_idx in range(1, n_runs + 1):
            print(f"{BOLD}=== Lauf {run_idx}/{n_runs} ==={NC}")
            proc = subprocess.run(
                [sys.executable, 'scripts/train_transformer.py', '--symbol', symbol,
                 '--history-days', str(history_days)],
                cwd=PROJECT_ROOT, capture_output=True, text=True,
            )
            if proc.returncode != 0:
                print(f"{RED}Lauf {run_idx} abgebrochen (Fehler):{NC}")
                print(proc.stderr[-1500:])
                results.append({'run': run_idx, 'error': True})
                continue

            # logger.info() (train_transformer.py) schreibt nach stderr, nicht stdout (Python-
            # logging-Standardverhalten) -- beide zusammen scannen, sonst wird nichts gefunden.
            training_info = _parse_best_epoch(proc.stdout + '\n' + proc.stderr)
            diagnosis = _diagnose_checkpoint()
            epoch_diagnostics = []
            diagnostics_path = os.path.join(ARTIFACTS_DIR, 'training_diagnostics.json')
            if os.path.exists(diagnostics_path):
                with open(diagnostics_path, encoding='utf-8') as f:
                    epoch_diagnostics = json.load(f)
            row = {'run': run_idx, 'error': False, **training_info, **diagnosis, 'epoch_diagnostics': epoch_diagnostics}
            results.append(row)

            status = f"{RED}KOLLABIERT{NC}" if diagnosis['collapsed'] else f"{GREEN}divers{NC}"
            print(f"  Bestes Epoche: {training_info['best_epoch']} | Verdict: {training_info['verdict']} | "
                  f"Status: {status} | Trend-Accuracy: {diagnosis['trend_accuracy']:.1%} | "
                  f"Verteilung: {diagnosis['trend_distribution']}\n")
    finally:
        shutil.move(backup_ckpt, CHECKPOINT_PATH)
        shutil.move(backup_scaler, SCALER_PATH)
        print(f"{CYAN}Produktions-Checkpoint wiederhergestellt.{NC}\n")

    valid_results = [r for r in results if not r.get('error')]
    n_collapsed = sum(1 for r in valid_results if r['collapsed'])
    n_valid = len(valid_results)
    collapse_rate = (n_collapsed / n_valid * 100) if n_valid else 0.0
    diverse_accuracies = [r['trend_accuracy'] for r in valid_results if not r['collapsed']]

    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"{BOLD}Seed-Zuverlaessigkeit ({symbol}, {history_days} Tage, {n_runs} Laeufe){NC}")
    print(f"{BOLD}{'=' * 70}{NC}")
    print(f"Kollaps-Rate: {n_collapsed}/{n_valid} ({collapse_rate:.0f}%)")
    if diverse_accuracies:
        print(f"Trend-Accuracy bei nicht-kollabierten Laeufen: "
              f"min={min(diverse_accuracies):.1%}, max={max(diverse_accuracies):.1%}, "
              f"mean={sum(diverse_accuracies)/len(diverse_accuracies):.1%}")
    print(f"{BOLD}{'=' * 70}{NC}")

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now(timezone.utc).isoformat(), 'symbol': symbol,
            'history_days': history_days, 'n_runs': n_runs, 'collapse_rate_pct': collapse_rate,
            'config_snapshot': {
                'diversity_weight': cfg_snapshot.get('diversity_weight'),
                'fusion_lr_multiplier': cfg_snapshot.get('fusion_lr_multiplier'),
                'batch_size': cfg_snapshot.get('batch_size'),
                'boosted_targets': cfg_snapshot.get('boosted_targets'),
            },
            'runs': results,
        }, f, indent=2, default=str)
    print(f"Ergebnisse gespeichert: {results_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-runs', type=int, default=5)
    parser.add_argument('--symbol', type=str, default='BTC/USDT:USDT')
    parser.add_argument('--history-days', type=int, default=900)
    parser.add_argument('--tag', type=str, default='', help='Suffix fuer die Ergebnisdatei, um Experimente nicht zu ueberschreiben')
    args = parser.parse_args()
    main(args.n_runs, args.symbol, args.history_days, args.tag)
