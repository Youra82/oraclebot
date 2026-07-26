# scripts/train_barrier_model.py
# Trainiert das Barriere-Modell (barrier_targets.py + barrier_model.py): sagt fuer jede 4h-Kerze
# vorher, ob eine symmetrische +-barrier_pct%-Bewegung zuerst nach oben oder unten erreicht wird.
# Deutlich einfachere Pipeline als train_transformer.py (kein Transformer, kein Multi-Timeframe-
# Fenster) -- ein einzelnes HistGradientBoostingClassifier auf den 4h-Referenzkerzen-Features.
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from oraclebot.analysis.evaluation import evaluate_walk_forward
from oraclebot.data.barrier_targets import build_barrier_examples
from oraclebot.data.dataset import save_dataset_jsonl
from oraclebot.data.features import FEATURE_NAMES
from oraclebot.data.scaler import FeatureScaler
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.utils.config import load_barrier_config
from oraclebot.utils.data_fetch import fetch_all_timeframes
from oraclebot.utils import training_history

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None, help='Ueberschreibt barrier_strategy_settings.symbol')
    parser.add_argument('--history-days', type=int, default=None)
    parser.add_argument('--reference-timeframe', default=None,
                         help='Ueberschreibt barrier_strategy_settings.reference_timeframe (z.B. fuer '
                              'optimize_barrier_model.py). Beim Aendern context_timeframes in settings.json '
                              'im Auge behalten -- der neue Referenz-Timeframe sollte dort nicht doppelt '
                              'als Kontext-Block auftauchen.')
    parser.add_argument('--no-cache', action='store_true')
    args = parser.parse_args()

    barrier_cfg = load_barrier_config(symbol=args.symbol, reference_timeframe=args.reference_timeframe)

    symbol = barrier_cfg['symbol']
    history_days = args.history_days or barrier_cfg['history_days']
    reference_tf = barrier_cfg['reference_timeframe']
    intraday_tf = barrier_cfg.get('intraday_timeframe', '15m')
    # reference_tf aus context_timeframes ausschliessen -- relevant, wenn --reference-timeframe
    # einen Wert setzt, der in settings.json noch als Kontext-Timeframe gelistet ist (sonst
    # taucht dieselbe Zeitebene doppelt auf: einmal als Referenz-, einmal als Kontext-Block).
    context_tfs = [tf for tf in barrier_cfg.get('context_timeframes', []) if tf != reference_tf]
    barrier_pct = barrier_cfg.get('barrier_pct', 1.0)
    max_depth = barrier_cfg.get('model_max_depth', 3)
    val_split = barrier_cfg['val_split']

    all_tfs = sorted(set([reference_tf, intraday_tf] + context_tfs))
    logger.info(f"Lade {symbol}: Referenz={reference_tf}, Intraday={intraday_tf}, "
                f"Kontext={context_tfs}, {history_days} Tage...")
    ohlcv = fetch_all_timeframes(symbol, all_tfs, history_days,
                                  cache_dir=ARTIFACTS_DIR, use_cache=not args.no_cache)

    examples = build_barrier_examples(
        ohlcv, reference_timeframe=reference_tf, intraday_timeframe=intraday_tf,
        context_timeframes=context_tfs, feature_kwargs=barrier_cfg['feature_settings'],
        feature_kwargs_by_timeframe=barrier_cfg.get('feature_settings_by_timeframe', {}),
        barrier_pct=barrier_pct)
    logger.info(f"{len(examples)} Beispiele gebaut ({examples[0]['date']} bis {examples[-1]['date']})")
    if len(examples) < 50:
        raise RuntimeError(f"Nur {len(examples)} Beispiele -- zu wenig fuer ein sinnvolles Training.")

    symbols_tag = symbol.replace('/', '_').replace(':', '_')
    dataset_path = os.path.join(ARTIFACTS_DIR, f"barrier_{symbols_tag}_{reference_tf}.jsonl")
    save_dataset_jsonl(examples, dataset_path)

    n_val = max(1, int(len(examples) * val_split))
    train_examples = examples[:-n_val]
    val_examples = examples[-n_val:]
    logger.info(f"Train={len(train_examples)} Val={len(val_examples)}")

    X_train = np.array([ex['features'] for ex in train_examples], dtype=np.float32)
    y_train = np.array([ex['target'] for ex in train_examples], dtype=int)
    X_val = np.array([ex['features'] for ex in val_examples], dtype=np.float32)
    y_val = np.array([ex['target'] for ex in val_examples], dtype=int)

    logger.info(f"\nWalk-Forward-Robustheitscheck (8 Fenster ueber die gesamte Historie)...")
    wf = evaluate_walk_forward(examples, n_folds=8, max_depth=max_depth)
    logger.info(f"Walk-Forward: {['%.1f%%' % (a * 100) for a in wf['accuracies']]}")
    logger.info(f"Mittel={wf['mean']:.1%} Worst-Case={wf['worst_case']:.1%}")

    scaler = FeatureScaler().fit_array(X_train)
    predictor = BarrierPredictor(max_depth=max_depth).fit(X_train, y_train, scaler)

    train_acc = predictor.score(X_train, y_train)
    val_acc = predictor.score(X_val, y_val)
    logger.info(f"\nOffizieller 70/30-Split: In-Sample={train_acc:.1%} Out-of-Sample={val_acc:.1%}")

    model_path = os.path.join(ARTIFACTS_DIR, f"barrier_model_{symbols_tag}_{reference_tf}.pkl")
    predictor.save(model_path)
    logger.info(f"\nModell gespeichert: {model_path}")

    diagnostics_path = os.path.join(ARTIFACTS_DIR, f"barrier_diagnostics_{symbols_tag}_{reference_tf}.json")
    with open(diagnostics_path, 'w', encoding='utf-8') as f:
        json.dump({
            'symbol': symbol, 'reference_timeframe': reference_tf, 'intraday_timeframe': intraday_tf,
            'barrier_pct': barrier_pct, 'max_depth': max_depth, 'n_examples': len(examples),
            'n_train': len(train_examples), 'n_val': len(val_examples),
            'train_accuracy': train_acc, 'val_accuracy': val_acc,
            'walk_forward_accuracies': wf['accuracies'], 'walk_forward_mean': wf['mean'],
            'walk_forward_worst_case': wf['worst_case'],
        }, f, indent=2)
    logger.info(f"Diagnose gespeichert: {diagnostics_path}")

    history_path = os.path.join(ARTIFACTS_DIR, f"training_history_{symbols_tag}_{reference_tf}.jsonl")
    training_history.append_entry(history_path, barrier_cfg, val_acc, wf['mean'], wf['worst_case'])
    warning = training_history.check_overfitting_risk(history_path)
    if warning:
        logger.warning(f"\n{warning}")
