# scripts/optimize_barrier_model.py
# Systematische Parametersuche fuer das Barriere-Modell: model_max_depth, min_confidence,
# Anti-Martingale (base_pct/growth_factor/streak_target) -- dieselbe Methodik, mit der diese
# Werte urspruenglich in der Recherche vom 2026-07-24 bis 2026-07-26 gefunden wurden
# (Walk-Forward-Vergleich, Sensitivitaets-Sweep, gebuehren-bewusste Bootstrap-Kalibrierung),
# jetzt als wiederholbares, git-getracktes Skript statt Ad-hoc-Scratch-Code.
#
# STRIKTE OOS-DISZIPLIN: Alle Parameter werden AUSSCHLIESSLICH anhand von Walk-Forward-
# Out-of-Fold-Vorhersagen ausgewaehlt (siehe evaluation.walk_forward_predictions). Der
# offizielle 70/30-Out-of-Sample-Split wird NIE zur Parameterwahl herangezogen -- nur fuer
# einen einmaligen Bestaetigungs-Bericht ganz am Ende, der selbst keine weitere Parameterwahl
# mehr beeinflusst. Andernfalls waere der OOS-Split nicht mehr wirklich "ungesehen"
# (Mensch-im-Loop-Overfitting, siehe training_history.py und README).
#
# Bewusst NICHT Teil der Suche (siehe README "Wichtige Regeln"): leverage, margin_mode,
# live_trading_enabled, history_days, val_split, backtest_start_capital, taker_fee_rate_pct,
# num_threads, Feature-Fenster (feature_settings/feature_settings_by_timeframe) -- entweder
# Strategie-Grundentscheidungen, Methodik-/Betriebsparameter, oder ein zu grosser Suchraum fuer
# die aktuelle Datenmenge (Overfitting-Risiko). leverage/backtest_start_capital/
# reference_timeframe sind trotzdem als Lauf-Parameter ueberschreibbar (siehe optimize.sh),
# weil sie die Anti-Martingale-Kalibrierung/den Datensatz beeinflussen, ohne selbst gesucht zu werden.
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

from oraclebot.analysis.evaluation import (
    build_trades, calibrate_anti_martingale_base_pct, evaluate_walk_forward,
    run_anti_martingale_backtest, walk_forward_predictions,
)
from oraclebot.data.barrier_targets import build_barrier_examples
from oraclebot.data.dataset import save_dataset_jsonl
from oraclebot.data.scaler import FeatureScaler
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.utils.config import load_barrier_config, load_settings, save_strategy_config
from oraclebot.utils.data_fetch import fetch_all_timeframes
from oraclebot.utils import training_history

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')

MAX_DEPTH_CANDIDATES = [2, 3, 4, 5, 6]
MIN_CONFIDENCE_CANDIDATES = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
# Sicherheitsmarge ueber der Gebuehren-Breakeven-Grenze (56.0%, siehe README) -- niedrigste
# Schwelle, die diese Marge noch haelt, gewinnt (maximiert Trades unter der Nebenbedingung).
MIN_CONFIDENCE_WINRATE_FLOOR = 0.70
STREAK_TARGET_CANDIDATES = [2, 3, 4, 5]
GROWTH_FACTOR_CANDIDATES = [1.5, 2.0, 3.0]


def select_best_max_depth(examples: list, n_folds: int = 8) -> dict:
    logger.info('=' * 70)
    logger.info('1) MODEL_MAX_DEPTH (Walk-Forward, Auswahl nach Worst-Case vor Mittelwert)')
    logger.info('=' * 70)
    results = []
    for depth in MAX_DEPTH_CANDIDATES:
        # Kein Fortschrittsbalken hier (jeder Fold-Fit dauert je nach CPU Sekunden bis Minuten) --
        # ohne diese Zeile sieht ein Nutzer, der live zuschaut, minutenlang gar nichts (auf einem
        # langsameren VPS beobachtet 2026-07-26).
        logger.info(f"  Rechne depth={depth} ({n_folds - 1} Walk-Forward-Fenster)...")
        wf = evaluate_walk_forward(examples, n_folds=n_folds, max_depth=depth)
        results.append({'max_depth': depth, **wf})
        logger.info(f"  depth={depth}: Mittel={wf['mean']:.1%} Worst-Case={wf['worst_case']:.1%}")
    best_by_score = max(results, key=lambda r: (r['worst_case'], r['mean']))
    # Ein-Standardfehler-Regel (Occam's Razor): mit nur wenigen Walk-Forward-Test-Folds liegen
    # mehrere Tiefen oft innerhalb der Rausch-Bandbreite. Ein tieferer Baum ohne echten
    # Out-of-Fold-Gewinn erhoeht nur das Overfitting-Risiko (In-Sample-Genauigkeit naehert sich
    # 100%, siehe README "Wichtige Regeln"). Deshalb unter allen Tiefen, deren Worst-Case UND
    # Mittel innerhalb eines Standardfehlers des Bestwerts liegen, die FLACHSTE waehlen.
    accs = np.array(best_by_score['accuracies'])
    se = accs.std(ddof=1) / np.sqrt(len(accs)) if len(accs) > 1 else 0.0
    tolerant = [r for r in results
                if r['worst_case'] >= best_by_score['worst_case'] - se
                and r['mean'] >= best_by_score['mean'] - se]
    best = min(tolerant, key=lambda r: r['max_depth'])
    logger.info(f"  (Standardfehler={se:.1%} -- {len(tolerant)}/{len(results)} Tiefen statistisch "
                f"nicht vom Bestwert unterscheidbar, waehle die flachste davon)")
    logger.info(f"-> Gewaehlt: model_max_depth={best['max_depth']} "
                f"(Worst-Case={best['worst_case']:.1%}, Mittel={best['mean']:.1%})")
    return best


def select_min_confidence(wf_preds: list, barrier_cfg: dict) -> dict:
    logger.info('')
    logger.info('=' * 70)
    logger.info('2) MIN_CONFIDENCE (Sensitivitaet auf Walk-Forward-Out-of-Fold-Vorhersagen)')
    logger.info('=' * 70)
    results = []
    for thresh in MIN_CONFIDENCE_CANDIDATES:
        trades = build_trades(wf_preds, barrier_cfg, min_confidence=thresh)
        if not trades:
            continue
        wins = sum(1 for t in trades if t['outcome'] == 'win')
        win_rate = wins / len(trades)
        results.append({'min_confidence': thresh, 'n_trades': len(trades), 'win_rate': win_rate})
        logger.info(f"  Schwelle={thresh:.2f}: Trades={len(trades)} WinRate={win_rate:.1%}")

    candidates = [r for r in results if r['win_rate'] >= MIN_CONFIDENCE_WINRATE_FLOOR]
    if candidates:
        best = min(candidates, key=lambda r: r['min_confidence'])
    else:
        best = max(results, key=lambda r: r['win_rate'])
        logger.warning(f"  Keine Schwelle erreicht die {MIN_CONFIDENCE_WINRATE_FLOOR:.0%}-Sicherheitsmarge -- "
                       f"nehme die mit der hoechsten Winrate ({best['min_confidence']:.2f}).")
    logger.info(f"-> Gewaehlt: min_confidence={best['min_confidence']:.2f} "
                f"(Trades={best['n_trades']}, WinRate={best['win_rate']:.1%})")
    return best


def select_anti_martingale(wf_preds: list, barrier_cfg: dict, min_confidence: float, dd_limit: float,
                            dd_percentile: float = 90.0) -> dict:
    logger.info('')
    logger.info('=' * 70)
    logger.info(f'3) ANTI-MARTINGALE (Bootstrap-Kalibrierung auf Walk-Forward-Trades, '
                f'Ziel: {dd_percentile:.0f}.-Perzentil-MaxDD <= {dd_limit:.0f}%)')
    logger.info('=' * 70)
    trades = build_trades(wf_preds, barrier_cfg, min_confidence=min_confidence)
    win_rate = sum(1 for t in trades if t['outcome'] == 'win') / len(trades)
    logger.info(f"  Grundlage: {len(trades)} Walk-Forward-Trades (WinRate={win_rate:.1%})")

    results = []
    for streak_target in STREAK_TARGET_CANDIDATES:
        for growth_factor in GROWTH_FACTOR_CANDIDATES:
            # Jede Kombination lost tausende Bootstrap-Trade-Sequenzen (22 Bisektionsschritte +
            # finale Praezisionsrunde) -- ohne diese Zeile bleibt die Konsole minutenlang leer.
            logger.info(f"  Rechne streak={streak_target} growth={growth_factor:.1f}...")
            calib = calibrate_anti_martingale_base_pct(
                trades, barrier_cfg, growth_factor, streak_target,
                dd_percentile=dd_percentile, dd_limit=dd_limit)
            results.append({'streak_target': streak_target, 'growth_factor': growth_factor, **calib})
            logger.info(f"  streak={streak_target} growth={growth_factor:.1f}: base_pct={calib['base_pct']:.3f}% "
                        f"P50-DD={calib['p50_dd']:.1f}% P{dd_percentile:.0f}-DD={calib['p_dd']:.1f}% "
                        f"MedSkips={calib['median_skips']:.1f}")

    # MedSkips>0 heisst: bei diesem Startkapital/Hebel faellt der Einsatz oft unter Bitgets
    # Mindest-Notional -- degenerierte Kandidaten (siehe README, 2026-07-25-Fund) ausschliessen.
    viable = [r for r in results if r['median_skips'] == 0]
    if not viable:
        logger.warning("  Keine Kombination ohne uebersprungene Trades (Mindest-Notional) -- nehme alle.")
        viable = results
    best = max(viable, key=lambda r: r['p50_pnl'])
    logger.info(f"-> Gewaehlt: streak_target={best['streak_target']} growth_factor={best['growth_factor']:.1f} "
                f"base_pct={best['base_pct']:.3f}%")
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None)
    parser.add_argument('--reference-timeframe', default=None)
    parser.add_argument('--history-days', type=int, default=None)
    parser.add_argument('--leverage', type=float, default=None,
                         help='Ueberschreibt barrier_strategy_settings.leverage fuer diesen Optimierungslauf '
                              '(fliesst in die Anti-Martingale-Kalibrierung ein).')
    parser.add_argument('--start-capital', type=float, default=None,
                         help='Ueberschreibt backtest_start_capital fuer diesen Optimierungslauf.')
    parser.add_argument('--dd-limit', type=float, default=50.0,
                         help='Ziel-Obergrenze fuer das 90.-Perzentil-MaxDD (Bootstrap) der Anti-Martingale-Kalibrierung.')
    parser.add_argument('--no-cache', action='store_true')
    parser.add_argument('--apply', action='store_true',
                         help='Schreibt die gefundenen Werte in die Strategie-Config-Datei (sonst nur Bericht).')
    args = parser.parse_args()

    settings = load_settings()
    barrier_cfg = load_barrier_config(settings, symbol=args.symbol, reference_timeframe=args.reference_timeframe)
    if args.leverage is not None:
        barrier_cfg['leverage'] = args.leverage
    if args.start_capital is not None:
        barrier_cfg['backtest_start_capital'] = args.start_capital

    symbol = barrier_cfg['symbol']
    reference_tf = barrier_cfg['reference_timeframe']
    intraday_tf = barrier_cfg.get('intraday_timeframe', '15m')
    context_tfs = [tf for tf in barrier_cfg.get('context_timeframes', []) if tf != reference_tf]
    barrier_pct = barrier_cfg.get('barrier_pct', 1.0)
    history_days = args.history_days or barrier_cfg['history_days']

    logger.info("=" * 70)
    logger.info("ORACLEBOT PARAMETER-OPTIMIERUNG")
    logger.info("=" * 70)
    logger.info(f"Symbol={symbol} Referenz={reference_tf} Hebel={barrier_cfg['leverage']}x "
                f"Startkapital={barrier_cfg.get('backtest_start_capital', 15.0):.2f} USDT "
                f"DD-Ziel={args.dd_limit:.0f}%")
    logger.info("")

    all_tfs = sorted(set([reference_tf, intraday_tf] + context_tfs))
    ohlcv = fetch_all_timeframes(symbol, all_tfs, history_days, cache_dir=ARTIFACTS_DIR, use_cache=not args.no_cache)
    examples = build_barrier_examples(
        ohlcv, reference_timeframe=reference_tf, intraday_timeframe=intraday_tf,
        context_timeframes=context_tfs, feature_kwargs=barrier_cfg['feature_settings'],
        feature_kwargs_by_timeframe=barrier_cfg.get('feature_settings_by_timeframe', {}),
        barrier_pct=barrier_pct)
    logger.info(f"{len(examples)} Beispiele gebaut ({examples[0]['date']} bis {examples[-1]['date']})\n")
    if len(examples) < 200:
        raise RuntimeError(f"Nur {len(examples)} Beispiele -- zu wenig fuer eine robuste Parametersuche.")

    # --- 1) model_max_depth ---
    best_depth = select_best_max_depth(examples)
    max_depth = best_depth['max_depth']

    # --- Walk-Forward-Out-of-Fold-Vorhersagen fuer den gewaehlten depth (Basis fuer 2+3) ---
    wf_preds = walk_forward_predictions(examples, max_depth=max_depth, n_folds=8)

    # --- 2) min_confidence ---
    best_confidence = select_min_confidence(wf_preds, barrier_cfg)
    min_confidence = best_confidence['min_confidence']

    # --- 3) Anti-Martingale ---
    best_am = select_anti_martingale(wf_preds, barrier_cfg, min_confidence, dd_limit=args.dd_limit)

    logger.info('')
    logger.info('=' * 70)
    logger.info('GEFUNDENE PARAMETER (nur per Walk-Forward, OOS-Split unberuehrt)')
    logger.info('=' * 70)
    logger.info(f"model_max_depth = {max_depth}")
    logger.info(f"min_confidence = {min_confidence:.2f}")
    logger.info(f"anti_martingale_growth_factor = {best_am['growth_factor']:.1f}")
    logger.info(f"anti_martingale_streak_target = {best_am['streak_target']}")
    logger.info(f"anti_martingale_base_pct = {best_am['base_pct']:.3f}")

    # --- Finaler EINMALIGER Bestaetigungs-Lauf auf dem offiziellen 70/30-Split ---
    logger.info('')
    logger.info('=' * 70)
    logger.info('BESTAETIGUNG auf dem offiziellen 70/30-Out-of-Sample-Split '
                '(einmalig, beeinflusst keine weitere Parameterwahl)')
    logger.info('=' * 70)
    val_split = barrier_cfg['val_split']
    n_val = max(1, int(len(examples) * val_split))
    train_examples = examples[:-n_val]
    val_examples = examples[-n_val:]
    X_train = np.array([ex['features'] for ex in train_examples], dtype=np.float32)
    y_train = np.array([ex['target'] for ex in train_examples], dtype=int)
    X_val = np.array([ex['features'] for ex in val_examples], dtype=np.float32)
    y_val = np.array([ex['target'] for ex in val_examples], dtype=int)

    scaler = FeatureScaler().fit_array(X_train)
    predictor = BarrierPredictor(max_depth=max_depth).fit(X_train, y_train, scaler)
    train_acc = predictor.score(X_train, y_train)
    val_acc = predictor.score(X_val, y_val)
    logger.info(f"In-Sample={train_acc:.1%} Out-of-Sample={val_acc:.1%}")

    val_preds = []
    for ex in val_examples:
        cls, conf = predictor.predict_one(ex['features'])
        val_preds.append({'date': ex['date'], 'entry': ex['entry'], 'cls': cls, 'conf': conf,
                          'label': ex['target'], 'exit_time': pd.Timestamp(ex['exit_time'])})
    val_preds.sort(key=lambda p: p['date'])

    final_cfg = dict(barrier_cfg)
    final_cfg['min_confidence'] = min_confidence
    final_cfg['anti_martingale_base_pct'] = best_am['base_pct']
    final_cfg['anti_martingale_growth_factor'] = best_am['growth_factor']
    final_cfg['anti_martingale_streak_target'] = best_am['streak_target']

    confirm_trades = build_trades(val_preds, final_cfg)
    if confirm_trades:
        confirm_backtest = run_anti_martingale_backtest(confirm_trades, final_cfg)
        confirm_wins = sum(1 for t in confirm_trades if t['outcome'] == 'win')
        logger.info(f"Trades={len(confirm_trades)} WinRate={confirm_wins/len(confirm_trades):.1%} "
                    f"MaxDD={confirm_backtest['max_dd_pct']:.1f}%")
    else:
        logger.warning("Keine Trades im OOS-Split mit den gefundenen Parametern.")

    # --- Uebernehmen? (per --apply ODER interaktiver Bestaetigung) ---
    # Bewusst NACH der vollen (teuren) Berechnung gefragt, nicht vorher -- die Entscheidung
    # "uebernehmen?" braucht ja gerade die Ergebnisse oben. optimize.sh ruft dieses Skript
    # deshalb nur EINMAL auf; ein zweiter Aufruf mit --apply wuerde die gesamte Suche unnoetig
    # wiederholen.
    #
    # WICHTIG: Modell/Datensatz/Diagnose/Verlaufs-Log UND die Strategie-Config werden beide
    # NUR bei Bestaetigung geschrieben. predict_next_barrier.py laedt das Modell von genau
    # diesem Pfad fuer die naechste Live-Vorhersage -- ein reiner "zeig mir nur den Bericht"-
    # Lauf (should_apply=False) darf das produktive Modell auf keinen Fall stillschweigend
    # ersetzen (Bugfix 2026-07-26, siehe README).
    should_apply = args.apply
    if not should_apply and sys.stdin.isatty():
        # isatty() kann in manchen nicht-interaktiven Umgebungen (z.B. bestimmte CI-/Tool-
        # Shells) trotzdem True liefern, obwohl kein Input ankommt -- input() wuerde dann mit
        # EOFError abstuerzen. Deshalb zusaetzlich defensiv abfangen statt dem isatty()-Check
        # blind zu vertrauen.
        try:
            answer = input("\nGefundene Parameter uebernehmen (Modell neu speichern + Strategie-Config "
                            "schreiben)? (j/n) [n]: ").strip().lower()
            should_apply = answer in ('j', 'y')
        except EOFError:
            logger.info("\n(Kein interaktiver Input verfuegbar -- nichts geaendert. "
                        "Mit --apply erneut aufrufen, um zu uebernehmen.)")
            should_apply = False

    if not should_apply:
        logger.info("\nNicht uebernommen -- Modell und Strategie-Config unveraendert.")
    else:
        # --- Modell/Datensatz/Diagnose speichern (identisch zu train_barrier_model.py) ---
        symbols_tag = symbol.replace('/', '_').replace(':', '_')
        dataset_path = os.path.join(ARTIFACTS_DIR, f"barrier_{symbols_tag}_{reference_tf}.jsonl")
        save_dataset_jsonl(examples, dataset_path)
        model_path = os.path.join(ARTIFACTS_DIR, f"barrier_model_{symbols_tag}_{reference_tf}.pkl")
        predictor.save(model_path)

        # best_depth enthaelt bereits die Walk-Forward-Kennzahlen fuer den gewaehlten max_depth
        # (aus select_best_max_depth()) -- keine erneute Berechnung noetig.
        diagnostics_path = os.path.join(ARTIFACTS_DIR, f"barrier_diagnostics_{symbols_tag}_{reference_tf}.json")
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            json.dump({
                'symbol': symbol, 'reference_timeframe': reference_tf, 'intraday_timeframe': intraday_tf,
                'barrier_pct': barrier_pct, 'max_depth': max_depth, 'n_examples': len(examples),
                'n_train': len(train_examples), 'n_val': len(val_examples),
                'train_accuracy': train_acc, 'val_accuracy': val_acc,
                'walk_forward_accuracies': best_depth['accuracies'], 'walk_forward_mean': best_depth['mean'],
                'walk_forward_worst_case': best_depth['worst_case'],
            }, f, indent=2)
        logger.info(f"\nModell/Datensatz/Diagnose gespeichert (wie train_barrier_model.py).")

        history_path = os.path.join(ARTIFACTS_DIR, f"training_history_{symbols_tag}_{reference_tf}.jsonl")
        training_history.append_entry(history_path, final_cfg, val_acc, best_depth['mean'], best_depth['worst_case'])
        overfit_warning = training_history.check_overfitting_risk(history_path)
        if overfit_warning:
            logger.warning(f"\n{overfit_warning}")

        meta = {
            'optimized_at': pd.Timestamp.now(tz='UTC').isoformat(),
            'leverage_used': barrier_cfg['leverage'],
            'start_capital_used': barrier_cfg.get('backtest_start_capital', 15.0),
            'dd_limit_used': args.dd_limit,
            'walk_forward_mean': best_depth['mean'], 'walk_forward_worst_case': best_depth['worst_case'],
            'confirmation_oos_accuracy': val_acc,
        }
        values = {
            'min_confidence': min_confidence, 'model_max_depth': max_depth,
            'anti_martingale_base_pct': best_am['base_pct'],
            'anti_martingale_growth_factor': best_am['growth_factor'],
            'anti_martingale_streak_target': best_am['streak_target'],
        }
        config_path = save_strategy_config(symbol, reference_tf, values, meta)
        logger.info(f"\nStrategie-Config geschrieben: {config_path}")
        logger.info("Naechste Schritte: ./show_results.sh ansehen, dann ./push_configs.sh")
