# src/oraclebot/analysis/evaluation.py
# Gemeinsame Backtest-/Walk-Forward-/Bootstrap-Logik, genutzt von train_barrier_model.py,
# scripts/show_results.py UND scripts/optimize_barrier_model.py -- ausgelagert, damit alle drei
# exakt denselben Code nutzen (keine Kopien, die im Laufe der Zeit auseinanderdriften koennten).
import logging

import numpy as np
import pandas as pd

from oraclebot.data.scaler import FeatureScaler
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.strategy.anti_martingale import compute_margin, record_pending_position, resolve_pending_outcome
from oraclebot.strategy.barrier_signal import compute_barrier_signal

logger = logging.getLogger(__name__)


def evaluate_walk_forward(examples: list, n_folds: int = 8, max_depth: int = 3) -> dict:
    """Robustheits-Check ueber mehrere chronologische Fenster (nicht nur den finalen 70/30-Split)
    -- siehe Recherche 2026-07-24: 8 Fenster ueber 2.5 Jahre zeigten 62.0-71.2% Accuracy,
    Std-Abw. nur 3pp. Wird bei jedem Training mit ausgegeben, damit ein Genauigkeits-Einbruch
    (z.B. durch veraendertes Marktregime) sofort auffaellt. Dient auch als Kern-Kriterium fuer
    optimize_barrier_model.py's model_max_depth-Vergleich -- bewusst NICHT der finale 70/30-Split,
    um den Out-of-Sample-Holdout bei der Parameterwahl strikt ungesehen zu halten."""
    X_all = np.array([ex['features'] for ex in examples], dtype=np.float32)
    y_all = np.array([ex['target'] for ex in examples], dtype=int)
    n = len(examples)
    fold_bounds = [int(n * i / n_folds) for i in range(n_folds + 1)]
    accs = []
    for fi in range(1, n_folds):
        ts_, te_ = fold_bounds[fi], fold_bounds[fi + 1]
        if ts_ < 10 or te_ - ts_ < 5:
            continue
        scaler = FeatureScaler().fit_array(X_all[:ts_])
        predictor = BarrierPredictor(max_depth=max_depth).fit(X_all[:ts_], y_all[:ts_], scaler)
        acc = predictor.score(X_all[ts_:te_], y_all[ts_:te_])
        accs.append(acc)
    return {'accuracies': accs, 'mean': float(np.mean(accs)), 'worst_case': float(np.min(accs))}


def walk_forward_predictions(examples: list, max_depth: int = 3, n_folds: int = 8) -> list:
    """Wie evaluate_walk_forward(), aber gibt statt der reinen Accuracy pro Fold die
    tatsaechlichen Modell-Vorhersagen (Klasse+Konfidenz) fuer JEDES Beispiel zurueck, das in
    einem Test-Fold lag (Folds 1..n_folds-1 -- Fold 0 dient nur als anfaengliches
    Trainingsfenster ohne eigene Vorhersagen). Dieselbe Dict-Form wie show_results.py's
    load_predictions() (date/entry/cls/conf/label/exit_time), also direkt mit build_trades()
    kompatibel.

    Grundlage fuer optimize_barrier_model.py's min_confidence-/Anti-Martingale-Sweeps: haelt
    den finalen 70/30-Out-of-Sample-Split komplett unberuehrt bei der Parameterwahl (der wird
    dort ausschliesslich fuer einen einmaligen, nicht-parameterbeeinflussenden
    Bestaetigungs-Bericht verwendet) -- sonst waere der OOS-Split nicht mehr wirklich
    "ungesehen" (Mensch-im-Loop-Overfitting, siehe training_history.py und README)."""
    X_all = np.array([ex['features'] for ex in examples], dtype=np.float32)
    y_all = np.array([ex['target'] for ex in examples], dtype=int)
    n = len(examples)
    fold_bounds = [int(n * i / n_folds) for i in range(n_folds + 1)]
    preds = []
    for fi in range(1, n_folds):
        ts_, te_ = fold_bounds[fi], fold_bounds[fi + 1]
        if ts_ < 10 or te_ - ts_ < 5:
            continue
        scaler = FeatureScaler().fit_array(X_all[:ts_])
        predictor = BarrierPredictor(max_depth=max_depth).fit(X_all[:ts_], y_all[:ts_], scaler)
        proba = predictor.predict_proba(X_all[ts_:te_])
        for i, ex in enumerate(examples[ts_:te_]):
            cls = int(np.argmax(proba[i]))
            conf = float(proba[i][cls])
            preds.append({'date': ex['date'], 'entry': ex['entry'], 'cls': cls, 'conf': conf,
                          'label': ex['target'], 'exit_time': pd.Timestamp(ex['exit_time'])})
    preds.sort(key=lambda p: p['date'])
    return preds


def build_trades(preds: list, barrier_cfg: dict, min_confidence: float = None) -> list:
    """Baut serielle Trades aus den Modell-Vorhersagen: nur Signale >= min_confidence, ein Trade
    laeuft bis zu seinem eigenen exit_time, alle Referenzkerzen darin werden uebersprungen (kein
    Stacking, wie live_trade.py es auch nicht erlaubt). `min_confidence` optional ueberschreibbar
    (fuer Schwellen-Sweeps in optimize_barrier_model.py, ohne barrier_cfg zu mutieren)."""
    barrier_pct = barrier_cfg['barrier_pct']
    # .get() mit Fallback (nicht barrier_cfg['min_confidence']): min_confidence lebt seit dem
    # Coin/Timeframe-Config-Umbau in der optionalen Strategie-Config-Datei (siehe config.py),
    # nicht mehr garantiert in settings.json -- fehlt sie (z.B. nach einem run_pipeline.sh-Reset
    # vor dem ersten optimize_barrier_model.py-Lauf), soll ein sinnvoller Standard greifen statt
    # eines KeyError, konsistent mit predict_next_barrier.py's eigenem Fallback.
    min_conf = min_confidence if min_confidence is not None else barrier_cfg.get('min_confidence', 0.60)
    trades = []
    idx = 0
    while idx < len(preds):
        p = preds[idx]
        signal = compute_barrier_signal(p['cls'], p['conf'], p['entry'],
                                         min_confidence=min_conf, barrier_pct=barrier_pct)
        if signal['direction'] is None:
            idx += 1
            continue
        won = (p['cls'] == p['label'])
        pnl_price = signal['tp_distance'] if won else -signal['sl_distance']
        exit_price = signal['take_profit'] if won else signal['stop_loss']
        trades.append({
            'entry_time': pd.Timestamp(p['date']), 'exit_time': p['exit_time'], 'entry': p['entry'],
            'exit': exit_price, 'direction': signal['direction'], 'frac': pnl_price / p['entry'],
            'outcome': 'win' if won else 'loss',
        })
        while idx < len(preds) and preds[idx]['date'] <= p['exit_time'].isoformat():
            idx += 1
    return trades


def run_anti_martingale_backtest(trades: list, barrier_cfg: dict, base_pct: float = None,
                                  growth_factor: float = None, streak_target: int = None) -> dict:
    """Simuliert die Anti-Martingale-Positionsgroesse (echte Module aus anti_martingale.py) MIT
    Taker-Gebuehren (settings.json: taker_fee_rate_pct, Entry+Exit = 2 Taker-Fills) -- ohne
    Gebuehren wirkt jede Kalibrierung bei 100x Hebel deutlich zu optimistisch (siehe README,
    Rekalibrierung 2026-07-26). Schreibt margin_used/pnl_usdt/equity_after direkt in die
    trades-Dicts (fuer den Excel-Export weiterverwendbar). base_pct/growth_factor/streak_target
    optional ueberschreibbar (fuer Parameter-Sweeps in optimize_barrier_model.py, ohne
    barrier_cfg zu mutieren)."""
    leverage = barrier_cfg['leverage']
    barrier_pct = barrier_cfg['barrier_pct']
    start_capital = barrier_cfg.get('backtest_start_capital', 15.0)
    fee_rate = barrier_cfg.get('taker_fee_rate_pct', 0.06) / 100.0
    # .get()-Fallbacks aus demselben Grund wie in build_trades() -- diese drei leben seit dem
    # Coin/Timeframe-Config-Umbau in der optionalen Strategie-Config-Datei. Fallback-Werte
    # bewusst identisch zu live_trade.py's execute_live_trade(), damit "kein Config-Datei
    # vorhanden" ueberall dieselben Standardwerte ergibt.
    am_base = base_pct if base_pct is not None else barrier_cfg.get('anti_martingale_base_pct', 5.0)
    am_growth = growth_factor if growth_factor is not None else barrier_cfg.get('anti_martingale_growth_factor', 2.0)
    am_streak_target = int(streak_target if streak_target is not None
                            else barrier_cfg.get('anti_martingale_streak_target', 3))

    # anti_martingale.resolve_pending_outcome() loggt pro Position (sinnvoll im Live-Betrieb, 1
    # Aufruf pro 4h) -- bei einem Mehrhundert-Trade-Backtest (oder gar einem Parameter-Sweep mit
    # tausenden Wiederholungen) waere das reines Zeilen-Rauschen.
    logging.getLogger('oraclebot.strategy.anti_martingale').setLevel(logging.WARNING)

    am_state = {'stake_pct': am_base, 'consecutive_wins': 0, 'pending_position': None}
    capital = start_capital
    peak = capital
    max_dd = 0.0
    for t in trades:
        margin = compute_margin(capital, am_state)
        balance_before = capital
        distance = t['entry'] * barrier_pct / 100.0
        contracts = (margin * leverage) / t['entry']
        fee = margin * leverage * fee_rate * 2
        pnl_usd = t['frac'] * leverage * margin - fee
        capital += pnl_usd
        capital = max(capital, 0.0)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
        expected_win_balance = balance_before + distance * contracts - fee
        expected_loss_balance = balance_before - distance * contracts - fee
        am_state = record_pending_position(am_state, balance_before, expected_win_balance, expected_loss_balance)
        am_state = resolve_pending_outcome(am_state, capital, am_base, am_growth, am_streak_target)
        t['margin_used'] = margin
        t['pnl_usdt'] = pnl_usd
        t['equity_after'] = capital
    return {'end_capital': capital, 'start_capital': start_capital, 'max_dd_pct': max_dd * 100}


def bootstrap_max_dd_percentile(trades: list, barrier_cfg: dict, base_pct: float, growth_factor: float,
                                 streak_target: int, percentile: float = 90, n_boot: int = 3000,
                                 min_notional_usdt: float = 5.0, seed: int = 42) -> dict:
    """Bootstrap-robuste MaxDD-Schaetzung (siehe README: Anti-Martingale-Drawdown haengt stark
    von der ZUFAELLIGEN Trade-Reihenfolge ab -- ein einzelner historischer Pfad kann zufaellig
    guenstig oder unguenstig geordnet sein). Zieht `n_boot` Trade-Sequenzen MIT Zuruecklegen aus
    den beobachteten Trade-Ergebnissen, berechnet je Sequenz MaxDD/PnL/uebersprungene Trades
    (Notional < min_notional_usdt, wie live_trade.py's Mindest-Notional-Pruefung), gibt die
    Perzentile ueber alle Sequenzen zurueck."""
    leverage = barrier_cfg['leverage']
    start_capital = barrier_cfg.get('backtest_start_capital', 15.0)
    fee_rate = barrier_cfg.get('taker_fee_rate_pct', 0.06) / 100.0
    fracs = np.array([t['frac'] for t in trades])
    n = len(fracs)
    rng = np.random.default_rng(seed)

    dds = np.empty(n_boot)
    pnls = np.empty(n_boot)
    skips = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        capital = start_capital
        stake_pct = base_pct
        consecutive_wins = 0
        peak = capital
        max_dd = 0.0
        n_skip = 0
        for frac in fracs[idx]:
            margin = capital * stake_pct / 100.0
            notional = margin * leverage
            if notional < min_notional_usdt or capital <= 0:
                n_skip += 1
                continue
            fee = notional * fee_rate * 2
            pnl_usd = frac * leverage * margin - fee
            capital += pnl_usd
            capital = max(capital, 0.0)
            peak = max(peak, capital)
            if peak > 0:
                max_dd = max(max_dd, (peak - capital) / peak)
            if frac > 0:
                consecutive_wins += 1
                if consecutive_wins >= streak_target:
                    stake_pct = base_pct
                    consecutive_wins = 0
                else:
                    stake_pct *= growth_factor
            else:
                consecutive_wins = 0
                stake_pct = base_pct
        dds[i] = max_dd * 100
        pnls[i] = (capital - start_capital) / start_capital * 100
        skips[i] = n_skip

    return {
        'p50_dd': float(np.percentile(dds, 50)),
        'p_dd': float(np.percentile(dds, percentile)),
        'p50_pnl': float(np.percentile(pnls, 50)),
        'median_skips': float(np.median(skips)),
    }


def calibrate_anti_martingale_base_pct(trades: list, barrier_cfg: dict, growth_factor: float, streak_target: int,
                                        dd_percentile: float, dd_limit: float, n_boot_search: int = 600,
                                        n_boot_final: int = 3000) -> dict:
    """Bisektion: findet den groessten `base_pct`, bei dem das Bootstrap-`dd_percentile`-MaxDD
    noch <= `dd_limit` bleibt, fuer eine feste (growth_factor, streak_target)-Kombination."""
    lo, hi = 0.05, 15.0
    for _ in range(22):
        mid = (lo + hi) / 2
        result = bootstrap_max_dd_percentile(trades, barrier_cfg, mid, growth_factor, streak_target,
                                              percentile=dd_percentile, n_boot=n_boot_search)
        if result['p_dd'] <= dd_limit:
            lo = mid
        else:
            hi = mid
    final = bootstrap_max_dd_percentile(trades, barrier_cfg, lo, growth_factor, streak_target,
                                         percentile=dd_percentile, n_boot=n_boot_final)
    return {'base_pct': lo, **final}
