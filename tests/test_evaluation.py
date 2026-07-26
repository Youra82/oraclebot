import numpy as np
import pandas as pd
import pytest

from oraclebot.analysis.evaluation import (
    bootstrap_max_dd_percentile, build_trades, calibrate_anti_martingale_base_pct,
    evaluate_walk_forward, run_anti_martingale_backtest, walk_forward_predictions,
)
from oraclebot.data.features import FEATURE_NAMES


def make_examples(n=240, seed=0):
    """Feature 0 traegt das Signal (wie test_barrier_model.py), Kerzen 4h auseinander, damit
    date/exit_time chronologisch plausibel sind."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)
    dates = pd.date_range('2024-01-01', periods=n, freq='4h', tz='UTC')
    examples = []
    for i in range(n):
        examples.append({
            'features': X[i].tolist(),
            'target': int(y[i]),
            'date': dates[i].isoformat(),
            'entry': 60000.0 + i,
            'exit_time': (dates[i] + pd.Timedelta(hours=4)).isoformat(),
        })
    return examples


BASE_CFG = {'barrier_pct': 1.0, 'leverage': 100, 'backtest_start_capital': 15.0,
            'taker_fee_rate_pct': 0.06, 'min_confidence': 0.60,
            'anti_martingale_base_pct': 5.0, 'anti_martingale_growth_factor': 2.0,
            'anti_martingale_streak_target': 3}


class TestEvaluateWalkForward:
    def test_returns_accuracies_mean_and_worst_case_in_valid_range(self):
        result = evaluate_walk_forward(make_examples(), n_folds=8, max_depth=3)
        assert len(result['accuracies']) > 0
        assert all(0.0 <= a <= 1.0 for a in result['accuracies'])
        assert result['worst_case'] == min(result['accuracies'])
        assert result['mean'] == pytest.approx(np.mean(result['accuracies']))

    def test_learns_above_random_baseline_on_separable_signal(self):
        result = evaluate_walk_forward(make_examples(n=400, seed=1), n_folds=8, max_depth=3)
        assert result['mean'] > 0.6  # deutlich ueber Zufall (50%)


class TestWalkForwardPredictions:
    def test_returns_prediction_dict_per_out_of_fold_example(self):
        examples = make_examples(n=240)
        preds = walk_forward_predictions(examples, max_depth=3, n_folds=8)
        assert len(preds) > 0
        for p in preds:
            assert set(p.keys()) == {'date', 'entry', 'cls', 'conf', 'label', 'exit_time'}
            assert p['cls'] in (0, 1)
            assert 0.0 <= p['conf'] <= 1.0

    def test_predictions_are_sorted_chronologically(self):
        preds = walk_forward_predictions(make_examples(n=240), max_depth=3, n_folds=8)
        dates = [p['date'] for p in preds]
        assert dates == sorted(dates)

    def test_never_touches_the_first_fold_used_only_as_training_seed(self):
        examples = make_examples(n=240)
        preds = walk_forward_predictions(examples, max_depth=3, n_folds=8)
        n = len(examples)
        first_fold_end = int(n * 1 / 8)
        first_fold_dates = {ex['date'] for ex in examples[:first_fold_end]}
        pred_dates = {p['date'] for p in preds}
        assert first_fold_dates.isdisjoint(pred_dates)


class TestBuildTrades:
    def _pred(self, date, entry, cls, conf, label, exit_hours=4):
        exit_time = pd.Timestamp(date) + pd.Timedelta(hours=exit_hours)
        return {'date': date, 'entry': entry, 'cls': cls, 'conf': conf, 'label': label,
                'exit_time': exit_time}

    def test_high_confidence_correct_prediction_is_a_win(self):
        preds = [self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.9, label=1)]
        trades = build_trades(preds, BASE_CFG)
        assert len(trades) == 1
        assert trades[0]['outcome'] == 'win'
        assert trades[0]['direction'] == 'long'

    def test_high_confidence_wrong_prediction_is_a_loss(self):
        preds = [self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.9, label=0)]
        trades = build_trades(preds, BASE_CFG)
        assert trades[0]['outcome'] == 'loss'

    def test_low_confidence_prediction_produces_no_trade(self):
        preds = [self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.55, label=1)]
        trades = build_trades(preds, BASE_CFG)
        assert trades == []

    def test_min_confidence_override_takes_precedence_over_barrier_cfg(self):
        preds = [self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.55, label=1)]
        trades = build_trades(preds, BASE_CFG, min_confidence=0.50)
        assert len(trades) == 1

    def test_missing_min_confidence_in_barrier_cfg_falls_back_to_default(self):
        cfg_without_min_confidence = {k: v for k, v in BASE_CFG.items() if k != 'min_confidence'}
        preds = [self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.65, label=1)]
        trades = build_trades(preds, cfg_without_min_confidence)  # Default ist 0.60
        assert len(trades) == 1

    def test_predictions_up_to_and_including_the_exit_time_boundary_are_skipped(self):
        preds = [
            self._pred('2024-01-01T00:00:00+00:00', 60000.0, cls=1, conf=0.9, label=1, exit_hours=8),
            self._pred('2024-01-01T04:00:00+00:00', 60100.0, cls=1, conf=0.9, label=1),
            self._pred('2024-01-01T08:00:00+00:00', 60200.0, cls=1, conf=0.9, label=1),  # == exit_time, ebenfalls uebersprungen ('<=')
            self._pred('2024-01-01T12:00:00+00:00', 60300.0, cls=1, conf=0.9, label=1),  # erst danach neuer Trade
        ]
        trades = build_trades(preds, BASE_CFG)
        assert len(trades) == 2
        assert trades[0]['entry'] == 60000.0
        assert trades[1]['entry'] == 60300.0


class TestRunAntiMartingaleBacktest:
    def _trade(self, entry=60000.0, frac=0.01, outcome='win'):
        return {'entry_time': pd.Timestamp('2024-01-01', tz='UTC'), 'exit_time': pd.Timestamp('2024-01-01', tz='UTC'),
                'entry': entry, 'exit': entry, 'direction': 'long', 'frac': frac, 'outcome': outcome}

    def test_writes_margin_pnl_and_equity_into_trade_dicts(self):
        trades = [self._trade(frac=0.01), self._trade(frac=-0.01)]
        run_anti_martingale_backtest(trades, BASE_CFG)
        for t in trades:
            assert 'margin_used' in t and 'pnl_usdt' in t and 'equity_after' in t

    def test_missing_anti_martingale_keys_fall_back_to_live_trade_defaults(self):
        cfg_without_am = {k: v for k, v in BASE_CFG.items() if not k.startswith('anti_martingale_')}
        trades = [self._trade(frac=0.01)]
        result = run_anti_martingale_backtest(trades, cfg_without_am)  # darf nicht mit KeyError abstuerzen
        assert result['start_capital'] == BASE_CFG['backtest_start_capital']

    def test_override_params_take_precedence_over_barrier_cfg(self):
        trades_default = [self._trade(frac=0.02) for _ in range(3)]
        trades_override = [self._trade(frac=0.02) for _ in range(3)]
        run_anti_martingale_backtest(trades_default, BASE_CFG)
        run_anti_martingale_backtest(trades_override, BASE_CFG, base_pct=10.0)
        # Doppelter Einsatz -> etwa doppelter PnL fuer denselben Gewinn-Trade
        assert trades_override[0]['pnl_usdt'] > trades_default[0]['pnl_usdt'] * 1.5

    def test_all_losses_reduce_capital_and_produce_positive_max_dd(self):
        trades = [self._trade(frac=-0.01, outcome='loss') for _ in range(5)]
        result = run_anti_martingale_backtest(trades, BASE_CFG)
        assert result['end_capital'] < result['start_capital']
        assert result['max_dd_pct'] > 0


class TestBootstrapMaxDdPercentile:
    def _trades(self, n=200, win_rate=0.7, seed=0):
        rng = np.random.default_rng(seed)
        wins = rng.random(n) < win_rate
        return [{'frac': 0.01 if w else -0.01} for w in wins]

    def test_returns_expected_keys_with_sane_ranges(self):
        result = bootstrap_max_dd_percentile(self._trades(), BASE_CFG, base_pct=5.0, growth_factor=2.0,
                                              streak_target=3, n_boot=200)
        assert set(result.keys()) == {'p50_dd', 'p_dd', 'p50_pnl', 'median_skips'}
        assert result['p_dd'] >= result['p50_dd'] >= 0  # 90. Perzentil >= Median per Definition
        assert result['median_skips'] == 0

    def test_flags_degenerate_config_where_every_trade_is_skipped(self):
        # Winziges Startkapital + winziger Einsatz -> Notional bleibt immer unter dem
        # Mindest-Notional-Floor, jeder simulierte Trade wird uebersprungen.
        trades = self._trades(n=200)
        tiny_cfg = dict(BASE_CFG, backtest_start_capital=1.0, leverage=1)
        result = bootstrap_max_dd_percentile(trades, tiny_cfg, base_pct=0.01, growth_factor=2.0,
                                              streak_target=3, n_boot=100, min_notional_usdt=5.0)
        assert result['median_skips'] == len(trades)  # jede Bootstrap-Sequenz zieht len(trades) Trades, alle uebersprungen
        assert result['p50_dd'] == 0.0

    def test_is_deterministic_for_a_fixed_seed(self):
        r1 = bootstrap_max_dd_percentile(self._trades(), BASE_CFG, 5.0, 2.0, 3, n_boot=100, seed=7)
        r2 = bootstrap_max_dd_percentile(self._trades(), BASE_CFG, 5.0, 2.0, 3, n_boot=100, seed=7)
        assert r1 == r2


class TestCalibrateAntiMartingaleBasePct:
    def _trades(self, n=300, win_rate=0.75, seed=0):
        rng = np.random.default_rng(seed)
        wins = rng.random(n) < win_rate
        return [{'frac': 0.01 if w else -0.01} for w in wins]

    def test_found_base_pct_keeps_target_percentile_dd_within_limit(self):
        trades = self._trades()
        result = calibrate_anti_martingale_base_pct(trades, BASE_CFG, growth_factor=2.0, streak_target=3,
                                                      dd_percentile=90, dd_limit=50.0,
                                                      n_boot_search=150, n_boot_final=300)
        assert 0.05 <= result['base_pct'] <= 15.0
        assert result['p_dd'] <= 50.0 + 1.0  # kleine Toleranz, da Bisektion nicht exakt konvergiert

    def test_looser_dd_limit_allows_a_larger_or_equal_base_pct(self):
        trades = self._trades()
        tight = calibrate_anti_martingale_base_pct(trades, BASE_CFG, 2.0, 3, 90, dd_limit=20.0,
                                                     n_boot_search=150, n_boot_final=300)
        loose = calibrate_anti_martingale_base_pct(trades, BASE_CFG, 2.0, 3, 90, dd_limit=60.0,
                                                     n_boot_search=150, n_boot_final=300)
        assert loose['base_pct'] >= tight['base_pct']
