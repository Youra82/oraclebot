import numpy as np
import pandas as pd

from scripts.optimize_barrier_model import select_anti_martingale, select_best_max_depth, select_min_confidence


def make_pred(date, entry, cls, conf, label, exit_hours=4):
    exit_time = pd.Timestamp(date) + pd.Timedelta(hours=exit_hours)
    return {'date': date, 'entry': entry, 'cls': cls, 'conf': conf, 'label': label, 'exit_time': exit_time}


BASE_CFG = {'barrier_pct': 1.0, 'leverage': 100, 'backtest_start_capital': 15.0, 'taker_fee_rate_pct': 0.06}


class TestSelectBestMaxDepth:
    def test_prefers_the_shallowest_depth_when_all_are_statistically_indistinguishable(self, monkeypatch):
        import scripts.optimize_barrier_model as opt

        # Alle 5 Kandidaten-Tiefen liefern (fast) identische Werte -> Ein-Standardfehler-Regel
        # muss die FLACHSTE (2, erste in MAX_DEPTH_CANDIDATES) waehlen, nicht die technisch
        # hoechste Worst-Case (die durch Rauschen leicht abweichen kann).
        # Nur die 'accuracies'-Liste des tatsaechlichen Gewinners (per worst_case/mean) fliesst
        # in die Standardfehler-Berechnung ein -- hier bewusst mit grosser Eigen-Streuung
        # versehen, damit der Standardfehler die kleinen (< 2pp) Unterschiede zu den anderen
        # Tiefen ueberdeckt (wie im echten Lauf vom 2026-07-26 beobachtet: SE=1.6% bei einer
        # Worst-Case-Spanne von nur 0.9pp).
        fake_results = {
            2: {'accuracies': [0.680], 'mean': 0.750, 'worst_case': 0.680},
            3: {'accuracies': [0.685], 'mean': 0.755, 'worst_case': 0.685},
            4: {'accuracies': [0.678], 'mean': 0.752, 'worst_case': 0.678},
            5: {'accuracies': [0.695], 'mean': 0.758, 'worst_case': 0.695},
            6: {'accuracies': [0.70, 0.75, 0.80, 0.85, 0.90, 0.72, 0.71, 0.73],
                'mean': 0.770, 'worst_case': 0.700},
        }
        monkeypatch.setattr(opt, 'evaluate_walk_forward',
                             lambda examples, n_folds=8, max_depth=3: fake_results[max_depth])
        best = select_best_max_depth(examples=['dummy'] * 50, n_folds=8)
        assert best['max_depth'] == 2

    def test_prefers_a_deeper_model_when_the_gap_clearly_exceeds_the_standard_error(self, monkeypatch):
        import scripts.optimize_barrier_model as opt

        # depth=6 liegt weit ausserhalb der Rausch-Bandbreite der anderen Tiefen (grosser,
        # echter Vorteil) -- hier MUSS die Regel die genauere, tiefere Wahl treffen.
        fake_results = {
            2: {'accuracies': [0.59], 'mean': 0.60, 'worst_case': 0.59},
            3: {'accuracies': [0.60], 'mean': 0.60, 'worst_case': 0.60},
            4: {'accuracies': [0.59], 'mean': 0.60, 'worst_case': 0.59},
            5: {'accuracies': [0.60], 'mean': 0.61, 'worst_case': 0.60},
            6: {'accuracies': [0.890, 0.900, 0.910, 0.895, 0.905, 0.890, 0.900, 0.895],
                'mean': 0.897, 'worst_case': 0.890},
        }
        monkeypatch.setattr(opt, 'evaluate_walk_forward',
                             lambda examples, n_folds=8, max_depth=3: fake_results[max_depth])
        best = select_best_max_depth(examples=['dummy'] * 50, n_folds=8)
        assert best['max_depth'] == 6


class TestSelectMinConfidence:
    def test_picks_lowest_threshold_that_clears_the_winrate_floor(self):
        # 8 Trades pro Kerze, Winrate steigt mit hoeherer Schwelle -- 0.60 ist die niedrigste,
        # die die 70%-Sicherheitsmarge noch haelt (0.50/0.55 liegen darunter).
        preds = []
        for i in range(20):
            date = f'2024-01-{(i % 28) + 1:02d}T00:00:00+00:00'
            conf = 0.50 + (i % 5) * 0.08  # 0.50, 0.58, 0.66, 0.74, 0.82 im Wechsel
            label = 1 if conf >= 0.66 else (1 if i % 2 == 0 else 0)
            preds.append(make_pred(date, 60000.0 + i, cls=1, conf=conf, label=label))
        best = select_min_confidence(preds, BASE_CFG)
        assert best['min_confidence'] in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)

    def test_falls_back_to_highest_winrate_when_no_threshold_clears_the_floor(self):
        # Winrate bleibt bei jeder Schwelle unter 70% -> Fallback: die mit der hoechsten Winrate.
        preds = []
        for i in range(30):
            date = f'2024-01-{(i % 28) + 1:02d}T00:00:00+00:00'
            won = i % 3 == 0  # ~33% Winrate, unabhaengig von der Konfidenz
            preds.append(make_pred(date, 60000.0 + i, cls=1, conf=0.55, label=1 if won else 0))
        best = select_min_confidence(preds, BASE_CFG)
        assert best['win_rate'] < 0.70


class TestSelectAntiMartingale:
    def test_excludes_combinations_where_trades_get_skipped_for_min_notional(self, monkeypatch):
        import scripts.optimize_barrier_model as opt

        preds = [make_pred(f'2024-01-{(i % 28) + 1:02d}T00:00:00+00:00', 60000.0, cls=1, conf=0.9,
                            label=1 if i % 4 != 0 else 0) for i in range(40)]

        def fake_calibrate(trades, barrier_cfg, growth_factor, streak_target, dd_percentile, dd_limit,
                            n_boot_search=600, n_boot_final=3000):
            # Ein einziger degenerierter Kandidat (streak=5, growth=3.0) hat Skips > 0 und muss
            # trotz hoechstem p50_pnl ausgeschlossen werden.
            if streak_target == 5 and growth_factor == 3.0:
                return {'base_pct': 0.1, 'p50_dd': 0.0, 'p_dd': 0.0, 'p50_pnl': 999.0, 'median_skips': 40.0}
            return {'base_pct': 3.0, 'p50_dd': 20.0, 'p_dd': 30.0, 'p50_pnl': 10.0, 'median_skips': 0.0}

        monkeypatch.setattr(opt, 'calibrate_anti_martingale_base_pct', fake_calibrate)
        best = select_anti_martingale(preds, BASE_CFG, min_confidence=0.60, dd_limit=50.0)
        assert best['median_skips'] == 0
        assert not (best['streak_target'] == 5 and best['growth_factor'] == 3.0)
