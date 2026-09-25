import os

from oraclebot.strategy import anti_martingale


def test_load_state_creates_fresh_state_when_no_file(tmp_path):
    path = os.path.join(tmp_path, 'state.json')
    state = anti_martingale.load_state(path, base_pct=7.25)
    assert state == {'stake_pct': 7.25, 'consecutive_wins': 0}


def test_save_then_load_roundtrip(tmp_path):
    path = os.path.join(tmp_path, 'nested', 'state.json')
    state = {'stake_pct': 14.5, 'consecutive_wins': 1}
    anti_martingale.save_state(path, state)
    loaded = anti_martingale.load_state(path, base_pct=7.25)
    assert loaded == state


def test_load_state_resyncs_stale_stake_pct_when_no_streak_in_progress(tmp_path):
    # Realer Fund 2026-07-28: nach einem optimize_barrier_model.py-Lauf, der
    # anti_martingale_base_pct z.B. von 4.03% auf 14.80% anhob, blieb ein bereits
    # abgeschlossener (consecutive_wins=0) Live-Zustand auf dem alten 4.03%-Einsatz haengen,
    # da load_state() den vorhandenen Wert bisher nie mit dem aktuellen base_pct abgeglichen hat.
    path = os.path.join(tmp_path, 'state.json')
    anti_martingale.save_state(path, {'stake_pct': 4.03, 'consecutive_wins': 0})
    loaded = anti_martingale.load_state(path, base_pct=14.80)
    assert loaded['stake_pct'] == 14.80


def test_load_state_keeps_compounded_stake_mid_streak(tmp_path):
    # Waehrend eines laufenden Gewinn-Streaks (consecutive_wins > 0) bleibt das Compounding
    # bewusst unangetastet -- nur der Reset-Zeitpunkt (Streak-Ende/Verlust) synct auf base_pct.
    path = os.path.join(tmp_path, 'state.json')
    anti_martingale.save_state(path, {'stake_pct': 14.5, 'consecutive_wins': 1})
    loaded = anti_martingale.load_state(path, base_pct=7.25)
    assert loaded['stake_pct'] == 14.5


def test_compute_margin_scales_with_current_balance():
    state = {'stake_pct': 10.0}
    assert anti_martingale.compute_margin(1000.0, state) == 100.0
    assert anti_martingale.compute_margin(500.0, state) == 50.0


def test_win_doubles_stake_when_streak_not_yet_reached():
    state = {'stake_pct': 7.25, 'consecutive_wins': 0}
    state = anti_martingale.update_after_close(state, base_pct=7.25, growth_factor=2.0,
                                                streak_target=3, is_win=True)
    assert state['consecutive_wins'] == 1
    assert state['stake_pct'] == 14.5


def test_loss_resets_stake_and_streak_immediately():
    state = {'stake_pct': 29.0, 'consecutive_wins': 2}
    state = anti_martingale.update_after_close(state, base_pct=7.25, growth_factor=2.0,
                                                streak_target=3, is_win=False)
    assert state['consecutive_wins'] == 0
    assert state['stake_pct'] == 7.25


def test_third_consecutive_win_resets_stake_to_base_instead_of_doubling_again():
    state = {'stake_pct': 29.0, 'consecutive_wins': 2}
    state = anti_martingale.update_after_close(state, base_pct=7.25, growth_factor=2.0,
                                                streak_target=3, is_win=True)
    assert state['consecutive_wins'] == 0
    assert state['stake_pct'] == 7.25


def test_multiple_consecutive_wins_across_different_symbols_share_one_streak():
    # Gemeinsamer Streak ueber alle Symbole (Fix 2026-09-25, Mehrfach-Positionen-Umbau) -- egal
    # welches Symbol gerade schliesst, der Streak ist EIN gemeinsamer Zustand fuers Portfolio.
    state = {'stake_pct': 7.25, 'consecutive_wins': 0}
    state = anti_martingale.update_after_close(state, base_pct=7.25, growth_factor=2.0,
                                                streak_target=3, is_win=True)  # z.B. NEAR schliesst
    state = anti_martingale.update_after_close(state, base_pct=7.25, growth_factor=2.0,
                                                streak_target=3, is_win=True)  # z.B. DOT schliesst
    assert state['consecutive_wins'] == 2
    assert state['stake_pct'] == 29.0
