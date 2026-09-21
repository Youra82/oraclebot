import pandas as pd
import pytest

from oraclebot.strategy.horizontal_breakout_signal import (backtest_horizontal_breakout,
                                                             simulate_with_open_state)


def make_bricks(directions, start_close=100.0, step=1.0, start='2024-01-01', freq='15min'):
    """directions: Liste von 'up'/'down' -- baut monoton steigende/fallende Brick-Schluesse
    entsprechend der jeweiligen Richtung (jeder Brick bewegt den Preis um `step`)."""
    idx = pd.date_range(start, periods=len(directions), freq=freq, tz='UTC')
    bricks = []
    close = start_close
    for ts, d in zip(idx, directions):
        close = close + step if d == 'up' else close - step
        bricks.append({'direction': d, 'close': close, 'ts': ts})
    return bricks


def test_no_signal_without_enough_history():
    bricks = make_bricks(['up', 'up', 'up'])
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=3, breakout_run=2)
    assert trades == []


def test_breakout_after_horizontal_phase_triggers_long_entry():
    # 4 Bricks Seitwaerts (gemischt), dann 2 Bricks am Stueck 'up' -> Ausbruch long.
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up'] + ['down']  # letzter Brick = Exit
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert len(trades) == 1
    assert trades[0]['direction'] == 'long'
    assert trades[0]['entry_idx'] == 5  # 0-indiziert: der 6. Brick (zweiter 'up' des Laufs)
    assert trades[0]['exit_idx'] == 6
    assert trades[0]['pnl_pct'] < 0  # Exit-Brick ist 'down' -> Verlust fuer den Long


def test_breakout_after_horizontal_phase_triggers_short_entry():
    # Praefix endet auf 'up' (anders als die Ausbruchsrichtung 'down'), sonst wuerde der letzte
    # Praefix-Brick faelschlich schon als erster Brick des Ausbruchslaufs mitgezaehlt.
    directions = ['down', 'up', 'down', 'up'] + ['down', 'down'] + ['up']
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert len(trades) == 1
    assert trades[0]['direction'] == 'short'


def test_no_breakout_if_lookback_window_was_already_trending():
    # Lookback-Fenster (die 4 Bricks VOR dem neuen Lauf) ist komplett 'down' (kein Seitwaerts,
    # sondern selbst schon ein Trend) -- der anschliessende frische 'up'-Lauf zaehlt deshalb
    # NICHT als Ausbruch aus einer Seitwaertsphase.
    directions = ['down', 'down', 'down', 'down'] + ['up', 'up']
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert trades == []


def test_trade_holds_through_multiple_same_direction_bricks_until_reversal():
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up', 'up', 'up'] + ['down']
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert len(trades) == 1
    assert trades[0]['bricks_held'] == 3  # von Index 5 (2. 'up' des Laufs) bis Index 8 ('down')


def test_open_trade_at_end_of_chain_is_not_returned():
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up']  # kein Gegen-Brick mehr danach
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert trades == []


def test_after_exit_a_new_breakout_can_start_from_the_reversal_brick():
    # Nach dem Exit-Brick (Beginn eines neuen Laufs) folgt sofort ein zweiter gleichgerichteter
    # Brick -- OHNE neue Seitwaertsphase davor darf das aber NICHT nochmal signalisieren, weil
    # unmittelbar davor (die letzten 4 Bricks des ersten Laufs) klar 'up' waren, nicht gemischt.
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up', 'up', 'up'] + ['down', 'down']
    bricks = make_bricks(directions)
    trades = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    assert len(trades) == 1  # nur der erste Ausbruch, der zweite Lauf ('down','down') hat kein Seitwaerts-Fenster davor


def test_open_state_trades_match_closed_trades():
    # simulate_with_open_state() muss fuer die geschlossenen Trades byte-identisch zu
    # backtest_horizontal_breakout() sein (gleiche Zustandsmaschine, nur mit zusaetzlichem
    # Rueckgabewert fuer den Live-Betrieb) -- Grundlage fuer die Live-Signalerkennung.
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up'] + ['down']
    bricks = make_bricks(directions)
    closed = backtest_horizontal_breakout(bricks, horizontal_lookback=4, breakout_run=2)
    trades, open_position = simulate_with_open_state(bricks, horizontal_lookback=4, breakout_run=2)
    assert trades == closed
    assert open_position is None  # letzter Brick ist der Exit, am Ende der Kette nichts offen


def test_open_state_reports_open_position_at_end_of_chain():
    # Gleiches Setup wie test_open_trade_at_end_of_chain_is_not_returned(), aber hier soll
    # simulate_with_open_state() die noch offene Position explizit zurueckgeben.
    directions = ['up', 'down', 'up', 'down'] + ['up', 'up']
    bricks = make_bricks(directions)
    trades, open_position = simulate_with_open_state(bricks, horizontal_lookback=4, breakout_run=2)
    assert trades == []
    assert open_position is not None
    assert open_position['direction'] == 'long'
    assert open_position['entry_idx'] == 5
