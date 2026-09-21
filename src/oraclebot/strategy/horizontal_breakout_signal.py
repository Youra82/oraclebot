# src/oraclebot/strategy/horizontal_breakout_signal.py
# Nutzer-Idee (2026-09-21): kein Vorhersage-Modell, keine Trendfolge-Regel auf OHLCV-Kerzen --
# stattdessen die EAR-Renko-Bricks aus zerobot (data/ear_bricks.py) als Grundlage, aber eine
# ANDERE Signallogik als zerobots eigener Entropy-Squeeze: Bricks bewegen sich eine Weile
# "horizontal" (wechselnde Richtung, keine klare Serie) -- sobald daraus ein ECHTER Ausbruch
# entsteht (N Bricks am Stueck in dieselbe Richtung, direkt nach einer Seitwaertsphase), wird
# ein Trade in Ausbruchsrichtung eroeffnet. Exit exakt beim ersten VOLLSTAENDIG ausgebildeten
# Brick in die Gegenrichtung -- kein festes SL/TP, die Brick-Struktur selbst definiert den Exit.


def backtest_horizontal_breakout(bricks: list, horizontal_lookback: int, breakout_run: int) -> list:
    """Simuliert die Strategie ueber eine bereits gebaute Brick-Kette (siehe
    data.ear_bricks.build_ear_bricks): Entry beim ersten `breakout_run`-Bricks-Lauf in dieselbe
    Richtung, sofern die `horizontal_lookback` Bricks UNMITTELBAR DAVOR eine Seitwaertsphase
    waren (gemischte Richtungen, kein bereits laufender Trend). Exit beim ersten Brick der
    Gegenrichtung.

    Args:
        bricks: chronologische Liste von Brick-Dicts (direction/close/ts), wie von
            build_ear_bricks() geliefert.
        horizontal_lookback: wie viele Bricks VOR dem Ausbruchs-Lauf auf "Seitwaerts"
            (gemischte Richtung) geprueft werden.
        breakout_run: wie viele Bricks am Stueck in dieselbe Richtung einen Ausbruch bestaetigen.

    Returns:
        Liste von Trade-Dicts: {direction ('long'/'short'), entry_idx, entry_price, entry_ts,
        exit_idx, exit_price, exit_ts, pnl_pct (vorzeichenkorrekt, OHNE Gebuehren),
        bricks_held}. Ein am Ende der Kette noch offener Trade wird NICHT zurueckgegeben (kein
        bestimmbares Ergebnis, analog zu barrier_targets.compute_barrier_labels()).
    """
    trades = []
    run_dir = None
    run_len = 0
    in_trade = False
    trade_dir = None
    entry_price = None
    entry_idx = None

    for i, brick in enumerate(bricks):
        d = brick['direction']

        if in_trade:
            if d != trade_dir:
                exit_price = brick['close']
                sign = 1 if trade_dir == 'up' else -1
                pnl_pct = sign * (exit_price - entry_price) / entry_price * 100.0
                trades.append({
                    'direction': 'long' if trade_dir == 'up' else 'short',
                    'entry_idx': entry_idx, 'entry_price': entry_price, 'entry_ts': bricks[entry_idx]['ts'],
                    'exit_idx': i, 'exit_price': exit_price, 'exit_ts': brick['ts'],
                    'pnl_pct': pnl_pct, 'bricks_held': i - entry_idx,
                })
                in_trade = False
                # Der Gegen-Brick selbst startet moeglicherweise schon einen neuen Lauf.
                run_dir, run_len = d, 1
            continue

        if d == run_dir:
            run_len += 1
        else:
            run_dir, run_len = d, 1

        if run_len == breakout_run:
            run_start = i - breakout_run + 1
            window_start = run_start - horizontal_lookback
            if window_start < 0:
                continue
            window_dirs = {b['direction'] for b in bricks[window_start:run_start]}
            if len(window_dirs) > 1:  # gemischt = Seitwaertsphase
                in_trade = True
                trade_dir = run_dir
                entry_price = brick['close']
                entry_idx = i

    return trades


def simulate_with_open_state(bricks: list, horizontal_lookback: int, breakout_run: int) -> tuple:
    """Wie backtest_horizontal_breakout(), liefert aber zusaetzlich den am Ende der Brick-Kette
    noch offenen Trade zurueck (falls vorhanden) -- fuer den Live-Betrieb, der wissen muss, ob
    der JUENGSTE Brick gerade einen frischen Entry ausgeloest hat. Identische Zustandsmaschine
    (siehe test_open_state_trades_match_closed_trades fuer den Aequivalenznachweis: die
    'trades'-Liste ist byte-identisch zu backtest_horizontal_breakout() fuer denselben Input).

    Returns:
        (trades, open_position): trades wie backtest_horizontal_breakout(); open_position ist
        None oder {direction, entry_idx, entry_price, entry_ts}, falls am Ende der Kette noch
        eine Position offen ist.
    """
    trades = []
    run_dir = None
    run_len = 0
    in_trade = False
    trade_dir = None
    entry_price = None
    entry_idx = None

    for i, brick in enumerate(bricks):
        d = brick['direction']

        if in_trade:
            if d != trade_dir:
                exit_price = brick['close']
                sign = 1 if trade_dir == 'up' else -1
                pnl_pct = sign * (exit_price - entry_price) / entry_price * 100.0
                trades.append({
                    'direction': 'long' if trade_dir == 'up' else 'short',
                    'entry_idx': entry_idx, 'entry_price': entry_price, 'entry_ts': bricks[entry_idx]['ts'],
                    'exit_idx': i, 'exit_price': exit_price, 'exit_ts': brick['ts'],
                    'pnl_pct': pnl_pct, 'bricks_held': i - entry_idx,
                })
                in_trade = False
                run_dir, run_len = d, 1
            continue

        if d == run_dir:
            run_len += 1
        else:
            run_dir, run_len = d, 1

        if run_len == breakout_run:
            run_start = i - breakout_run + 1
            window_start = run_start - horizontal_lookback
            if window_start < 0:
                continue
            window_dirs = {b['direction'] for b in bricks[window_start:run_start]}
            if len(window_dirs) > 1:
                in_trade = True
                trade_dir = run_dir
                entry_price = brick['close']
                entry_idx = i

    open_position = None
    if in_trade:
        open_position = {
            'direction': 'long' if trade_dir == 'up' else 'short',
            'entry_idx': entry_idx, 'entry_price': entry_price, 'entry_ts': bricks[entry_idx]['ts'],
        }
    return trades, open_position
