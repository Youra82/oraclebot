# src/oraclebot/strategy/renko_portfolio_state.py
# Persistenter Live-Zustand fuer die Portfolio-weite Renko-Breakout-Strategie (mehrere Symbole,
# aber je Runde maximal EINE offene Position ueber das gesamte Portfolio -- erstes Signal
# gewinnt, alle anderen werden verworfen statt nachgeholt, exakt wie die arbitrate()-Funktion im
# Backtest). Baut auf data.ear_bricks.build_ear_bricks() auf, das die inkrementelle Fortsetzung
# ueber `precomputed_H_roll`/`init_close`/`init_direction` bereits unterstuetzt (siehe
# test_incremental_continuation_with_precomputed_h_roll_matches_fresh_build) -- Grund fuer dieses
# Muster ist der dokumentierte zerobot-Vorfall, bei dem Live und Backtest die Brick-Kette
# strukturell unterschiedlich aufbauten ([[research_zerobot_live_vs_backtest_2026_08]]).
#
# Zustandsformat pro Symbol (JSON-Datei, ein Dict fuer alle Symbole):
#   {
#     "<symbol>": {
#       "last_candle_ts": "2026-09-21T16:05:00+00:00",
#       "buffer_candles": [{"ts":..,"open":..,"high":..,"low":..,"close":..}, ...],  # letzte h_window
#       "last_brick_close": 4.321, "last_brick_direction": "up"|"down"|null,
#       "recent_bricks": [{"ts":.., "direction":.., "close":..}, ...],  # getrimmtes Fenster
#     }, ...
#   }
import json
import logging
import os

import numpy as np
import pandas as pd

from oraclebot.data.ear_bricks import build_ear_bricks, candle_entropy

logger = logging.getLogger(__name__)

MAX_STORED_BRICKS = 60  # deutlich mehr als horizontal_lookback+breakout_run je gebraucht wird


def load_state(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_state(path: str, state: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2, default=str)


def _candles_to_records(df: pd.DataFrame) -> list:
    return [{'ts': ts.isoformat(), 'open': float(r.open), 'high': float(r.high),
              'low': float(r.low), 'close': float(r.close)} for ts, r in
             zip(df.index, df.itertuples())]


def _records_to_df(records: list) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    idx = pd.to_datetime([r['ts'] for r in records], utc=True)
    return pd.DataFrame({'open': [r['open'] for r in records], 'high': [r['high'] for r in records],
                          'low': [r['low'] for r in records], 'close': [r['close'] for r in records]},
                         index=idx)


def update_symbol_bricks(symbol_state: dict, new_candles: pd.DataFrame, base_pct: float,
                          k_entropy: float, h_window: int) -> tuple:
    """Verarbeitet neue, noch nicht gesehene Kerzen fuer ein Symbol und setzt die Brick-Kette
    inkrementell fort (kein Neuaufbau der kompletten Historie -- siehe Moduldoc fuer den Grund).

    Args:
        symbol_state: bisheriger Zustand fuer dieses Symbol (leeres Dict beim allerersten Lauf).
        new_candles: NUR die Kerzen NACH symbol_state['last_candle_ts'] (aufsteigend, DatetimeIndex).
        base_pct/k_entropy/h_window: wie build_ear_bricks(), MUSS mit dem Backtest identisch sein.

    Returns:
        (new_state, fresh_bricks): fresh_bricks sind ausschliesslich die in DIESEM Lauf neu
        entstandenen Bricks (chronologisch) -- fuer die Signalerkennung, die nur auf brandneuen
        Bricks feuern darf (siehe renko_portfolio_state Moduldoc: ein Signal, das waehrend einer
        offenen Position verworfen wurde, darf spaeter NICHT nachtraeglich nachgeholt werden).
    """
    if new_candles.empty:
        return symbol_state, []

    buffer_records = symbol_state.get('buffer_candles', [])
    buffer_df = _records_to_df(buffer_records)
    combined = pd.concat([buffer_df, new_candles]) if not buffer_df.empty else new_candles
    combined = combined[~combined.index.duplicated(keep='last')].sort_index()

    H_raw = candle_entropy(combined['open'].to_numpy(), combined['high'].to_numpy(),
                            combined['low'].to_numpy(), combined['close'].to_numpy())
    H_roll_full = pd.Series(H_raw).rolling(h_window, min_periods=1).mean().to_numpy()

    n_new = len(new_candles)
    H_roll_new = H_roll_full[-n_new:]

    last_close = symbol_state.get('last_brick_close')
    last_direction = symbol_state.get('last_brick_direction')
    if last_close is None:
        # Allererster Lauf ueberhaupt fuer dieses Symbol: build_ear_bricks() wuerde intern
        # ohnehin lc=closes[0] als Anker nehmen (init_close=None, start_i=1) -- das explizit zu
        # setzen (init_close=closes[0], damit build_ear_bricks ab Index 0 startet) ist aequivalent
        # (Index 0 gegen sich selbst als Anker ergibt nie einen Brick), macht die Fortsetzung aber
        # unabhaengig von der Batch-Groesse: ohne persistenten Anker wuerde jeder folgende Aufruf
        # mit weiterhin leerem last_close faelschlich WIEDER bei closes[0] DIESES Batches neu
        # verankern und die eigentliche Ankerkerze verlieren (siehe
        # test_incremental_batches_match_one_shot_build, batch_size=1).
        last_close = float(new_candles['close'].iloc[0])

    fresh_bricks = build_ear_bricks(new_candles, base_pct=base_pct, k_entropy=k_entropy,
                                     h_window=h_window, init_close=last_close,
                                     init_direction=last_direction, precomputed_H_roll=H_roll_new)

    recent_bricks = symbol_state.get('recent_bricks', []) + [
        {'ts': b['ts'].isoformat() if hasattr(b['ts'], 'isoformat') else str(b['ts']),
         'direction': b['direction'], 'close': b['close']} for b in fresh_bricks]
    recent_bricks = recent_bricks[-MAX_STORED_BRICKS:]

    new_buffer_df = combined.iloc[-h_window:]
    new_state = {
        'last_candle_ts': new_candles.index[-1].isoformat(),
        'buffer_candles': _candles_to_records(new_buffer_df),
        'last_brick_close': fresh_bricks[-1]['close'] if fresh_bricks else last_close,
        'last_brick_direction': fresh_bricks[-1]['direction'] if fresh_bricks else last_direction,
        'recent_bricks': recent_bricks,
    }
    return new_state, fresh_bricks


def detect_fresh_entry(recent_bricks: list, n_fresh: int, horizontal_lookback: int,
                        breakout_run: int) -> dict:
    """Prueft, ob der JUENGSTE Brick in `recent_bricks` gerade einen frischen Ausbruchs-Entry
    ausloest -- und dieser Ausloeser-Brick tatsaechlich einer der in DIESEM Lauf neu entstandenen
    ist (n_fresh > 0 Bricks am Ende der Liste), nicht ein alter, bereits einmal (zurecht)
    ignorierter Signal-Brick aus einer frueheren Runde. Nutzt dieselbe Zustandsmaschine wie
    horizontal_breakout_signal.simulate_with_open_state() -- ein Fenster von `recent_bricks`
    reicht aus (siehe Moduldoc: run_len/Fenster-Check haengen nur von den letzten
    horizontal_lookback+breakout_run Bricks ab, alles Aeltere aendert das Ergebnis am Ende der
    Kette nicht).

    Returns:
        None, oder {'direction': 'long'/'short', 'entry_price': float, 'entry_ts': str}.
    """
    from oraclebot.strategy.horizontal_breakout_signal import simulate_with_open_state

    if n_fresh <= 0 or len(recent_bricks) < horizontal_lookback + breakout_run:
        return None

    _, open_position = simulate_with_open_state(recent_bricks, horizontal_lookback, breakout_run)
    if open_position is None:
        return None
    # Der Entry-Brick muss innerhalb der zuletzt neu hinzugekommenen Bricks liegen -- sonst ist
    # es ein alter, in einer frueheren Runde bereits (korrekt) verworfener Entry.
    if open_position['entry_idx'] < len(recent_bricks) - n_fresh:
        return None
    return {'direction': open_position['direction'], 'entry_price': open_position['entry_price'],
            'entry_ts': str(open_position['entry_ts'])}


def detect_exit(recent_bricks: list, n_fresh: int, position_direction: str) -> dict:
    """Prueft, ob einer der in diesem Lauf neu entstandenen Bricks die Gegenrichtung der
    offenen Position bildet (= Exit-Signal, exakt die Backtest-Regel: erster vollstaendig
    ausgebildeter Gegen-Brick). Nimmt den ERSTEN passenden frischen Brick (nicht den letzten) --
    das entspricht dem Backtest, der beim allerersten Gegen-Brick exitet, auch wenn danach in
    derselben Kerze/demselben Lauf noch mehr Bricks entstanden sind.

    Args:
        position_direction: 'long' oder 'short' (aus der echten Exchange-Position).
    """
    if n_fresh <= 0:
        return None
    trade_dir = 'up' if position_direction == 'long' else 'down'
    for b in recent_bricks[-n_fresh:]:
        if b['direction'] != trade_dir:
            return {'exit_price': b['close'], 'exit_ts': b['ts']}
    return None
