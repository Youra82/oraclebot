# src/oraclebot/strategy/anti_martingale.py
# Anti-Martingale (Paroli)-Positionsgroesse: Einsatz (in % vom aktuellen Guthaben) verdoppelt
# sich nach jedem GEWINN, bis `streak_target` Gewinne in Folge erreicht sind (dann Reset auf die
# Basis), und faellt nach jedem VERLUST sofort auf die Basis zurueck -- das Gegenteil einer
# klassischen Martingale (die den Einsatz nach Verlusten erhoeht). Backtest (2026-07-24, BTC,
# SL=TP=1% manuell, Hebel=100x, min_trend_confidence=0.60): Basis=7.25% haelt den MaxDD knapp
# unter 50% bei +3792% PnL aus 15 USDT im getesteten Zeitraum -- das ist ein historisches
# Backtest-Ergebnis, keine Garantie fuer zukuenftige Performance.
#
# Zustandsbehaftet (im Gegensatz zur zustandslosen %-Risiko-Groesse in signal.py):
# braucht Persistenz ueber Bot-Neustarts/Cron-Laeufe hinweg, da der naechste Einsatz vom Ausgang
# der VORHERIGEN Position abhaengt.
import json
import logging
import os

logger = logging.getLogger(__name__)


def load_state(path: str, base_pct: float) -> dict:
    """Laedt den Anti-Martingale-Zustand von Platte, oder erzeugt einen frischen Start-Zustand."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        if not state.get('stake_pct'):
            state['stake_pct'] = base_pct
        return state
    return {'stake_pct': base_pct, 'consecutive_wins': 0, 'pending_position': None}


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def compute_margin(balance: float, state: dict) -> float:
    """Positionsgroesse (Margin in USDT) fuer den naechsten Trade: aktueller Einsatz-Prozentsatz
    vom AKTUELLEN Guthaben (nicht vom Startkapital -- reines Compounding)."""
    return balance * state.get('stake_pct', 0.0) / 100.0


def record_pending_position(state: dict, balance_before: float, expected_win_balance: float,
                             expected_loss_balance: float) -> dict:
    """Merkt sich die gerade eroeffnete Position, damit der naechste Lauf (wenn keine offene
    Position mehr gefunden wird) anhand des Guthabens bestimmen kann, ob SL oder TP gegriffen hat.
    """
    state['pending_position'] = {
        'balance_before': balance_before,
        'expected_win_balance': expected_win_balance,
        'expected_loss_balance': expected_loss_balance,
    }
    return state


def resolve_pending_outcome(state: dict, current_balance: float, base_pct: float,
                             growth_factor: float = 2.0, streak_target: int = 3) -> dict:
    """Bestimmt anhand des aktuellen Guthabens, ob die zuletzt eroeffnete Position als Gewinn
    oder Verlust geschlossen wurde -- es gibt keine direkte Order-Historie-Abfrage in Exchange,
    daher der indirekte Vergleich gegen die beiden bei Eroeffnung erwarteten Ausgaenge (naeher an
    Win- oder Loss-Erwartung gewinnt; robust gegen kleine Abweichungen durch Fees/Slippage).
    Aktualisiert stake_pct/consecutive_wins nach der Anti-Martingale-Regel und loescht
    pending_position. Kein Effekt, wenn keine pending_position vorliegt.
    """
    pending = state.get('pending_position')
    if pending is None:
        return state

    dist_to_win = abs(current_balance - pending['expected_win_balance'])
    dist_to_loss = abs(current_balance - pending['expected_loss_balance'])
    is_win = dist_to_win < dist_to_loss

    if is_win:
        state['consecutive_wins'] = state.get('consecutive_wins', 0) + 1
        if state['consecutive_wins'] >= streak_target:
            state['stake_pct'] = base_pct
            state['consecutive_wins'] = 0
        else:
            state['stake_pct'] = state.get('stake_pct', base_pct) * growth_factor
    else:
        state['consecutive_wins'] = 0
        state['stake_pct'] = base_pct

    logger.info(f"Anti-Martingale: letzte Position als {'GEWINN' if is_win else 'VERLUST'} erkannt "
                f"(Guthaben={current_balance:.2f}, erwartet Win={pending['expected_win_balance']:.2f}/"
                f"Loss={pending['expected_loss_balance']:.2f}). Naechster Einsatz: {state['stake_pct']:.2f}%.")
    state['pending_position'] = None
    return state
