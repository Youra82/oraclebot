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
#
# GEMEINSAMER Streak ueber alle Symbole (Fund 2026-09-25, Mehrfach-Positionen-Umbau): stake_pct/
# consecutive_wins ist EIN gemeinsamer Zustand fuers ganze Portfolio, nicht pro Symbol -- welcher
# Trade zuerst SCHLIESST (unabhaengig von welchem Symbol), aktualisiert den Streak zuerst. Die
# fruehere indirekte Gewinn/Verlust-Erkennung ueber einen Kontostand-Vorher/Nachher-Vergleich
# (record_pending_position/resolve_pending_outcome) funktionierte nur, weil GENAU EINE Position
# gleichzeitig offen war -- der Kontostand-Delta war dadurch eindeutig einem Trade zuordenbar. Bei
# mehreren gleichzeitig offenen Positionen bewegen mehrere Trades den Kontostand gleichzeitig,
# der Trick wird uneindeutig. Ersetzt durch direkte Gewinn/Verlust-Uebergabe (is_win) vom Aufrufer
# (renko_live_trade.py), der den echten Fuellpreis der Order bzw. die echte Positions-Historie
# kennt -- kein Kontostand-Rateraten mehr noetig.
import json
import logging
import os

logger = logging.getLogger(__name__)


def load_state(path: str, base_pct: float) -> dict:
    """Laedt den Anti-Martingale-Zustand von Platte, oder erzeugt einen frischen Start-Zustand.

    Ausserhalb eines laufenden Gewinn-Streaks (consecutive_wins == 0) MUSS stake_pct exakt
    base_pct entsprechen -- wird beim Laden erzwungen, auch wenn bereits ein (dann zwangslaeufig
    veralteter) Wert gespeichert ist. Ohne diesen Sync bleibt ein Bot nach einer Config-Aenderung,
    die anti_martingale_base_pct aendert, bis zu streak_target-1 weitere Trades lang auf dem ALTEN
    Basiswert haengen (realer Fund 2026-07-28). Waehrend eines laufenden Streaks bleibt das
    Compounding bewusst unangetastet -- update_after_close() synct ohnehin bei jedem Verlust oder
    Streak-Abschluss auf den dann aktuellen base_pct."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        if not state.get('stake_pct') or state.get('consecutive_wins', 0) == 0:
            state['stake_pct'] = base_pct
        return state
    return {'stake_pct': base_pct, 'consecutive_wins': 0}


def save_state(path: str, state: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state, f, indent=2)


def compute_margin(balance: float, state: dict) -> float:
    """Positionsgroesse (Margin in USDT) fuer den naechsten Trade: aktueller Einsatz-Prozentsatz
    vom AKTUELLEN Guthaben (nicht vom Startkapital -- reines Compounding). `balance` ist das
    FREIE (nicht durch andere offene Positionen gebundene) Guthaben -- bei mehreren gleichzeitig
    offenen Positionen automatisch kleiner, weil exchange.fetch_balance_usdt() nur den freien
    Anteil zurueckgibt (selbstbegrenzend, keine zusaetzliche Logik hier noetig)."""
    return balance * state.get('stake_pct', 0.0) / 100.0


def update_after_close(state: dict, base_pct: float, growth_factor: float, streak_target: int,
                        is_win: bool) -> dict:
    """Aktualisiert stake_pct/consecutive_wins nach der Anti-Martingale-Regel, anhand eines
    BEKANNTEN Gewinn/Verlust-Ausgangs (vom Aufrufer anhand des echten Fuellpreises bzw. der
    echten Positions-Historie ermittelt -- siehe Moduldoc)."""
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

    logger.info(f"Anti-Martingale: Trade als {'GEWINN' if is_win else 'VERLUST'} verbucht. "
                f"Naechster Einsatz: {state['stake_pct']:.2f}%.")
    return state
