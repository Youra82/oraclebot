# src/oraclebot/strategy/signal.py
# Risiko-basierte Positionsgroesse, geteilt von live_trade.py fuer beide Strategien (nur
# genutzt, wenn anti_martingale_enabled=false -- siehe anti_martingale.py fuer die Alternative).


def compute_position_size(balance: float, risk_per_trade_pct: float, entry: float, stop_loss: float) -> float:
    """Risiko-basierte Positionsgroesse: (balance * risk%) / SL-Abstand.

    KEIN volles Kapital pro Trade -- nur der Betrag, der bei SL-Treffer verloren gehen darf.
    """
    sl_distance_price = abs(entry - stop_loss)
    if sl_distance_price <= 0:
        return 0.0
    risk_amount = balance * (risk_per_trade_pct / 100.0)
    return risk_amount / sl_distance_price
