# src/oraclebot/strategy/barrier_signal.py
# Uebersetzt eine BarrierPredictor-Vorhersage in ein konkretes Handelssignal. Bewusst deutlich
# einfacher als signal.py's compute_trade_signal(): das Barriere-Ziel ist immer symmetrisch
# (SL-Abstand == TP-Abstand == derselbe barrier_pct, mit dem das Modell trainiert wurde), es
# gibt keine range-/wick-basierte SL/TP-Berechnung.


def compute_barrier_signal(predicted_class: int, confidence: float, entry_price: float,
                            min_confidence: float = 0.60, barrier_pct: float = 1.0) -> dict:
    """Args:
        predicted_class: 0 (runter zuerst) oder 1 (hoch zuerst), von BarrierPredictor.predict_one().
        confidence: Wahrscheinlichkeit der vorhergesagten Klasse.
        entry_price: aktueller Schlusskurs (Referenzkerze).
        min_confidence: kein Trade, wenn confidence darunter liegt.
        barrier_pct: symmetrischer SL/TP-Abstand in Prozent vom Entry (muss zum trainierten
            Modell passen -- siehe barrier_targets.py).

    Returns:
        dict mit 'direction' ('long'/'short'/None), 'entry', 'stop_loss', 'take_profit',
        'sl_distance', 'tp_distance', 'confidence', 'reason' (nur gesetzt wenn kein Trade).
    """
    if confidence < min_confidence:
        return {'direction': None, 'reason': 'low_confidence', 'confidence': confidence}

    direction = 'long' if predicted_class == 1 else 'short'
    distance = entry_price * barrier_pct / 100.0
    if direction == 'long':
        stop_loss, take_profit = entry_price - distance, entry_price + distance
    else:
        stop_loss, take_profit = entry_price + distance, entry_price - distance

    return {
        'direction': direction,
        'entry': entry_price,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sl_distance': distance,
        'tp_distance': distance,
        'confidence': confidence,
    }
