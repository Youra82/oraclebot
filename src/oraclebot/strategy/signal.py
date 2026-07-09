# src/oraclebot/strategy/signal.py
# Uebersetzt eine Modell-Vorhersage in ein konkretes Handelssignal (Richtung, Entry, SL, TP).
#
# Bewusste Design-Entscheidung: SL/TP haengen NICHT an den exakten rekonstruierten
# Body-/Wick-Koordinaten (reconstruct.py), sondern werden aus `trend` (Richtung) und `range`
# (Groesse) kalibriert -- laut Out-of-Sample-Auswertung ist `range` der zuverlaessigste Kopf
# (50-59% Genauigkeit, Baseline 25%), waehrend `upper_wick`/`lower_wick` teils nicht besser
# als Raten sind (23-44%, Baseline 33%). Die exakten Docht-Koordinaten als harte SL/TP-Grenze
# zu verwenden wuerde die staerkste Schwaeche des Modells direkt ins Risiko-Management vererben.
from oraclebot.model.reconstruct import RANGE_BUCKET_VALUES


def compute_trade_signal(prediction: dict, prev_close: float, atr: float,
                          min_trend_confidence: float = 0.40,
                          sl_range_fraction: float = 0.5,
                          risk_reward: float = 2.0) -> dict:
    """Berechnet Richtung, Entry, Stop-Loss und Take-Profit aus einer Modell-Vorhersage.

    Args:
        prediction: Rueckgabe von MarketTransformer.predict_beam() (braucht 'trend', 'range',
            'step_probabilities').
        prev_close: Anker-Preis (letzter bekannter Schlusskurs, siehe reconstruct.py).
        atr: Average True Range der Referenz-Kerze (absoluter Preis, keine Ratio).
        min_trend_confidence: Kein Trade, wenn die Trend-Wahrscheinlichkeit darunter liegt
            (Baseline bei 3 Klassen ist 33% -- ein sinnvoller Schwellwert liegt darueber).
        sl_range_fraction: SL-Abstand = sl_range_fraction * range_atr_multiple * ATR.
            0.5 heisst: SL liegt bei der Haelfte der vom Modell erwarteten Tagesrange.
        risk_reward: TP-Abstand = risk_reward * SL-Abstand.

    Returns:
        dict mit 'direction' ('long'/'short'/None -- None heisst kein Trade), 'entry',
        'stop_loss', 'take_profit', 'sl_distance', 'tp_distance', 'confidence',
        'range_atr_multiple', 'reason' (nur gesetzt wenn kein Trade).
    """
    trend = prediction['trend']
    confidence = prediction['step_probabilities']['trend']

    if trend == 1:
        return {'direction': None, 'reason': 'neutral_trend', 'confidence': confidence}
    if confidence < min_trend_confidence:
        return {'direction': None, 'reason': 'low_confidence', 'confidence': confidence}

    range_atr_multiple = RANGE_BUCKET_VALUES[prediction['range']]
    sl_distance = sl_range_fraction * range_atr_multiple * atr
    tp_distance = risk_reward * sl_distance

    direction = 'long' if trend == 2 else 'short'
    entry = prev_close
    if direction == 'long':
        stop_loss = entry - sl_distance
        take_profit = entry + tp_distance
    else:
        stop_loss = entry + sl_distance
        take_profit = entry - tp_distance

    return {
        'direction': direction,
        'entry': entry,
        'stop_loss': stop_loss,
        'take_profit': take_profit,
        'sl_distance': sl_distance,
        'tp_distance': tp_distance,
        'confidence': confidence,
        'range_atr_multiple': range_atr_multiple,
    }


def compute_position_size(balance: float, risk_per_trade_pct: float, entry: float, stop_loss: float) -> float:
    """Risiko-basierte Positionsgroesse (wie bei mbot/dnabot): (balance * risk%) / SL-Abstand.

    KEIN volles Kapital pro Trade -- nur der Betrag, der bei SL-Treffer verloren gehen darf.
    """
    sl_distance_price = abs(entry - stop_loss)
    if sl_distance_price <= 0:
        return 0.0
    risk_amount = balance * (risk_per_trade_pct / 100.0)
    return risk_amount / sl_distance_price
