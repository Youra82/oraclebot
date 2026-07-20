# src/oraclebot/strategy/signal.py
# Uebersetzt eine Modell-Vorhersage in ein konkretes Handelssignal (Richtung, Entry, SL, TP).
#
# Bewusste Design-Entscheidung: SL/TP haengen NICHT an den exakten rekonstruierten
# Body-/Wick-Koordinaten (reconstruct.py), sondern nur an `trend` (Richtung) und `range`
# (Groesse). Diese beiden -- wie alle geometrischen Ziele ausser high_first/gap_yn -- kommen
# seit 2026-07-10 vom RandomForest-Ensemble (tree_ensemble.py), nicht mehr vom Transformer:
# `range` sah anhand der reinen Accuracy zunaechst wie der zuverlaessigste Kopf aus, war aber
# tatsaechlich zu 131/132 Beispielen auf einen einzigen Bucket kollabiert (ein Chart-Vergleich
# mit den echten Kerzen deckte das auf, die Accuracy allein hatte es verschleiert).
from oraclebot.model.reconstruct import RANGE_BUCKET_VALUES


def compute_trade_signal(prediction: dict, prev_close: float, atr: float,
                          min_trend_confidence: float = 0.40,
                          sl_range_fraction: float = 0.5,
                          risk_reward: float = 2.0,
                          manual_sl_pct: float = None,
                          manual_tp_pct: float = None) -> dict:
    """Berechnet Richtung, Entry, Stop-Loss und Take-Profit aus einer Modell-Vorhersage.

    Args:
        prediction: Rueckgabe von MarketTransformer.predict_beam() (braucht 'trend', 'range',
            'step_probabilities').
        prev_close: Anker-Preis (letzter bekannter Schlusskurs, siehe reconstruct.py).
        atr: Average True Range der Referenz-Kerze (absoluter Preis, keine Ratio).
        min_trend_confidence: Kein Trade, wenn die Trend-Wahrscheinlichkeit darunter liegt
            (Baseline bei 2 Klassen ist 50% -- ein sinnvoller Schwellwert liegt darueber).
        sl_range_fraction: SL-Abstand = sl_range_fraction * range_atr_multiple * ATR.
            0.5 heisst: SL liegt bei der Haelfte der vom Modell erwarteten Tagesrange.
            Wird ignoriert, wenn manual_sl_pct gesetzt ist.
        risk_reward: TP-Abstand = risk_reward * SL-Abstand. Wird ignoriert, wenn
            manual_tp_pct gesetzt ist.
        manual_sl_pct: wenn gesetzt (zusammen mit manual_tp_pct), ersetzt SL/TP-Abstand
            komplett durch feste Prozentsaetze vom Entry-Preis (settings.json
            strategy_settings.manual_sl_pct/manual_tp_pct) statt der Modell-Berechnung.
            2.0 heisst: SL 2% vom Entry entfernt -- reine Preis-Prozent-Angabe, der Hebel
            wird hier NICHT einberechnet (der wirkt erst auf die Positionsgroesse/Margin,
            nicht auf den Preisabstand).
        manual_tp_pct: analog zu manual_sl_pct fuer den Take-Profit.

    Returns:
        dict mit 'direction' ('long'/'short'/None -- None heisst kein Trade), 'entry',
        'stop_loss', 'take_profit', 'sl_distance', 'tp_distance', 'confidence',
        'range_atr_multiple' (None im manuellen Modus), 'reason' (nur gesetzt wenn kein Trade).
    """
    trend = prediction['trend']
    confidence = prediction['step_probabilities']['trend']

    if confidence < min_trend_confidence:
        return {'direction': None, 'reason': 'low_confidence', 'confidence': confidence}

    if manual_sl_pct is not None and manual_tp_pct is not None:
        range_atr_multiple = None
        sl_distance = prev_close * manual_sl_pct / 100.0
        tp_distance = prev_close * manual_tp_pct / 100.0
    else:
        range_atr_multiple = RANGE_BUCKET_VALUES[prediction['range']]
        sl_distance = sl_range_fraction * range_atr_multiple * atr
        tp_distance = risk_reward * sl_distance

    direction = 'long' if trend == 1 else 'short'
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
