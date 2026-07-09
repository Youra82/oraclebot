# src/oraclebot/model/reconstruct.py
# Rueckuebersetzung der kategorialen Vorhersage (trend/range/close_position/wicks) in
# konkrete Preis-Koordinaten: High (Docht oben), Body oben, Body unten, Low (Docht unten).
#
# Die 5 kategorialen Ziele sind NICHT alle unabhaengige Freiheitsgrade: gegeben trend
# (Richtung), range (Gesamtgroesse) und die beiden Wick-Anteile ist die Kerzenform bereits
# vollstaendig bestimmt -- close_position ist geometrisch redundant (close_position =
# (Close-Low)/(High-Low) folgt zwingend aus den anderen vieren). Wir nutzen close_position
# deshalb nicht zur Rekonstruktion, sondern als Konsistenz-Check gegen das, was trend+wicks
# ergeben (siehe 'close_position_consistent' im Rueckgabewert).
from oraclebot.data.targets import RANGE_LABELS, WICK_LABELS

# Reprae­sentative Werte je Bucket = Bin-Mittelpunkte derselben Grenzen wie in targets.py.
RANGE_BUCKET_VALUES = [0.25, 0.75, 1.5, 2.5]           # zu RANGE_LABELS-Bins [0,0.5,1,2,inf]
CLOSE_POSITION_BUCKET_VALUES = [1 / 6, 0.5, 5 / 6]      # zu Bins [0,1/3,2/3,1]
WICK_BUCKET_VALUES = [0.075, 0.25, 0.45]                # zu WICK_LABELS-Bins [0,0.15,0.35,inf]


def reconstruct_candle(prev_close: float, atr: float, trend: int, range_cat: int,
                        close_position_cat: int, upper_wick_cat: int, lower_wick_cat: int) -> dict:
    """Rekonstruiert Preis-Koordinaten aus den kategorialen Modell-Vorhersagen.

    Annahme (Standard bei durchgehend handelbaren Krypto-Perpetuals): Open der neuen
    Kerze = Close der letzten bekannten Kerze (`prev_close`), kein Gap.

    Args:
        prev_close: Schlusskurs der letzten bekannten (Referenz-)Kerze -- Anker fuer Open.
        atr: Average True Range der letzten bekannten Kerze (absoluter Preis, keine Ratio).
        trend: 0=bearish, 1=neutral, 2=bullish (siehe targets.TREND_LABELS).
        range_cat, close_position_cat, upper_wick_cat, lower_wick_cat: Kategorie-Indizes
            wie von MarketTransformer.predict_beam() zurueckgegeben.

    Returns:
        dict mit open, close, high, low, body_top, body_bottom, upper_wick_size,
        lower_wick_size (in Preiseinheiten) sowie close_position_consistent (bool):
        stimmt die geometrisch implizierte Close-Position mit dem vorhergesagten
        close_position-Bucket ueberein?
    """
    open_price = prev_close
    range_price = RANGE_BUCKET_VALUES[range_cat] * atr
    upper_wick_ratio = WICK_BUCKET_VALUES[upper_wick_cat]
    lower_wick_ratio = WICK_BUCKET_VALUES[lower_wick_cat]
    body_ratio = max(0.0, 1.0 - upper_wick_ratio - lower_wick_ratio)

    if trend == 2:    # bullish
        close_price = open_price + body_ratio * range_price
    elif trend == 0:  # bearish
        close_price = open_price - body_ratio * range_price
    else:             # neutral
        close_price = open_price

    body_top = max(open_price, close_price)
    body_bottom = min(open_price, close_price)
    high = body_top + upper_wick_ratio * range_price
    low = body_bottom - lower_wick_ratio * range_price

    hl_range = high - low
    implied_cp = (close_price - low) / hl_range if hl_range > 0 else 0.5
    implied_cp_cat = 0 if implied_cp < 1 / 3 else (2 if implied_cp > 2 / 3 else 1)

    return {
        'open': open_price,
        'close': close_price,
        'high': high,
        'low': low,
        'body_top': body_top,
        'body_bottom': body_bottom,
        'upper_wick_size': high - body_top,
        'lower_wick_size': body_bottom - low,
        'close_position_consistent': implied_cp_cat == close_position_cat,
    }
