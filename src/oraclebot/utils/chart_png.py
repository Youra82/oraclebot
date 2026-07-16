# src/oraclebot/utils/chart_png.py
# Statisches PNG (matplotlib) fuer den Telegram-Versand -- die interaktive Plotly-HTML-Datei
# aus interactive_chart.py ist fuer Backtests/Analyse gedacht, nicht fuer eine Chat-Nachricht.
# Kein neuer Abhaengigkeits-Zweig noetig: matplotlib ist bereits vorhanden, im Gegensatz zu
# Plotly-Bildexport (braucht zusaetzlich `kaleido`).
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from oraclebot.model.reconstruct import reconstruct_candle

UP_COLOR = '#26a69a'
DOWN_COLOR = '#ef5350'
BODY_ALPHA = 0.6
# Dochte sind ein schwaecheres, weniger validiertes Ziel als trend/range (siehe
# reconstruct_candle()-Docstring) -- deutlich niedrigere Alpha macht das auch optisch klar:
# der Body ist die verlaessliche Prognose, die Dochte sind eine ungefaehre Ergaenzung.
WICK_ALPHA = 0.25
PO3_LEVEL_COLOR = '#9575cd'


def _draw_candle(ax, x: int, o: float, h: float, l: float, c: float, width: float = 0.6,
                  color: str = None, alpha: float = 1.0, linestyle: str = 'solid'):
    color = color or (UP_COLOR if c >= o else DOWN_COLOR)
    ax.plot([x, x], [l, h], color=color, linewidth=1, alpha=alpha, linestyle=linestyle, zorder=2)
    body_bottom, body_top = min(o, c), max(o, c)
    ax.bar(x, body_top - body_bottom, bottom=body_bottom, width=width, color=color,
           alpha=alpha, edgecolor=color, linewidth=1, linestyle=linestyle, zorder=3)


def _draw_predicted_candle(ax, x: int, o: float, h: float, l: float, c: float, width: float = 0.6):
    """Wie _draw_candle, aber Docht und Body getrennt eingefaerbt: Body in BODY_ALPHA (die
    verlaessliche trend+range-Prognose), Docht-Linien in WICK_ALPHA (upper_wick/lower_wick --
    ungefaehr, schwaecher validiert). Gestrichelter Rand bleibt das Merkmal "das ist eine
    Prognose, keine echte Kerze"."""
    color = UP_COLOR if c >= o else DOWN_COLOR
    body_bottom, body_top = min(o, c), max(o, c)
    ax.plot([x, x], [l, body_bottom], color=color, linewidth=1, alpha=WICK_ALPHA, linestyle='dashed', zorder=2)
    ax.plot([x, x], [body_top, h], color=color, linewidth=1, alpha=WICK_ALPHA, linestyle='dashed', zorder=2)
    ax.bar(x, body_top - body_bottom, bottom=body_bottom, width=width, color=color,
           alpha=BODY_ALPHA, edgecolor=color, linewidth=1, linestyle='dashed', zorder=3)


def _draw_po3_levels(ax, recent: pd.DataFrame, n_candles: int = 3):
    """PO3 (Power of Three)-Widerstands-/Unterstuetzungszonen: horizontale Linien auf Hoehe von
    Body (Open/Close) und Docht (High/Low) der letzten `n_candles` ECHTEN Tageskerzen -- Preis-
    Niveaus, an denen kuerzlich Kaeufer/Verkaeufer reagiert haben und die deshalb oft erneut als
    Widerstand/Unterstuetzung wirken. Body-Niveaus deutlicher als Docht-Niveaus (gleiche
    Verlaesslichkeits-Abstufung wie beim Rest des Charts)."""
    last_candles = recent.iloc[-n_candles:]
    for _, row in last_candles.iterrows():
        body_top, body_bottom = max(row['open'], row['close']), min(row['open'], row['close'])
        ax.axhline(body_top, color=PO3_LEVEL_COLOR, linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        ax.axhline(body_bottom, color=PO3_LEVEL_COLOR, linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        ax.axhline(row['high'], color=PO3_LEVEL_COLOR, linestyle=':', linewidth=0.8, alpha=0.3, zorder=1)
        ax.axhline(row['low'], color=PO3_LEVEL_COLOR, linestyle=':', linewidth=0.8, alpha=0.3, zorder=1)

    body_handle = mlines.Line2D([], [], color=PO3_LEVEL_COLOR, linestyle='--', linewidth=0.8, alpha=0.7,
                                 label=f'Body-Niveau (letzte {n_candles} Kerzen)')
    wick_handle = mlines.Line2D([], [], color=PO3_LEVEL_COLOR, linestyle=':', linewidth=0.8, alpha=0.5,
                                 label=f'Docht-Niveau (letzte {n_candles} Kerzen)')
    ax.legend(handles=[body_handle, wick_handle], loc='upper left', fontsize=7, framealpha=0.7)


def _draw_hourly_path_panel(ax, path_profile: dict, trend: int, predicted_open: float,
                             predicted_low: float, predicted_high: float):
    """Fuellt die vorhergesagte Tageskerze mit einer plausiblen stuendlichen Kerzenfolge:
    historisch typische Position innerhalb der Tagesspanne pro Stunde, getrennt nach
    bullischen/baerischen Tagen (siehe analysis/hourly_volatility.compute_hourly_path_profile()),
    skaliert auf die HEUTIGE Prognose (Trend + Range). Gezeichnet mit derselben _draw_candle()-
    Funktion wie die echten Kerzen oben -- also echte rot/gruen-Faerbung je nach Stunden-
    Richtung, kein neutraler Platzhalter.

    WICHTIG: das ist KEINE Pfad-Vorhersage fuer heute im engeren Sinne -- es zeigt, wie sich
    Tage mit dieser Trendrichtung historisch im Schnitt stundenweise aufgebaut haben, auf die
    heutige Range projiziert. Der Docht (Streuung, WICK_ALPHA) zeigt die Tag-zu-Tag-Variation
    dieses Musters -- schmal = konsistentes historisches Muster, breit = mit Vorsicht zu
    geniessen (bewusst keine einzelne "Konfidenz"-Zahl, siehe fruehere Version dieser Funktion
    in der Git-Historie fuer die Begruendung)."""
    hours_data = path_profile.get(trend, path_profile.get(str(trend), {}))
    hours = sorted(hours_data.keys())
    predicted_range = predicted_high - predicted_low

    prev_close_price = predicted_open
    for h in hours:
        mean_frac = hours_data[h]['mean']
        std_frac = hours_data[h]['std']
        close_price = predicted_low + mean_frac * predicted_range
        wick_extra = std_frac * predicted_range / 2
        o, c = prev_close_price, close_price
        color = UP_COLOR if c >= o else DOWN_COLOR
        body_bottom, body_top = min(o, c), max(o, c)
        ax.plot([h, h], [body_bottom - wick_extra, body_top + wick_extra], color=color,
                linewidth=1, alpha=WICK_ALPHA, zorder=2)
        ax.bar(h, body_top - body_bottom, bottom=body_bottom, width=0.6, color=color,
               alpha=BODY_ALPHA, edgecolor=color, linewidth=1, zorder=3)
        prev_close_price = close_price

    ax.axhline(predicted_low, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.axhline(predicted_high, color='gray', linestyle=':', linewidth=0.8, alpha=0.4)
    ax.set_xlabel('Stunde (UTC)')
    ax.set_ylabel('USDT')
    trend_word = 'bullisch' if trend == 1 else 'baerisch'
    ax.set_title(f"Historisch typischer stuendlicher Pfad fuer {trend_word} vorhergesagte Tage (skaliert)", fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.grid(alpha=0.2)


def plot_prediction_chart(daily_df: pd.DataFrame, prev_close: float, atr: float, trend: int, range_cat: int,
                           close_position_cat: int, upper_wick_cat: int, lower_wick_cat: int,
                           target_date, save_path: str, n_recent: int = 30, hourly_path_profile: dict = None):
    """Baut ein kompaktes PNG: die letzten `n_recent` echten Tageskerzen + die vorhergesagte
    Kerze (gestrichelt). Body (trend+range, verlaesslicher) in normaler Deckkraft, Dochte
    (upper_wick/lower_wick/close_position, schwaecher validiert) deutlich transparenter --
    siehe reconstruct_candle()-Docstring fuer die Begruendung der unterschiedlichen
    Verlaesslichkeit.

    `hourly_path_profile` (optional, von hourly_volatility.load_path_profile()): wenn angegeben,
    wird ein zweites Panel darunter angehaengt (siehe _draw_hourly_path_panel())."""
    recent = daily_df.iloc[-n_recent:]
    pred = reconstruct_candle(prev_close, atr, trend, range_cat, close_position_cat, upper_wick_cat, lower_wick_cat)

    if hourly_path_profile:
        fig, (ax, ax_hourly) = plt.subplots(
            2, 1, figsize=(9, 6.5), dpi=120, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

    _draw_po3_levels(ax, recent, n_candles=3)

    for i, (_, row) in enumerate(recent.iterrows()):
        _draw_candle(ax, i, row['open'], row['high'], row['low'], row['close'])

    # Farbe folgt der tatsaechlich vorhergesagten Richtung (wie bei den echten Kerzen) --
    # NICHT ein fixes PRED_COLOR: das zeigte vorher unabhaengig von trend immer dasselbe
    # blasse Blau an, was bei einer SHORT-Prognose faelschlich neutral/gruenlich statt
    # erkennbar rot wirkte (Nutzer-Meldung 2026-07-10).
    pred_x = len(recent)
    _draw_predicted_candle(ax, pred_x, pred['open'], pred['high'], pred['low'], pred['close'])

    tick_positions = list(range(0, len(recent), max(1, len(recent) // 8))) + [pred_x]
    tick_labels = [recent.index[i].strftime('%m-%d') for i in tick_positions if i < len(recent)] + [
        f"{target_date.strftime('%m-%d')}?"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    trend_label = 'LONG' if trend == 1 else 'SHORT'
    ax.set_title(f"BTC/USDT -- Prognose {target_date.strftime('%Y-%m-%d')}: {trend_label} (gestrichelt)")
    ax.set_ylabel("USDT")
    # Dichtere Preis-Ticks auf der linken Achse (Standard war nur alle ~2000 USDT, zu grob
    # zum Ablesen einzelner Kurspreise) -- MaxNLocator statt fixem Abstand, damit es sich
    # automatisch an unterschiedliche Preisspannen anpasst.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=14))
    ax.grid(alpha=0.2)

    if hourly_path_profile:
        _draw_hourly_path_panel(ax_hourly, hourly_path_profile, trend,
                                 predicted_open=pred['open'], predicted_low=pred['low'], predicted_high=pred['high'])

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
