# src/oraclebot/utils/chart_png.py
# Statisches PNG (matplotlib) fuer den Telegram-Versand -- die interaktive Plotly-HTML-Datei
# aus interactive_chart.py ist fuer Backtests/Analyse gedacht, nicht fuer eine Chat-Nachricht.
# Kein neuer Abhaengigkeits-Zweig noetig: matplotlib ist bereits vorhanden, im Gegensatz zu
# Plotly-Bildexport (braucht zusaetzlich `kaleido`).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from oraclebot.model.reconstruct import reconstruct_candle

UP_COLOR = '#26a69a'
DOWN_COLOR = '#ef5350'
BODY_ALPHA = 0.6
# Dochte sind ein schwaecheres, weniger validiertes Ziel als trend/range (siehe
# reconstruct_candle()-Docstring) -- deutlich niedrigere Alpha macht das auch optisch klar:
# der Body ist die verlaessliche Prognose, die Dochte sind eine ungefaehre Ergaenzung.
WICK_ALPHA = 0.25


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


def _draw_hourly_volatility_panel(ax, hourly_profile: dict, predicted_range: float):
    """Historisches stuendliches Volatilitaets-Profil (siehe analysis/hourly_volatility.py),
    skaliert auf die heute vorhergesagte Tages-Range -- KEINE Pfad-Vorhersage, nur die
    historisch typische Groessenordnung pro Stunde. Fehlerbalken = Tag-zu-Tag-Streuung in der
    Historie: schmal = verlaesslicheres Muster, breit = mit Vorsicht zu geniessen. Bewusst
    keine einzelne "Konfidenz"-Kennzahl (ein erster Versuch mit 1/(1+Variationskoeffizient)
    unterschied kaum zwischen den Stunden und taeuschte Praezision vor, die die Streuung nicht
    hergibt -- die Fehlerbalken selbst sind die ehrlichere Darstellung)."""
    hours = sorted(hourly_profile.keys())
    means = [hourly_profile[h]['mean'] * predicted_range for h in hours]
    stds = [hourly_profile[h]['std'] * predicted_range for h in hours]

    ax.bar(hours, means, color='#5b8def', alpha=0.7, width=0.7)
    ax.errorbar(hours, means, yerr=stds, fmt='none', ecolor='#333333', alpha=0.4, capsize=2)
    ax.set_xlabel('Stunde (UTC)')
    ax.set_ylabel('USDT')
    ax.set_title('Historisches stuendliches Volatilitaets-Profil (skaliert, Fehlerbalken = Streuung)', fontsize=9)
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.2)


def plot_prediction_chart(daily_df: pd.DataFrame, prev_close: float, atr: float, trend: int, range_cat: int,
                           close_position_cat: int, upper_wick_cat: int, lower_wick_cat: int,
                           target_date, save_path: str, n_recent: int = 30, hourly_profile: dict = None):
    """Baut ein kompaktes PNG: die letzten `n_recent` echten Tageskerzen + die vorhergesagte
    Kerze (gestrichelt). Body (trend+range, verlaesslicher) in normaler Deckkraft, Dochte
    (upper_wick/lower_wick/close_position, schwaecher validiert) deutlich transparenter --
    siehe reconstruct_candle()-Docstring fuer die Begruendung der unterschiedlichen
    Verlaesslichkeit.

    `hourly_profile` (optional, von hourly_volatility.load_profile()): wenn angegeben, wird ein
    zweites Panel darunter angehaengt (siehe _draw_hourly_volatility_panel())."""
    recent = daily_df.iloc[-n_recent:]
    pred = reconstruct_candle(prev_close, atr, trend, range_cat, close_position_cat, upper_wick_cat, lower_wick_cat)

    if hourly_profile:
        fig, (ax, ax_hourly) = plt.subplots(
            2, 1, figsize=(9, 6.5), dpi=120, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(9, 5), dpi=120)

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
    ax.grid(alpha=0.2)

    if hourly_profile:
        predicted_range = pred['high'] - pred['low']
        _draw_hourly_volatility_panel(ax_hourly, hourly_profile, predicted_range)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
