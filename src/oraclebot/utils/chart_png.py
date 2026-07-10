# src/oraclebot/utils/chart_png.py
# Statisches PNG (matplotlib) fuer den Telegram-Versand -- die interaktive Plotly-HTML-Datei
# aus interactive_chart.py ist fuer Backtests/Analyse gedacht, nicht fuer eine Chat-Nachricht.
# Kein neuer Abhaengigkeits-Zweig noetig: matplotlib ist bereits vorhanden, im Gegensatz zu
# Plotly-Bildexport (braucht zusaetzlich `kaleido`).
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from oraclebot.model.reconstruct import reconstruct_simple_candle

UP_COLOR = '#26a69a'
DOWN_COLOR = '#ef5350'
PRED_COLOR = '#42a5f5'


def _draw_candle(ax, x: int, o: float, h: float, l: float, c: float, width: float = 0.6,
                  color: str = None, alpha: float = 1.0, linestyle: str = 'solid'):
    color = color or (UP_COLOR if c >= o else DOWN_COLOR)
    ax.plot([x, x], [l, h], color=color, linewidth=1, alpha=alpha, linestyle=linestyle, zorder=2)
    body_bottom, body_top = min(o, c), max(o, c)
    ax.bar(x, body_top - body_bottom, bottom=body_bottom, width=width, color=color,
           alpha=alpha, edgecolor=color, linewidth=1, linestyle=linestyle, zorder=3)


def plot_prediction_chart(daily_df: pd.DataFrame, prev_close: float, atr: float, trend: int, range_cat: int,
                           target_date, save_path: str, n_recent: int = 30):
    """Baut ein kompaktes PNG: die letzten `n_recent` echten Tageskerzen + die vorhergesagte
    Kerze (gestrichelt, nur trend+range -- siehe reconstruct_simple_candle() fuer die Begruendung,
    warum NICHT die volle wick-Geometrie gezeigt wird)."""
    recent = daily_df.iloc[-n_recent:]
    pred = reconstruct_simple_candle(prev_close, atr, trend, range_cat)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=120)
    for i, (_, row) in enumerate(recent.iterrows()):
        _draw_candle(ax, i, row['open'], row['high'], row['low'], row['close'])

    pred_x = len(recent)
    _draw_candle(ax, pred_x, pred['open'], pred['high'], pred['low'], pred['close'],
                 color=PRED_COLOR, alpha=0.5, linestyle='dashed')

    tick_positions = list(range(0, len(recent), max(1, len(recent) // 8))) + [pred_x]
    tick_labels = [recent.index[i].strftime('%m-%d') for i in tick_positions if i < len(recent)] + [
        f"{target_date.strftime('%m-%d')}?"]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')

    trend_label = 'LONG' if trend == 1 else 'SHORT'
    ax.set_title(f"BTC/USDT -- Prognose {target_date.strftime('%Y-%m-%d')}: {trend_label} (gestrichelt)")
    ax.set_ylabel("USDT")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path
