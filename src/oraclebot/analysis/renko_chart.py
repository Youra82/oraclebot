# src/oraclebot/analysis/renko_chart.py
# Interaktive Illustration eines einzelnen Symbols der Renko-Breakout-Strategie -- analog zu
# zerobots interactive_chart.py (gleiche visuelle Sprache: Bricks als Candlestick, Entry/Exit als
# Marker, Equity auf zweiter Y-Achse), aber an die tatsaechliche Signallogik hier angepasst:
# Entry braucht eine vorausgehende Seitwaertsphase (horizontal_lookback Bricks gemischter
# Richtung), Exit ist immer der erste Gegen-Brick (kein festes TP/SL wie bei zerobots
# N-Bricks-Trendfolge).
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from oraclebot.data.ear_bricks import build_ear_bricks
from oraclebot.strategy.horizontal_breakout_signal import backtest_horizontal_breakout

UP_COLOR = "#26a69a"
DOWN_COLOR = "#ef5350"


def generate_renko_chart(df, symbol: str, base_pct: float, k_entropy: float, h_window: int,
                          horizontal_lookback: int, breakout_run: int, start_capital: float,
                          out_path: str) -> dict:
    """Baut die EAR-Bricks fuer `df` (OHLCV, DatetimeIndex), simuliert die Strategie darauf und
    speichert einen interaktiven Plotly-HTML-Chart unter `out_path`.

    Zeigt: Bricks als Candlestick (Brick-Index als X-Achse, nicht Zeit -- Renko-Konvention),
    die Seitwaerts-Fenster vor jedem Entry grau schattiert (macht sichtbar, WARUM genau dort
    ein Trade eroeffnet wurde), Entry-Dreiecke, Exit-Marker (Kreis=Win, Kreuz=Loss), und die
    Kapitalkurve auf einer zweiten Y-Achse.

    Returns:
        dict mit 'n_bricks', 'n_trades', 'win_rate', 'pnl_pct', 'final_capital'.
    """
    bricks = build_ear_bricks(df, base_pct=base_pct, k_entropy=k_entropy, h_window=h_window)
    if not bricks:
        raise ValueError(f"{symbol}: keine Bricks aus den gegebenen Daten erzeugt.")

    trades = backtest_horizontal_breakout(bricks, horizontal_lookback, breakout_run)

    n_bricks = len(bricks)
    x_idx = list(range(n_bricks))
    brick_closes = [b["close"] for b in bricks]
    brick_opens = [brick_closes[i - 1] if i > 0 else b["close"] for i, b in enumerate(bricks)]
    brick_highs = [max(o, c) for o, c in zip(brick_opens, brick_closes)]
    brick_lows = [min(o, c) for o, c in zip(brick_opens, brick_closes)]
    colors = [UP_COLOR if b["direction"] == "up" else DOWN_COLOR for b in bricks]

    n_ticks = min(20, n_bricks)
    tick_step = max(1, n_bricks // n_ticks)
    tick_vals = list(range(0, n_bricks, tick_step))
    tick_text = [str(bricks[i]["ts"])[:16] for i in tick_vals]

    # Reine Illustration -- keine Positionsgroesse/Hebel/Fees hier (das ist Sache des
    # Realismus-Backtests, siehe run_renko_breakout.py/export_oos_trades.py), nur ob ein Trade
    # gewonnen/verloren hat und die Kapitalkurve bei 100% Risiko je Trade rein zur visuellen
    # Nachvollziehbarkeit (NICHT die reale Live-Positionsgroesse).
    capital = start_capital
    equity_x, equity_y = [0], [capital]
    for t in trades:
        capital *= (1 + t["pnl_pct"] / 100.0)
        equity_x.append(t["exit_idx"])
        equity_y.append(capital)

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    hover_text = [
        f"Brick #{i}<br>Zeit: {str(bricks[i]['ts'])[:16]}<br>Close: {bricks[i]['close']:.6f}<br>"
        f"Richtung: {'▲ UP' if bricks[i]['direction'] == 'up' else '▼ DOWN'}"
        for i in range(n_bricks)
    ]
    fig.add_trace(go.Candlestick(
        x=x_idx, open=brick_opens, high=brick_highs, low=brick_lows, close=brick_closes,
        name="Bricks", increasing_line_color=UP_COLOR, increasing_fillcolor=UP_COLOR,
        decreasing_line_color=DOWN_COLOR, decreasing_fillcolor=DOWN_COLOR,
        text=hover_text, hoverinfo="text",
    ), row=1, col=1, secondary_y=False)

    # Seitwaerts-Fenster vor jedem Entry grau schattieren -- macht die Entry-Bedingung sichtbar.
    for t in trades:
        run_start = t["entry_idx"] - breakout_run + 1
        window_start = run_start - horizontal_lookback
        if window_start < 0:
            continue
        fig.add_vrect(x0=window_start - 0.5, x1=run_start - 0.5, fillcolor="rgba(255,255,255,0.08)",
                      line_width=0, row=1, col=1)

    if trades:
        long_e = [t for t in trades if t["direction"] == "long"]
        short_e = [t for t in trades if t["direction"] == "short"]
        wins = [t for t in trades if t["pnl_pct"] > 0]
        losses = [t for t in trades if t["pnl_pct"] <= 0]

        if long_e:
            fig.add_trace(go.Scatter(
                x=[t["entry_idx"] for t in long_e], y=[t["entry_price"] for t in long_e],
                mode="markers", marker=dict(symbol="triangle-up", size=14, color=UP_COLOR,
                                             line=dict(color="#ffffff", width=1)),
                name="Entry Long ▲",
                hovertemplate="Entry Long<br>Brick %{x}<br>Preis: %{y:.6f}<extra></extra>",
            ), row=1, col=1, secondary_y=False)
        if short_e:
            fig.add_trace(go.Scatter(
                x=[t["entry_idx"] for t in short_e], y=[t["entry_price"] for t in short_e],
                mode="markers", marker=dict(symbol="triangle-down", size=14, color="#ffa726",
                                             line=dict(color="#ffffff", width=1)),
                name="Entry Short ▼",
                hovertemplate="Entry Short<br>Brick %{x}<br>Preis: %{y:.6f}<extra></extra>",
            ), row=1, col=1, secondary_y=False)
        if wins:
            fig.add_trace(go.Scatter(
                x=[t["exit_idx"] for t in wins], y=[t["exit_price"] for t in wins],
                mode="markers", marker=dict(symbol="circle", size=12, color="#00bcd4",
                                             line=dict(color="#ffffff", width=1)),
                name="Exit Win ✓",
                hovertemplate="Exit (Gegen-Brick)<br>Brick %{x}<br>Preis: %{y:.6f}<br>"
                              "PnL: %{customdata:.2f}%<extra></extra>",
                customdata=[t["pnl_pct"] for t in wins],
            ), row=1, col=1, secondary_y=False)
        if losses:
            fig.add_trace(go.Scatter(
                x=[t["exit_idx"] for t in losses], y=[t["exit_price"] for t in losses],
                mode="markers", marker=dict(symbol="x", size=13, color=DOWN_COLOR,
                                             line=dict(color=DOWN_COLOR, width=3)),
                name="Exit Loss ✗",
                hovertemplate="Exit (Gegen-Brick)<br>Brick %{x}<br>Preis: %{y:.6f}<br>"
                              "PnL: %{customdata:.2f}%<extra></extra>",
                customdata=[t["pnl_pct"] for t in losses],
            ), row=1, col=1, secondary_y=False)

    fig.add_trace(go.Scatter(
        x=equity_x, y=equity_y, mode="lines", line=dict(color="#5c9bd6", width=1.5),
        name="Kapital (illustrativ, 100% Risiko/Trade)",
        hovertemplate="Kapital: %{y:.2f}<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate = (n_wins / n_trades * 100) if trades else 0.0
    final_capital = equity_y[-1]
    pnl_pct = (final_capital / start_capital - 1) * 100

    title = (f"{symbol} — oraclebot Renko-Breakout | Bricks: {n_bricks} | Trades: {n_trades} | "
             f"WR: {win_rate:.1f}% | PnL (illustrativ): {pnl_pct:+.1f}%")
    fig.update_layout(
        title=dict(text=title, font=dict(size=13), x=0.5, xanchor="center"),
        template="plotly_dark", xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
                    font=dict(size=11)),
        height=650, margin=dict(l=60, r=80, t=90, b=40),
        yaxis2=dict(title="Kapital", showgrid=False, tickfont=dict(color="#5c9bd6"),
                    title_font=dict(color="#5c9bd6")),
    )
    fig.update_xaxes(tickmode="array", tickvals=tick_vals, ticktext=tick_text, tickangle=-45,
                      title="Brick-Index (Zeit)")
    fig.update_yaxes(title_text="Preis", row=1, col=1, secondary_y=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.write_html(out_path)

    return {"n_bricks": n_bricks, "n_trades": n_trades, "win_rate": win_rate,
            "pnl_pct": pnl_pct, "final_capital": final_capital}
