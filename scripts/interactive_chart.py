# scripts/interactive_chart.py
# OracleBot Interaktive Charts (Modus 4)
#
# Zeigt echte Tageskerzen (Candlestick) UND die vom Modell auf denselben (Out-of-Sample-)
# Tagen prognostizierten Kerzen im selben Chart -- als eigene, klar unterscheidbare Ebene
# (durchsichtiger + etwas breiter als die echten Kerzen), damit auf einen Blick sichtbar
# ist, wie nah Prognose und Realitaet beieinander liegen.
#
# Die Prognose-Kerze nutzt NUR trend+range (reconstruct_simple_candle), nicht die volle
# Docht-/Close-Geometrie -- die volle Rekonstruktion wuerde Praezision vortaeuschen, die das
# Modell nicht hat (Lehre aus pbot: lieber wenig und ehrlich vorhersagen als viel und falsch).
# trend/range kommen inzwischen selbst vom RandomForest-Hybrid-Ensemble (siehe
# tree_ensemble.py), nicht mehr direkt vom Transformer -- auch range war trotz plausibel
# aussehender Accuracy grossteils kollabiert (2026-07-10, per Chart-Vergleich entdeckt).
#
# Nur Out-of-Sample-Beispiele (Validierungsmenge, wie beim Signal-Backtest) werden gezeigt --
# Trainings-Beispiele wuerden auswendig gelernte (zu optimistische) Treffer vorspiegeln.
import argparse
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import torch

from backtest_signal import (
    ARTIFACTS_DIR, load_ohlcv_by_symbol, load_settings, load_model_and_scaler, load_tree_ensemble,
    load_val_examples_by_symbol,
)
from oraclebot.model.reconstruct import reconstruct_simple_candle
from oraclebot.strategy.backtest import predict_for_examples

GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
RED = '\033[0;31m'
CYAN = '\033[0;36m'
BOLD = '\033[1m'
NC = '\033[0m'

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'charts')

# go.Candlestick berechnet seine Breite selbst (~±0.35 bei ganzzahligem x-Abstand) -- REAL_HALF_WIDTH
# ist nur Dokumentation dieser Annahme, kein Parameter, den wir setzen koennen. PRED_HALF_WIDTH muss
# DEUTLICH darueber liegen, sonst verschwindet der Breiten-Unterschied bei vielen Kerzen im Antialiasing
# (bei ±0.42 war er praktisch unsichtbar -- siehe Screenshot-Feedback vom 2026-07-09).
REAL_HALF_WIDTH = 0.35
PRED_HALF_WIDTH = 0.58   # spuerbar breiter als die echte Kerze, Kanten ragen sichtbar heraus
PRED_OPACITY = 0.40      # Fuellung durchsichtig; der Rand (siehe _add_candle_shapes) bleibt kraeftig+gestrichelt

TREND_COLOR = {0: '#ef5350', 1: '#26a69a'}  # bearish, bullish -- kein Neutral-Bucket mehr
REAL_UP_COLOR = '#26a69a'
REAL_DOWN_COLOR = '#ef5350'


def _rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'


def _add_candle_shapes(fig, idx: int, o: float, h: float, l: float, c: float, half_width: float,
                        color: str, opacity: float, name: str, legend_group: str, show_legend: bool,
                        hover_text: str, row: int = 1, col: int = 1):
    import plotly.graph_objects as go

    body_top, body_bottom = max(o, c), min(o, c)
    # Rand bleibt kraeftig (fast deckend) UND gestrichelt -- nur die Flaeche ist durchsichtig.
    # Ein rein transparenter Rand ging bei dichten Charts (viele Kerzen/wenig Pixel pro Kerze)
    # im echten Kerzenkoerper visuell unter; gestrichelt+kraeftig liest sich auch bei Ueberlappung
    # eindeutig als "Prognose-Ebene", nicht als Teil der echten Kerze.
    line_color = _rgba(color, 0.95)
    fill_color = _rgba(color, opacity)

    # Docht (volle High-Low-Linie), dahinter -- der Body-Trace wird darueber gezeichnet.
    fig.add_trace(go.Scatter(
        x=[idx, idx], y=[l, h], mode='lines',
        line=dict(color=line_color, width=2, dash='dot'),
        name=name, legendgroup=legend_group, showlegend=False,
        hoverinfo='skip',
    ), row=row, col=col)

    # Body als gefuelltes Rechteck -- Breite und Deckkraft frei steuerbar (im Gegensatz zu
    # go.Candlestick, das Breite/Opacity nicht pro Trace anpassbar macht).
    fig.add_trace(go.Scatter(
        x=[idx - half_width, idx + half_width, idx + half_width, idx - half_width, idx - half_width],
        y=[body_bottom, body_bottom, body_top, body_top, body_bottom],
        mode='lines', fill='toself',
        line=dict(color=line_color, width=2.5, dash='dot'),
        fillcolor=fill_color,
        name=name, legendgroup=legend_group, showlegend=show_legend,
        text=hover_text, hoverinfo='text',
    ), row=row, col=col)


def generate_chart(symbol: str, start_date: str, end_date: str, settings: dict) -> str:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    device = torch.device('cpu')
    model, scaler = load_model_and_scaler(settings, device)
    tree_ensemble = load_tree_ensemble()

    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']

    symbols = ds_cfg.get('symbols', ['BTC/USDT:USDT'])
    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in symbols)
    dataset_path = os.path.join(ARTIFACTS_DIR, f"{symbols_tag}_full.jsonl")
    if not os.path.exists(dataset_path):
        raise RuntimeError(f"Kein Datensatz gefunden: {dataset_path}. Erst train_transformer.py ausfuehren.")

    val_by_symbol = load_val_examples_by_symbol(dataset_path, train_cfg['val_split'])
    examples = val_by_symbol.get(symbol, [])
    if not examples:
        print(f"{RED}Keine Out-of-Sample-Beispiele fuer {symbol} gefunden.{NC}")
        return ''

    ohlcv_by_symbol = load_ohlcv_by_symbol([symbol], model_cfg['timeframes'], train_cfg['history_days'])
    daily_df = ohlcv_by_symbol[symbol]['1d']

    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    real_df = daily_df[(daily_df.index >= start_ts) & (daily_df.index < end_ts)]
    if real_df.empty:
        print(f"{RED}Keine echten Kerzen im Zeitraum {start_date} bis {end_date} fuer {symbol}.{NC}")
        return ''

    examples_in_range = [ex for ex in examples if start_ts <= pd.Timestamp(ex['date']) < end_ts]
    print(f"INFO: {len(real_df)} echte Kerzen, {len(examples_in_range)} Out-of-Sample-Prognosen im Zeitraum.")

    predictions = predict_for_examples(
        examples_in_range, model, scaler, ohlcv_by_symbol, model_cfg['timeframes'],
        beam_width=model_cfg['beam_width'], device=device, tree_ensemble=tree_ensemble) if examples_in_range else []

    n_bars = len(real_df)
    x_idx = list(range(n_bars))
    n_ticks = min(20, n_bars)
    tick_step = max(1, n_bars // n_ticks)
    tick_vals = list(range(0, n_bars, tick_step))
    tick_text = [str(real_df.index[i])[:10] for i in tick_vals]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.8, 0.2], subplot_titles=['', 'Volumen'],
    )

    real_colors = [REAL_UP_COLOR if c >= o else REAL_DOWN_COLOR for o, c in zip(real_df['open'], real_df['close'])]
    fig.add_trace(go.Candlestick(
        x=x_idx, open=real_df['open'], high=real_df['high'], low=real_df['low'], close=real_df['close'],
        name='Real', increasing_line_color=REAL_UP_COLOR, increasing_fillcolor=REAL_UP_COLOR,
        decreasing_line_color=REAL_DOWN_COLOR, decreasing_fillcolor=REAL_DOWN_COLOR,
        showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=x_idx, y=real_df['volume'], marker_color=real_colors, name='Volumen', showlegend=False, opacity=0.6,
    ), row=2, col=1)

    date_to_idx = {ts: i for i, ts in enumerate(real_df.index)}
    n_matched, n_trend_correct = 0, 0
    for pred in predictions:
        pred_ts = pd.Timestamp(pred['date'])
        if pred_ts not in date_to_idx:
            continue
        idx = date_to_idx[pred_ts]
        trend = pred['prediction']['trend']
        range_cat = pred['prediction']['range']
        # NUR trend+range verwenden -- die einzigen beiden Targets mit belegter Vorhersagekraft
        # (OOS-Tests 2026-07-09). Die volle Docht-/Close-Geometrie (reconstruct_candle) wuerde
        # Praezision vortaeuschen, die das Modell nicht hat -- siehe reconstruct_simple_candle().
        coords = reconstruct_simple_candle(pred['prev_close'], pred['atr'], trend, range_cat)
        color = TREND_COLOR.get(trend, '#ffca28')
        trend_label = {0: 'bearish', 1: 'bullish'}.get(trend, '?')
        confidence = pred['prediction'].get('step_probabilities', {}).get('trend', 0.0)
        n_matched += 1
        real_bar = real_df.loc[pred_ts]
        actual_up = real_bar['close'] > real_bar['open']
        n_trend_correct += int((trend == 1) == actual_up)

        hover = (f"Prognose {str(pred_ts)[:10]}<br>Trend: {trend_label} ({confidence:.0%})<br>"
                 f"O:{coords['open']:.2f} H:{coords['high']:.2f} L:{coords['low']:.2f} C:{coords['close']:.2f}")
        _add_candle_shapes(
            fig, idx, coords['open'], coords['high'], coords['low'], coords['close'],
            half_width=PRED_HALF_WIDTH, color=color, opacity=PRED_OPACITY,
            name='Prognose (OOS)', legend_group='prediction', show_legend=(n_matched == 1),
            hover_text=hover, row=1, col=1)

    trend_hitrate_pct = (n_trend_correct / n_matched * 100) if n_matched else 0.0
    title_text = (f"{symbol} 1d — OracleBot Prognose-Overlay | Zeitraum: {start_date} bis {end_date} | "
                  f"{n_matched} OOS-Prognosen | Trend-Trefferquote: {trend_hitrate_pct:.0f}%")

    fig.update_layout(
        title=dict(text=title_text, font=dict(size=13), x=0.5, xanchor='center'),
        template='plotly_dark',
        xaxis_rangeslider_visible=False,
        legend=dict(orientation='h', yanchor='bottom', y=1.01, xanchor='center', x=0.5, font=dict(size=11)),
        height=850, margin=dict(l=60, r=40, t=90, b=40),
    )
    xaxis_cfg = dict(tickmode='array', tickvals=tick_vals, ticktext=tick_text, tickangle=-45, title='Datum')
    fig.update_xaxes(xaxis_cfg)
    fig.update_yaxes(title_text='Preis', row=1, col=1)
    fig.update_yaxes(title_text='Vol', row=2, col=1)

    os.makedirs(CHARTS_DIR, exist_ok=True)
    safe_symbol = symbol.replace('/', '').replace(':', '')
    ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    chart_path = os.path.join(CHARTS_DIR, f'chart_{safe_symbol}_1d_{ts_stamp}.html')
    fig.write_html(chart_path, include_plotlyjs='inline')
    return chart_path


def run_interactive_chart(symbol: str = None, start_date: str = None, end_date: str = None, last_days: int = None):
    print(f"\n{CYAN}========== INTERAKTIVE CHARTS (OracleBot) =========={NC}\n")
    settings = load_settings()
    symbols = settings['dataset_settings'].get('symbols', ['BTC/USDT:USDT'])

    if symbol is None:
        print(f"{BOLD}Verfuegbare Symbole:{NC}")
        for idx, s in enumerate(symbols, 1):
            print(f"  {idx}) {s}")
        raw = input('\nAuswahl [Standard: 1]: ').strip()
        try:
            symbol = symbols[int(raw) - 1] if raw else symbols[0]
        except (ValueError, IndexError):
            symbol = symbols[0]

    if start_date is None and end_date is None and last_days is None:
        raw = input('Startdatum (JJJJ-MM-TT) [leer=vor 180 Tagen]: ').strip()
        start_date = raw if raw else (datetime.now(timezone.utc) - timedelta(days=180)).strftime('%Y-%m-%d')
        raw = input('Enddatum (JJJJ-MM-TT) [leer=heute]: ').strip()
        end_date = raw if raw else datetime.now(timezone.utc).strftime('%Y-%m-%d')
        raw = input('Letzten N Tage anzeigen (ueberschreibt Start/Ende) [leer=aus]: ').strip()
        if raw:
            try:
                last_days = int(raw)
            except ValueError:
                pass

    if last_days:
        end_dt = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=last_days)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
    else:
        start_date = start_date or (datetime.now(timezone.utc) - timedelta(days=180)).strftime('%Y-%m-%d')
        end_date = end_date or datetime.now(timezone.utc).strftime('%Y-%m-%d')

    print(f"\nINFO: Verarbeite {symbol} ({start_date} bis {end_date})...")
    try:
        path = generate_chart(symbol, start_date, end_date, settings)
    except ImportError:
        print(f'{RED}Fehler: plotly nicht installiert.{NC}')
        return
    if path:
        print(f"\nINFO: {GREEN}Chart gespeichert: {path}{NC}")
    else:
        print(f"\n{RED}Kein Chart generiert.{NC}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', type=str, default=None)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    parser.add_argument('--last-days', type=int, default=None)
    args = parser.parse_args()
    run_interactive_chart(args.symbol, args.start_date, args.end_date, args.last_days)
