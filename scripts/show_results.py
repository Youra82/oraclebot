# scripts/show_results.py
# Zeigt Trainings-Diagnose + einen vollen Anti-Martingale-Backtest (inkl. Gebuehren) fuer das
# aktuell trainierte Barriere-Modell an. Analog zum show_results.py/.sh-Muster der anderen
# Bots -- oraclebot hat nur EIN Symbol/Modell (kein Multi-Coin/Genome-Discovery wie dnabot),
# daher deutlich einfacher: keine Coin/Timeframe-Auswahl noetig, nur drei Modi.
import argparse
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from oraclebot.data.dataset import load_dataset_jsonl
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.strategy.anti_martingale import compute_margin, record_pending_position, resolve_pending_outcome
from oraclebot.strategy.barrier_signal import compute_barrier_signal

PROJECT_ROOT = os.path.join(os.path.dirname(__file__), '..')
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'datasets')
CHARTS_DIR = os.path.join(PROJECT_ROOT, 'artifacts', 'charts')


def load_secrets() -> dict:
    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        return {}
    with open(secret_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_settings() -> dict:
    with open(os.path.join(PROJECT_ROOT, 'settings.json'), 'r', encoding='utf-8') as f:
        return json.load(f)


def load_predictions(barrier_cfg: dict):
    """Laedt Modell + gecachten Trainings-Datensatz, gibt die Modell-Vorhersagen fuer den
    Out-of-Sample-Validierungsanteil zurueck (chronologisch letzte val_split% der Beispiele --
    exakt derselbe Anteil, der beim Training als Holdout diente)."""
    symbol = barrier_cfg['symbol']
    reference_tf = barrier_cfg['reference_timeframe']
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    model_path = os.path.join(ARTIFACTS_DIR, f'barrier_model_{safe_symbol}_{reference_tf}.pkl')
    dataset_path = os.path.join(ARTIFACTS_DIR, f'barrier_{safe_symbol}_{reference_tf}.jsonl')
    if not os.path.exists(model_path):
        logger.error(f"Kein Modell gefunden: {model_path}. Erst train_barrier_model.py "
                     f"(bzw. run_pipeline.sh) ausfuehren.")
        sys.exit(1)
    if not os.path.exists(dataset_path):
        logger.error(f"Kein Trainings-Datensatz gefunden: {dataset_path}. Erst "
                     f"train_barrier_model.py ausfuehren (Datensatz-Cache ist nicht in Git).")
        sys.exit(1)

    predictor = BarrierPredictor.load(model_path)
    examples = load_dataset_jsonl(dataset_path)
    examples.sort(key=lambda e: e['date'])
    n_val = max(1, int(len(examples) * barrier_cfg['val_split']))
    val_examples = examples[-n_val:]

    preds = []
    for ex in val_examples:
        cls, conf = predictor.predict_one(ex['features'])
        preds.append({'date': ex['date'], 'entry': ex['entry'], 'cls': cls, 'conf': conf,
                      'label': ex['target'], 'exit_time': pd.Timestamp(ex['exit_time'])})
    preds.sort(key=lambda p: p['date'])
    return preds, safe_symbol


def build_trades(preds: list, barrier_cfg: dict) -> list:
    """Baut serielle Trades aus den Vorhersagen: nur Signale >= min_confidence, ein Trade
    laeuft bis zu seinem eigenen exit_time, alle Referenzkerzen darin werden uebersprungen
    (kein Stacking, wie live_trade.py es auch nicht erlaubt)."""
    barrier_pct = barrier_cfg['barrier_pct']
    min_conf = barrier_cfg['min_confidence']
    trades = []
    idx = 0
    while idx < len(preds):
        p = preds[idx]
        signal = compute_barrier_signal(p['cls'], p['conf'], p['entry'],
                                         min_confidence=min_conf, barrier_pct=barrier_pct)
        if signal['direction'] is None:
            idx += 1
            continue
        won = (p['cls'] == p['label'])
        pnl_price = signal['tp_distance'] if won else -signal['sl_distance']
        exit_price = signal['take_profit'] if won else signal['stop_loss']
        trades.append({
            'entry_time': pd.Timestamp(p['date']), 'exit_time': p['exit_time'], 'entry': p['entry'],
            'exit': exit_price, 'direction': signal['direction'], 'frac': pnl_price / p['entry'],
            'outcome': 'win' if won else 'loss',
        })
        while idx < len(preds) and preds[idx]['date'] <= p['exit_time'].isoformat():
            idx += 1
    return trades


def run_anti_martingale_backtest(trades: list, barrier_cfg: dict) -> dict:
    """Simuliert die Anti-Martingale-Positionsgroesse (echte Module aus anti_martingale.py)
    MIT Taker-Gebuehren (settings.json: taker_fee_rate_pct, Entry+Exit = 2 Taker-Fills) --
    ohne Gebuehren wirkt jede Kalibrierung bei 100x Hebel deutlich zu optimistisch (siehe
    README, Rekalibrierung 2026-07-26). Schreibt margin_used/pnl_usdt/equity_after direkt in
    die trades-Dicts (fuer den Excel-Export weiterverwendbar)."""
    leverage = barrier_cfg['leverage']
    barrier_pct = barrier_cfg['barrier_pct']
    start_capital = barrier_cfg.get('backtest_start_capital', 15.0)
    fee_rate = barrier_cfg.get('taker_fee_rate_pct', 0.06) / 100.0
    am_base = barrier_cfg['anti_martingale_base_pct']
    am_growth = barrier_cfg['anti_martingale_growth_factor']
    am_streak_target = int(barrier_cfg['anti_martingale_streak_target'])

    # anti_martingale.resolve_pending_outcome() loggt pro Position (sinnvoll im Live-Betrieb,
    # 1 Aufruf pro 4h) -- bei einem 684-Trade-Backtest waere das reine Zeilen-Rauschen.
    logging.getLogger('oraclebot.strategy.anti_martingale').setLevel(logging.WARNING)

    am_state = {'stake_pct': am_base, 'consecutive_wins': 0, 'pending_position': None}
    capital = start_capital
    peak = capital
    max_dd = 0.0
    for t in trades:
        margin = compute_margin(capital, am_state)
        balance_before = capital
        distance = t['entry'] * barrier_pct / 100.0
        contracts = (margin * leverage) / t['entry']
        fee = margin * leverage * fee_rate * 2
        pnl_usd = t['frac'] * leverage * margin - fee
        capital += pnl_usd
        capital = max(capital, 0.0)
        peak = max(peak, capital)
        max_dd = max(max_dd, (peak - capital) / peak if peak > 0 else 0.0)
        expected_win_balance = balance_before + distance * contracts - fee
        expected_loss_balance = balance_before - distance * contracts - fee
        am_state = record_pending_position(am_state, balance_before, expected_win_balance, expected_loss_balance)
        am_state = resolve_pending_outcome(am_state, capital, am_base, am_growth, am_streak_target)
        t['margin_used'] = margin
        t['pnl_usdt'] = pnl_usd
        t['equity_after'] = capital
    return {'end_capital': capital, 'start_capital': start_capital, 'max_dd_pct': max_dd * 100}


def print_summary(trades: list, backtest: dict, safe_symbol: str, reference_tf: str):
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    win_rate = wins / len(trades) * 100 if trades else 0.0
    pnl_pct = (backtest['end_capital'] - backtest['start_capital']) / backtest['start_capital'] * 100

    diagnostics_path = os.path.join(ARTIFACTS_DIR, f'barrier_diagnostics_{safe_symbol}_{reference_tf}.json')
    logger.info('=' * 70)
    logger.info('TRAININGS-DIAGNOSE (letzter train_barrier_model.py-Lauf)')
    logger.info('=' * 70)
    if os.path.exists(diagnostics_path):
        with open(diagnostics_path, 'r', encoding='utf-8') as f:
            diag = json.load(f)
        logger.info(f"Beispiele gesamt: {diag['n_examples']} (Train={diag['n_train']} Val={diag['n_val']})")
        logger.info(f"70/30-Split: In-Sample={diag['train_accuracy']:.1%} Out-of-Sample={diag['val_accuracy']:.1%}")
        wf = diag['walk_forward_accuracies']
        logger.info(f"Walk-Forward ({len(wf)} Fenster): {['%.1f%%' % (a*100) for a in wf]}")
        logger.info(f"  Mittel={diag['walk_forward_mean']:.1%} Worst-Case={diag['walk_forward_worst_case']:.1%}")
    else:
        logger.info(f"Keine Diagnose-Datei gefunden ({diagnostics_path}).")

    logger.info('')
    logger.info('=' * 70)
    logger.info('BACKTEST MIT ANTI-MARTINGALE (Out-of-Sample, inkl. Gebuehren)')
    logger.info('=' * 70)
    logger.info(f"Trades={len(trades)} WinRate={win_rate:.1f}%")
    logger.info(f"Startkapital={backtest['start_capital']:.2f} USDT -> Endkapital={backtest['end_capital']:.2e} USDT "
                f"({'+' if pnl_pct >= 0 else ''}{pnl_pct:.1f}%)")
    logger.info(f"MaxDD={backtest['max_dd_pct']:.1f}%")
    logger.info("Hinweis: Endkapital bei vielen Trades + Compounding wird schnell astronomisch --")
    logger.info("nur als relativer Vergleichswert zwischen Konfigurationen aussagekraeftig, siehe README.")


def generate_chart(trades: list, barrier_cfg: dict, backtest: dict, safe_symbol: str):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib nicht installiert -- Chart uebersprungen (pip install matplotlib).")
        return None

    from oraclebot.utils.data_fetch import fetch_all_timeframes

    reference_tf = barrier_cfg['reference_timeframe']
    symbol = barrier_cfg['symbol']
    start_capital = barrier_cfg.get('backtest_start_capital', 15.0)

    ohlcv = fetch_all_timeframes(symbol, [reference_tf], barrier_cfg['history_days'],
                                  cache_dir=ARTIFACTS_DIR, use_cache=True)
    price_df = ohlcv[reference_tf]

    wins = [t for t in trades if t['outcome'] == 'win']
    liqs = [t for t in trades if t['outcome'] == 'loss']

    # Kapitalkurve war bereits Teil von run_anti_martingale_backtest() (equity_after pro Trade).
    dates_ = [trades[0]['entry_time']] + [t['entry_time'] for t in trades]
    capitals = [start_capital] + [t['equity_after'] for t in trades]

    streak_values = []
    current_streak, current_type = 0, None
    for t in trades:
        if t['outcome'] == current_type:
            current_streak += 1
        else:
            current_type = t['outcome']
            current_streak = 1
        streak_values.append(current_streak if t['outcome'] == 'win' else -current_streak)
    max_win_streak = max(streak_values)
    max_liq_streak = abs(min(streak_values))

    plot_start = trades[0]['entry_time'] - pd.Timedelta(days=3)
    plot_end = trades[-1]['entry_time'] + pd.Timedelta(days=3)
    price_zoom = price_df[(price_df.index >= plot_start) & (price_df.index <= plot_end)]

    fig, (ax_price, ax_equity, ax_streak) = plt.subplots(
        3, 1, figsize=(15, 12), dpi=120, gridspec_kw={'height_ratios': [2, 1.3, 0.9]}, sharex=False)

    ax_price.plot(price_zoom.index, price_zoom['close'], color='#888888', linewidth=0.9, alpha=0.7,
                  label=f'{symbol.split("/")[0]} Close ({reference_tf})')
    ax_price.scatter([t['entry_time'] for t in wins], [t['entry'] for t in wins],
                      marker='^', color='#26a69a', s=18, zorder=3, alpha=0.75, label=f'Win (n={len(wins)})')
    ax_price.scatter([t['entry_time'] for t in liqs], [t['entry'] for t in liqs],
                      marker='v', color='#ef5350', s=18, zorder=3, alpha=0.75, label=f'Liquidation (n={len(liqs)})')
    win_rate = len(wins) / len(trades) * 100
    ax_price.set_ylabel('USDT')
    ax_price.set_title(f'{symbol} -- {reference_tf}-Barriere-Modell, conf>={barrier_cfg["min_confidence"]:.2f} '
                        f'(SL=TP={barrier_cfg["barrier_pct"]:.0f}%, Hebel={barrier_cfg["leverage"]}x): '
                        f'{len(trades)} Trades, {len(wins)}W/{len(liqs)}L ({win_rate:.1f}% Winrate)')
    ax_price.legend(loc='upper left', fontsize=9)
    ax_price.grid(alpha=0.2)
    ax_price.set_xlim(plot_start, plot_end)

    ax_equity.plot(dates_, capitals, color='#ab47bc', linewidth=1.1,
                    label=f'Anti-Martingale (Basis={barrier_cfg["anti_martingale_base_pct"]:.2f}%) '
                          f'-> {capitals[-1]:.2e} USDT')
    ax_equity.axhline(start_capital, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax_equity.set_yscale('log')
    ax_equity.set_ylabel('USDT (log)')
    ax_equity.set_title(f'Kapitalkurve (Start={start_capital:.0f} USDT, inkl. Gebuehren) '
                         f'-- MaxDD={backtest["max_dd_pct"]:.1f}%')
    ax_equity.legend(loc='upper left', fontsize=9)
    ax_equity.grid(alpha=0.2, which='both')

    colors2 = ['#26a69a' if v > 0 else '#ef5350' for v in streak_values]
    ax_streak.bar([t['entry_time'] for t in trades], streak_values, color=colors2, width=0.15)
    ax_streak.axhline(0, color='black', linewidth=0.8)
    ax_streak.set_ylabel('Serienlaenge')
    ax_streak.set_xlabel('Datum')
    ax_streak.set_title(f'Laufende Serie -- laengste Gewinn-Serie={max_win_streak}, laengste Liq-Serie={max_liq_streak}')
    ax_streak.grid(alpha=0.2)

    for ax in (ax_price, ax_equity, ax_streak):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax.set_xlim(plot_start, plot_end)

    fig.tight_layout()
    os.makedirs(CHARTS_DIR, exist_ok=True)
    outfile = os.path.join(CHARTS_DIR, 'combined_overview.png')
    fig.savefig(outfile)
    logger.info(f"Chart gespeichert: {outfile}")
    return outfile


def generate_excel(trades: list, barrier_cfg: dict, since: str = None):
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        logger.error("openpyxl nicht installiert -- Excel-Export uebersprungen (pip install openpyxl).")
        return None

    if since:
        since_ts = pd.Timestamp(since, tz='UTC')
        trades = [t for t in trades if t['entry_time'] >= since_ts]
        if not trades:
            logger.error(f"Keine Trades ab {since} im Out-of-Sample-Zeitraum.")
            return None

    coin = barrier_cfg['symbol'].split('/')[0]
    reference_tf = barrier_cfg['reference_timeframe']
    rows = []
    for i, t in enumerate(trades, 1):
        outcome_label = 'TP erreicht' if t['outcome'] == 'win' else 'SL erreicht'
        rows.append({
            'Nr': i,
            'Datum (Entry)': t['entry_time'].strftime('%Y-%m-%d %H:%M'),
            'Datum (Exit)': t['exit_time'].strftime('%Y-%m-%d %H:%M'),
            'Coin': coin,
            'Timeframe': reference_tf,
            'Richtung': t['direction'].upper(),
            'Entry': round(t['entry'], 2),
            'Exit': round(t['exit'], 2),
            'Ergebnis': outcome_label,
            'Margin (USDT)': round(t['margin_used'], 4),
            'PnL (USDT)': round(t['pnl_usdt'], 4),
            'Gesamtkapital': round(t['equity_after'], 4),
        })

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Trades'

    header_fill = PatternFill('solid', fgColor='1E3A5F')
    win_fill = PatternFill('solid', fgColor='D6F4DC')
    loss_fill = PatternFill('solid', fgColor='FAD7D7')
    alt_fill = PatternFill('solid', fgColor='F2F2F2')
    thin_border = Border(left=Side(style='thin', color='CCCCCC'), right=Side(style='thin', color='CCCCCC'),
                          top=Side(style='thin', color='CCCCCC'), bottom=Side(style='thin', color='CCCCCC'))
    col_widths = {'Nr': 6, 'Datum (Entry)': 18, 'Datum (Exit)': 18, 'Coin': 10, 'Timeframe': 12,
                  'Richtung': 10, 'Entry': 14, 'Exit': 14, 'Ergebnis': 14, 'Margin (USDT)': 16,
                  'PnL (USDT)': 14, 'Gesamtkapital': 16}

    headers = list(rows[0].keys())
    for col, h in enumerate(headers, 1):
        cell = ws.cell(row=1, column=col, value=h)
        cell.fill = header_fill
        cell.font = Font(bold=True, color='FFFFFF', size=11)
        cell.alignment = Alignment(horizontal='center', vertical='center')
        cell.border = thin_border
        ws.column_dimensions[get_column_letter(col)].width = col_widths.get(h, 14)
    ws.row_dimensions[1].height = 22

    for r_idx, row in enumerate(rows, 2):
        fill = win_fill if row['Ergebnis'] == 'TP erreicht' else (loss_fill if r_idx % 2 == 0 else alt_fill)
        for col, key in enumerate(headers, 1):
            cell = ws.cell(row=r_idx, column=col, value=row[key])
            cell.fill = fill
            cell.border = thin_border
            cell.alignment = Alignment(horizontal='center', vertical='center')
            if key in ('Entry', 'Exit', 'Margin (USDT)', 'PnL (USDT)', 'Gesamtkapital'):
                cell.number_format = '#,##0.0000'
        ws.row_dimensions[r_idx].height = 18

    start_capital = barrier_cfg.get('backtest_start_capital', 15.0)
    end_capital = rows[-1]['Gesamtkapital']
    pnl_pct = (end_capital - start_capital) / start_capital * 100
    win_rate = sum(1 for t in trades if t['outcome'] == 'win') / len(trades) * 100
    sign = '+' if pnl_pct >= 0 else ''

    summary_row = len(rows) + 3
    ws.cell(row=summary_row, column=1, value='Zusammenfassung').font = Font(bold=True, size=11)
    summary_row += 1
    am_base = barrier_cfg['anti_martingale_base_pct']
    am_growth = barrier_cfg['anti_martingale_growth_factor']
    am_streak = barrier_cfg['anti_martingale_streak_target']
    fee_pct = barrier_cfg.get('taker_fee_rate_pct', 0.06)
    for label, value in [
        ('Trades gesamt', len(rows)),
        ('Win-Rate', f'{win_rate:.1f}%'),
        ('PnL', f'{sign}{pnl_pct:.1f}%'),
        ('Final Equity', f'{end_capital:.2f} USDT'),
        ('Startkapital', f'{start_capital:.2f} USDT'),
        ('Anti-Martingale', f'Basis={am_base:.2f}% Growth={am_growth:.1f}x Streak-Ziel={am_streak}'),
        ('Gebuehren', f'{fee_pct:.2f}%/Seite Taker (Roundtrip), bereits in PnL/Margin eingerechnet'),
        ('Zeitraum', f"{trades[0]['entry_time'].date()} -> {trades[-1]['exit_time'].date()}"),
        ('Hinweis', 'Backtest auf Out-of-Sample-Validierungsdaten, KEIN Live-Trade-Log.'),
    ]:
        ws.cell(row=summary_row, column=1, value=label).font = Font(bold=True)
        ws.cell(row=summary_row, column=2, value=value)
        summary_row += 1

    ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(CHARTS_DIR, exist_ok=True)
    outfile = os.path.join(CHARTS_DIR, f'oraclebot_trades_{ts_stamp}.xlsx')
    wb.save(outfile)
    logger.info(f"Excel-Tabelle gespeichert: {outfile}")
    return outfile


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chart', action='store_true', help="Aktualisiert artifacts/charts/combined_overview.png.")
    parser.add_argument('--excel', action='store_true', help="Erstellt artifacts/charts/oraclebot_trades_<Zeitstempel>.xlsx.")
    parser.add_argument('--since', type=str, default=None,
                         help="Nur Trades ab diesem Datum (JJJJ-MM-TT) fuer --excel beruecksichtigen. "
                              "Kapitalkurve startet fuer den Export dann frisch bei backtest_start_capital.")
    parser.add_argument('--start-capital', type=float, default=None,
                         help="Ueberschreibt barrier_strategy_settings.backtest_start_capital fuer diesen Lauf "
                              "(betrifft Zusammenfassung, Chart und Excel-Export gleichermassen).")
    args = parser.parse_args()

    settings = load_settings()
    barrier_cfg = settings['barrier_strategy_settings']
    if args.start_capital is not None:
        barrier_cfg['backtest_start_capital'] = args.start_capital
    preds, safe_symbol = load_predictions(barrier_cfg)
    trades = build_trades(preds, barrier_cfg)
    if not trades:
        logger.error("Keine Trades im Out-of-Sample-Zeitraum (min_confidence evtl. zu hoch?).")
        sys.exit(1)

    backtest = run_anti_martingale_backtest(trades, barrier_cfg)
    print_summary(trades, backtest, safe_symbol, barrier_cfg['reference_timeframe'])

    telegram_enabled = settings.get('notification_settings', {}).get('telegram_enabled', False)
    telegram_cfg = load_secrets().get('telegram', {}) if telegram_enabled else {}
    win_rate = sum(1 for t in trades if t['outcome'] == 'win') / len(trades) * 100

    if args.chart:
        logger.info('')
        chart_path = generate_chart(trades, barrier_cfg, backtest, safe_symbol)
        if chart_path and telegram_enabled:
            from oraclebot.utils.telegram import send_photo
            caption = (f"oraclebot combined_overview.png\n"
                       f"{len(trades)} Trades, {win_rate:.1f}% Winrate, MaxDD {backtest['max_dd_pct']:.1f}%")
            send_photo(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), chart_path, caption=caption)

    if args.excel:
        logger.info('')
        excel_path = generate_excel(trades, barrier_cfg, since=args.since)
        if excel_path and telegram_enabled:
            from oraclebot.utils.telegram import send_document
            excel_trades = trades
            if args.since:
                since_ts = pd.Timestamp(args.since, tz='UTC')
                excel_trades = [t for t in trades if t['entry_time'] >= since_ts]
            excel_win_rate = sum(1 for t in excel_trades if t['outcome'] == 'win') / len(excel_trades) * 100
            since_note = f" (ab {args.since})" if args.since else ""
            caption = f"oraclebot Trade-Log-Export{since_note}: {len(excel_trades)} Trades, {excel_win_rate:.1f}% Winrate"
            send_document(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), excel_path, caption=caption)
