import openpyxl
import pandas as pd
import pytest

import scripts.show_results as sr
from oraclebot.analysis.evaluation import run_anti_martingale_backtest

BASE_CFG = {'barrier_pct': 1.0, 'leverage': 100, 'backtest_start_capital': 15.0,
            'taker_fee_rate_pct': 0.06, 'symbol': 'BTC/USDT:USDT', 'reference_timeframe': '4h',
            'anti_martingale_base_pct': 5.0, 'anti_martingale_growth_factor': 2.0,
            'anti_martingale_streak_target': 3}


def make_trade(entry_time, frac, outcome):
    ts = pd.Timestamp(entry_time, tz='UTC')
    return {'entry_time': ts, 'exit_time': ts, 'entry': 60000.0, 'exit': 60000.0,
            'direction': 'long', 'frac': frac, 'outcome': outcome}


@pytest.fixture
def isolated_charts_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sr, 'CHARTS_DIR', str(tmp_path))
    return tmp_path


def test_since_filter_restarts_capital_curve_fresh_instead_of_carrying_over_prior_compounding(
        isolated_charts_dir):
    # Vier Gewinn-Trades, die ersten zwei VOR dem since-Datum -- ein voller (ungefilterter)
    # Anti-Martingale-Lauf laesst das Kapital vor dem since-Datum bereits deutlich wachsen.
    trades = [
        make_trade('2026-01-01', 0.05, 'win'),
        make_trade('2026-01-02', 0.05, 'win'),
        make_trade('2026-02-01', 0.05, 'win'),
        make_trade('2026-02-02', 0.05, 'win'),
    ]
    run_anti_martingale_backtest(trades, BASE_CFG)  # simuliert den vollen OOS-Lauf in show_results.py
    inflated_third_trade_equity = trades[2]['equity_after']

    outfile = sr.generate_excel(trades, BASE_CFG, since='2026-02-01')
    assert outfile is not None

    # Bugfix-Kern: eine unabhaengige, frische Simulation NUR der beiden gefilterten Trades muss
    # exakt denselben Endstand ergeben wie generate_excel() intern berechnet -- keine
    # Uebernahme des bereits (durch die ersten beiden Trades) aufgezinsten Zwischenstands.
    fresh_trades = [make_trade('2026-02-01', 0.05, 'win'), make_trade('2026-02-02', 0.05, 'win')]
    run_anti_martingale_backtest(fresh_trades, BASE_CFG)

    wb = openpyxl.load_workbook(outfile)
    ws = wb.active
    header = [c.value for c in ws[1]]
    capital_col = header.index('Gesamtkapital') + 1
    written_capitals = [ws.cell(row=r, column=capital_col).value for r in (2, 3)]

    assert written_capitals == pytest.approx([fresh_trades[0]['equity_after'], fresh_trades[1]['equity_after']],
                                              rel=1e-6)
    # Vor dem Fix waere hier der (durch die ersten beiden Trades) bereits aufgezinste Stand
    # gelandet -- deutlich hoeher als der frische Wert.
    assert written_capitals[0] < inflated_third_trade_equity


def test_since_filter_with_no_matching_trades_returns_none(isolated_charts_dir):
    trades = [make_trade('2026-01-01', 0.05, 'win')]
    run_anti_martingale_backtest(trades, BASE_CFG)
    assert sr.generate_excel(trades, BASE_CFG, since='2027-01-01') is None


def test_without_since_filter_keeps_full_period_capital_curve(isolated_charts_dir):
    trades = [make_trade('2026-01-01', 0.05, 'win'), make_trade('2026-01-02', 0.05, 'win')]
    run_anti_martingale_backtest(trades, BASE_CFG)
    expected_final_equity = trades[-1]['equity_after']

    outfile = sr.generate_excel(trades, BASE_CFG, since=None)
    wb = openpyxl.load_workbook(outfile)
    ws = wb.active
    header = [c.value for c in ws[1]]
    capital_col = header.index('Gesamtkapital') + 1
    assert ws.cell(row=3, column=capital_col).value == pytest.approx(expected_final_equity, rel=1e-6)
