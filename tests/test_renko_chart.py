import os

import numpy as np
import pandas as pd

from oraclebot.analysis.renko_chart import generate_renko_chart


def make_ohlcv(closes, start="2024-01-01", freq="5min"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"open": closes, "high": closes + 0.05, "low": closes - 0.05,
                          "close": closes}, index=idx)


def test_generate_renko_chart_writes_html_and_returns_stats(tmp_path):
    closes = 100 + np.cumsum(np.sin(np.linspace(0, 20, 400)) * 0.6)
    df = make_ohlcv(closes)
    out_path = str(tmp_path / "chart.html")

    stats = generate_renko_chart(df, "TEST/USDT:USDT", base_pct=0.01, k_entropy=0.5, h_window=5,
                                  horizontal_lookback=4, breakout_run=2, start_capital=100.0,
                                  out_path=out_path)

    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 1000
    assert stats["n_bricks"] > 0
    assert "n_trades" in stats and "win_rate" in stats and "pnl_pct" in stats


def test_generate_renko_chart_raises_on_no_bricks(tmp_path):
    df = make_ohlcv([100.0])
    out_path = str(tmp_path / "chart.html")
    try:
        generate_renko_chart(df, "TEST/USDT:USDT", base_pct=0.01, k_entropy=0.5, h_window=5,
                              horizontal_lookback=4, breakout_run=2, start_capital=100.0,
                              out_path=out_path)
        assert False, "haette ValueError werfen muessen"
    except ValueError:
        pass
