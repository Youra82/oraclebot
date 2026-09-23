# src/oraclebot/utils/realtime_bars.py
# Aggregiert einen rohen Trade-Tick-Strom (Preis, Zeitstempel) zu sehr feinen OHLCV-Bars
# (Standard 5 Sekunden) -- die Bruecke zwischen dem WebSocket-Tick-Feed und der bestehenden,
# bereits getesteten Brick-Logik (ear_bricks.py/renko_portfolio_state.py), die eine
# OHLCV-DataFrame mit DatetimeIndex erwartet. Reine Aggregations-Logik, kein Netzwerk-Code --
# unabhaengig vom WebSocket-Client testbar.
import pandas as pd


class BarAggregator:
    """Sammelt Ticks (ts_ms, price) fuer EIN Symbol und gibt bei jedem abgeschlossenen
    `bar_seconds`-Fenster einen fertigen Bar zurueck. Bar-Grenzen liegen auf runden
    `bar_seconds`-Vielfachen der Unix-Zeit (wie Candle-Grenzen bei OHLCV-Daten), nicht relativ
    zum ersten Tick -- macht die Bar-Zuordnung deterministisch reproduzierbar."""

    def __init__(self, bar_seconds: int = 5):
        self.bar_seconds = bar_seconds
        self._current_bar_start_ms = None
        self._open = self._high = self._low = self._close = None

    def _bar_start_ms(self, ts_ms: int) -> int:
        bar_ms = self.bar_seconds * 1000
        return (ts_ms // bar_ms) * bar_ms

    def add_tick(self, ts_ms: int, price: float) -> dict | None:
        """Verarbeitet einen Tick. Gibt einen fertigen Bar (dict) zurueck, sobald ein Tick in
        ein NEUES Zeitfenster faellt (der vorherige Bar gilt dann als abgeschlossen) -- sonst
        None. Ticks, die (z.B. durch Netzwerk-Jitter) in ein BEREITS abgeschlossenes, aelteres
        Fenster fallen wuerden, werden verworfen (kein rueckwirkendes Neuschreiben)."""
        bar_start = self._bar_start_ms(ts_ms)

        if self._current_bar_start_ms is None:
            self._current_bar_start_ms = bar_start
            self._open = self._high = self._low = self._close = price
            return None

        if bar_start < self._current_bar_start_ms:
            return None  # veralteter/verspaeteter Tick, verwerfen

        if bar_start == self._current_bar_start_ms:
            self._high = max(self._high, price)
            self._low = min(self._low, price)
            self._close = price
            return None

        # Neues Fenster begonnen -> vorherigen Bar abschliessen und zurueckgeben.
        finished = {
            "ts": pd.Timestamp(self._current_bar_start_ms, unit="ms", tz="UTC"),
            "open": self._open, "high": self._high, "low": self._low, "close": self._close,
        }
        self._current_bar_start_ms = bar_start
        self._open = self._high = self._low = self._close = price
        return finished

    def flush_stale_bars(self, now_ms: int) -> list:
        """Schliesst den aktuell laufenden Bar zwangsweise ab, falls seit dem letzten Tick
        bereits `bar_seconds` oder mehr vergangen sind, OHNE dass ein neuer Tick das ausgeloest
        hat (z.B. bei niedrigem Handelsvolumen ohne neue Trades). Erzeugt dabei einen oder
        mehrere flache Bars (open=high=low=close=letzter bekannter Preis) fuer jedes
        uebersprungene Fenster, damit die Brick-Kontinuitaet (H_roll-Fenster) nicht durch
        fehlende Zeitpunkte durcheinanderkommt. Gibt die Liste der so erzeugten Bars zurueck."""
        if self._current_bar_start_ms is None:
            return []
        bar_ms = self.bar_seconds * 1000
        out = []
        while now_ms - self._current_bar_start_ms >= bar_ms:
            finished = {
                "ts": pd.Timestamp(self._current_bar_start_ms, unit="ms", tz="UTC"),
                "open": self._open, "high": self._high, "low": self._low, "close": self._close,
            }
            out.append(finished)
            self._current_bar_start_ms += bar_ms
            self._open = self._high = self._low = self._close = finished["close"]
        return out


def bars_to_df(bars: list) -> pd.DataFrame:
    """Wandelt eine Liste von Bar-Dicts (wie von BarAggregator geliefert) in eine OHLCV-
    DataFrame mit DatetimeIndex um -- das Format, das build_ear_bricks()/update_symbol_bricks()
    erwarten."""
    if not bars:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    idx = pd.DatetimeIndex([b["ts"] for b in bars])
    return pd.DataFrame({"open": [b["open"] for b in bars], "high": [b["high"] for b in bars],
                          "low": [b["low"] for b in bars], "close": [b["close"] for b in bars]},
                         index=idx)
