# src/oraclebot/utils/bitget_ws.py
# Roher WebSocket-Client fuer Bitgets oeffentlichen Trade-Kanal (v2 API) -- kein ccxt.pro
# verfuegbar/installiert, daher direkte Implementierung ueber die `websockets`-Bibliothek.
# Liefert einen kontinuierlichen Strom von (symbol, ts_ms, price)-Tripeln aus echten,
# ausgefuehrten Trades -- die Rohdatenquelle fuer die Echtzeit-Bar-Aggregation
# (utils/realtime_bars.py), die wiederum die bestehende Brick-Logik fuettert.
#
# Bitget-Protokoll (2026-09-23 aus der offiziellen API-Doku + SDK-Referenzen ermittelt):
#   Endpunkt: wss://ws.bitget.com/v2/ws/public
#   Subscribe: {"op": "subscribe", "args": [{"instType": "USDT-FUTURES", "channel": "trade",
#               "instId": "<SYMBOL>USDT"}, ...]}
#   Push-Daten: {"action": "snapshot"|"update", "arg": {...}, "data": [{"ts": "<ms>",
#                "price": "<str>", "size": "<str>", "side": "buy"|"sell", "tradeId": "..."}]}
#   Heartbeat: Client sendet alle ~25s den WOERTLICHEN String "ping" (kein JSON!), Server
#              antwortet mit dem woertlichen String "pong". Server trennt die Verbindung, wenn
#              2 Minuten lang kein "ping" vom Client kommt.
import asyncio
import json
import logging
import time

import websockets

logger = logging.getLogger(__name__)

WS_URL = "wss://ws.bitget.com/v2/ws/public"
PING_INTERVAL_SECONDS = 25
RECONNECT_BACKOFF_SECONDS = [1, 2, 5, 10, 30, 60]  # letzter Wert wiederholt sich


def to_bitget_inst_id(symbol: str) -> str:
    """'NEAR/USDT:USDT' -> 'NEARUSDT' (Bitgets instId-Format, kein Slash/Doppelpunkt)."""
    base = symbol.split("/")[0]
    return f"{base}USDT"


def from_bitget_inst_id(inst_id: str) -> str:
    """'NEARUSDT' -> 'NEAR/USDT:USDT' -- Umkehrung von to_bitget_inst_id() fuer die 7
    unterstuetzten Symbole (alle USDT-quotiert, USDT-Perpetual)."""
    base = inst_id[:-4]  # 'USDT' abschneiden
    return f"{base}/USDT:USDT"


def parse_trade_message(raw: str) -> list:
    """Parst eine einzelne WebSocket-Textnachricht. Gibt eine Liste von
    (symbol, ts_ms, price)-Tripeln zurueck -- leer, wenn die Nachricht kein Trade-Push ist
    (z.B. 'pong', ein Subscribe-Bestaetigung-Event, oder eine Fehlermeldung).

    Bewusst tolerant: unerwartete/fehlende Felder fuehren zu einer leeren Rueckgabe (geloggt),
    nie zu einer Exception -- ein einzelner kaputter Tick darf den Live-Stream nie abreissen
    lassen."""
    if raw == "pong":
        return []
    try:
        msg = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Bitget-WS: unparsebare Nachricht ignoriert: {raw[:200]!r}")
        return []

    if msg.get("event") in ("subscribe", "error"):
        if msg.get("event") == "error":
            logger.error(f"Bitget-WS: Fehler-Event: {msg}")
        return []

    arg = msg.get("arg", {})
    if arg.get("channel") != "trade":
        return []

    inst_id = arg.get("instId")
    data = msg.get("data", [])
    if not inst_id or not isinstance(data, list):
        return []

    try:
        symbol = from_bitget_inst_id(inst_id)
    except Exception:
        logger.warning(f"Bitget-WS: unbekannte instId ignoriert: {inst_id!r}")
        return []

    out = []
    for entry in data:
        try:
            ts_ms = int(entry["ts"])
            price = float(entry["price"])
        except (KeyError, TypeError, ValueError):
            logger.warning(f"Bitget-WS: unvollstaendiger Trade-Eintrag ignoriert: {entry!r}")
            continue
        out.append((symbol, ts_ms, price))
    return out


class BitgetTradeStream:
    """Haelt eine WebSocket-Verbindung zu Bitgets oeffentlichem Trade-Kanal offen (mit
    automatischem Reconnect bei Verbindungsabbruch) und liefert ueber `stream()` einen
    kontinuierlichen Async-Generator von (symbol, ts_ms, price)-Tripeln fuer alle
    abonnierten Symbole."""

    def __init__(self, symbols: list, product_type: str = "USDT-FUTURES"):
        self.symbols = symbols
        self.product_type = product_type
        self._stop = False

    def stop(self):
        self._stop = True

    async def _subscribe(self, ws):
        args = [{"instType": self.product_type, "channel": "trade", "instId": to_bitget_inst_id(s)}
                for s in self.symbols]
        await ws.send(json.dumps({"op": "subscribe", "args": args}))

    async def _ping_loop(self, ws):
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            try:
                await ws.send("ping")
            except Exception:
                return  # Verbindung schon weg -- die Haupt-Empfangsschleife kuemmert sich ums Reconnect

    async def stream(self):
        """Async-Generator: liefert (symbol, ts_ms, price) fuer jeden echten Trade-Tick,
        reconnected automatisch (mit steigendem Backoff) bei Verbindungsabbruch. Laeuft
        unbegrenzt, bis stop() aufgerufen wird."""
        backoff_idx = 0
        while not self._stop:
            try:
                async with websockets.connect(WS_URL, ping_interval=None) as ws:
                    logger.info(f"Bitget-WS verbunden, abonniere {len(self.symbols)} Symbole...")
                    await self._subscribe(ws)
                    backoff_idx = 0  # erfolgreiche Verbindung setzt den Backoff zurueck
                    ping_task = asyncio.create_task(self._ping_loop(ws))
                    try:
                        async for raw in ws:
                            for symbol, ts_ms, price in parse_trade_message(raw):
                                yield symbol, ts_ms, price
                    finally:
                        ping_task.cancel()
            except (websockets.exceptions.ConnectionClosed, OSError, asyncio.TimeoutError) as e:
                if self._stop:
                    return
                delay = RECONNECT_BACKOFF_SECONDS[min(backoff_idx, len(RECONNECT_BACKOFF_SECONDS) - 1)]
                backoff_idx += 1
                logger.warning(f"Bitget-WS Verbindung verloren ({e}), Reconnect in {delay}s...")
                await asyncio.sleep(delay)
