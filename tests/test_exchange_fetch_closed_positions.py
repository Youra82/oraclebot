"""Regressionstest fuer den Fund 2026-09-25: ccxt's vereinheitlichtes fetchPositionsHistory()
existiert erst ab einer neueren Version als die gepinnte ccxt==4.3.5 -- fetch_closed_positions()
MUSS stattdessen den rohen Bitget-Endpunkt (privateMixGetV2MixPositionHistoryPosition) nutzen,
der auch in 4.3.5 bereits registriert ist. Live gegen den echten Account UND explizit gegen die
gepinnte ccxt-Version verifiziert (siehe project_oraclebot-Memory) -- dieser Test haelt die
Normalisierung des rohen Antwortformats fest, damit ein kuenftiger Umbau nicht denselben Fehler
unbemerkt wiederholt."""
from oraclebot.utils.exchange import Exchange


class FakeCcxtExchange:
    """Ersetzt nur die eine Methode, auf die fetch_closed_positions() sich verlaesst -- kein
    echter API-Zugriff, damit dieser Test ohne Bitget-Verbindung laeuft."""

    def __init__(self, raw_list):
        self._raw_list = raw_list
        self.calls = []

    def market(self, symbol):
        return {'id': symbol.split('/')[0] + 'USDT'}

    def privateMixGetV2MixPositionHistoryPosition(self, params):
        self.calls.append(params)
        return {'code': '00000', 'msg': 'success', 'data': {'list': self._raw_list, 'endId': '123'}}


def _make_exchange_with_fake(raw_list):
    ex = Exchange.__new__(Exchange)  # __init__ ueberspringen (baut echte ccxt-Verbindung auf)
    ex.markets = {'placeholder': True}
    ex.exchange = FakeCcxtExchange(raw_list)
    return ex


def test_fetch_closed_positions_uses_raw_endpoint_not_missing_unified_method():
    raw = [{
        'positionId': '1', 'symbol': 'ADAUSDT', 'holdSide': 'short',
        'openAvgPrice': '0.2526', 'closeAvgPrice': '0.2531',
        'pnl': '-0.0215', 'netProfit': '-0.03454706',
        'ctime': '1790330103654', 'utime': '1790332204556',
    }]
    ex = _make_exchange_with_fake(raw)

    result = ex.fetch_closed_positions('ADA/USDT:USDT', limit=5)

    assert len(result) == 1
    assert result[0]['symbol'] == 'ADA/USDT:USDT'
    assert result[0]['side'] == 'short'
    assert result[0]['entryPrice'] == 0.2526
    assert result[0]['lastPrice'] == 0.2531
    assert result[0]['realizedPnl'] == -0.03454706  # netProfit, NICHT das rohe (gebuehrenfreie) pnl
    assert result[0]['timestamp'] == 1790330103654


def test_fetch_closed_positions_normalizes_long_side():
    raw = [{'holdSide': 'long', 'openAvgPrice': '4.5', 'closeAvgPrice': '4.6',
            'netProfit': '0.5', 'ctime': '1000', 'utime': '2000'}]
    ex = _make_exchange_with_fake(raw)
    result = ex.fetch_closed_positions('NEAR/USDT:USDT')
    assert result[0]['side'] == 'long'
    assert result[0]['realizedPnl'] == 0.5


def test_fetch_closed_positions_returns_empty_list_on_error():
    ex = _make_exchange_with_fake([])

    class BrokenCcxt(FakeCcxtExchange):
        def privateMixGetV2MixPositionHistoryPosition(self, params):
            raise AttributeError("simuliert die tatsaechliche VPS-Panne (2026-09-25)")

    ex.exchange = BrokenCcxt([])
    result = ex.fetch_closed_positions('ADA/USDT:USDT')
    assert result == []


def test_fetch_closed_positions_passes_correct_params_to_raw_endpoint():
    ex = _make_exchange_with_fake([])
    ex.fetch_closed_positions('ADA/USDT:USDT', limit=7)
    call = ex.exchange.calls[0]
    assert call['symbol'] == 'ADAUSDT'
    assert call['productType'] == 'USDT-FUTURES'
    assert call['limit'] == '7'
