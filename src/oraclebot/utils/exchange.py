# src/oraclebot/utils/exchange.py
# Authentifizierter Bitget-Wrapper fuer Live-Order-Platzierung. Adaptiert vom selben Muster
# wie mbot/dbot/ltbbot (Exchange-Klasse um ccxt), damit die Order-Logik so nah wie moeglich an
# bereits produktiv gehaertetem Code bleibt statt eigene Bitget-Eigenheiten neu zu entdecken.
import logging

import ccxt

logger = logging.getLogger(__name__)


class Exchange:
    def __init__(self, account_config: dict):
        self.account = account_config
        self.exchange = ccxt.bitget({
            'apiKey': self.account.get('apiKey'),
            'secret': self.account.get('secret'),
            'password': self.account.get('password'),
            'options': {'defaultType': 'swap'},
            'enableRateLimit': True,
        })
        try:
            self.markets = self.exchange.load_markets()
            logger.info("Maerkte erfolgreich geladen.")
        except Exception as e:
            logger.critical(f"Maerkte konnten nicht geladen werden: {e}")
            self.markets = {}

    # --- Balance ---

    def fetch_balance_usdt(self) -> float:
        if not self.markets:
            return 0.0
        try:
            params = {'marginCoin': 'USDT', 'productType': 'USDT-FUTURES'}
            balance = self.exchange.fetch_balance(params=params)
            usdt = 0.0
            if 'USDT' in balance and balance['USDT'].get('free') is not None:
                usdt = float(balance['USDT']['free'])
            elif 'info' in balance and isinstance(balance['info'], list):
                for item in balance['info']:
                    if item.get('marginCoin') == 'USDT':
                        usdt = float(item.get('available', 0.0))
                        break
            if usdt == 0.0 and 'total' in balance and 'USDT' in balance['total']:
                usdt = float(balance['total']['USDT'])
            logger.info(f"Verfuegbares Guthaben: {usdt:.2f} USDT")
            return usdt
        except ccxt.AuthenticationError as e:
            logger.critical(f"Authentifizierungsfehler: {e}")
            return 0.0
        except Exception as e:
            logger.error(f"Fehler beim Abrufen des Guthabens: {e}", exc_info=True)
            return 0.0

    # --- Precision helpers ---

    def amount_to_precision(self, symbol: str, amount: float) -> str:
        try:
            return self.exchange.amount_to_precision(symbol, amount)
        except Exception:
            return str(amount)

    def price_to_precision(self, symbol: str, price: float) -> str:
        try:
            return self.exchange.price_to_precision(symbol, price)
        except Exception:
            return str(price)

    def fetch_min_amount_tradable(self, symbol: str) -> float:
        try:
            if symbol not in self.markets:
                self.markets = self.exchange.load_markets()
            min_amount = self.markets[symbol].get('limits', {}).get('amount', {}).get('min')
            return float(min_amount) if min_amount is not None else 0.0
        except Exception:
            return 0.0

    # --- Positions ---

    def fetch_open_positions(self, symbol: str) -> list:
        if not self.markets:
            return []
        try:
            params = {'productType': 'USDT-FUTURES', 'marginCoin': 'USDT'}
            positions = self.exchange.fetch_positions([symbol], params=params)
            open_pos = []
            for p in positions:
                try:
                    contracts = p.get('contracts') or p.get('contractSize')
                    if contracts is not None and abs(float(contracts)) > 1e-9:
                        open_pos.append(p)
                except (ValueError, TypeError):
                    continue
            return open_pos
        except Exception as e:
            logger.error(f"Fehler beim Abrufen offener Positionen fuer {symbol}: {e}", exc_info=True)
            return []

    # --- Margin / Leverage ---

    def set_margin_mode(self, symbol: str, margin_mode: str = 'isolated'):
        try:
            params = {'productType': 'USDT-FUTURES', 'marginCoin': 'USDT'}
            self.exchange.set_margin_mode(margin_mode.lower(), symbol, params=params)
            logger.info(f"Margin-Modus fuer {symbol}: {margin_mode}")
        except ccxt.ExchangeError as e:
            if any(x in str(e) for x in ['Margin mode is the same', '40051']):
                logger.debug(f"Margin-Modus bereits {margin_mode}.")
            else:
                logger.error(f"Fehler beim Setzen des Margin-Modus: {e}")
        except Exception as e:
            logger.error(f"Fehler bei Margin-Modus: {e}")

    def set_leverage(self, symbol: str, leverage: int, margin_mode: str = 'isolated'):
        try:
            params = {'productType': 'USDT-FUTURES', 'marginCoin': 'USDT', 'marginMode': margin_mode.lower()}
            self.exchange.set_leverage(leverage, symbol, params=params)
            logger.info(f"Hebel fuer {symbol} auf {leverage}x gesetzt.")
        except ccxt.ExchangeError as e:
            if any(x in str(e) for x in ['Leverage not changed', '40052']):
                logger.debug(f"Hebel bereits {leverage}x.")
            else:
                logger.error(f"Fehler beim Setzen des Hebels: {e}")
        except Exception as e:
            logger.error(f"Fehler bei Hebel: {e}")

    # --- Orders ---

    def place_market_order(self, symbol: str, side: str, amount: float, reduce: bool = False,
                            margin_mode: str = 'isolated'):
        try:
            params = {
                'productType': 'USDT-FUTURES',
                'marginCoin': 'USDT',
                'marginMode': margin_mode,
                'hedged': True,
                'reduceOnly': reduce,
            }
            amount_str = self.amount_to_precision(symbol, amount)
            logger.info(f"Market Order: {side.upper()} {amount_str} {symbol} hedged=True reduce={reduce}")
            return self.exchange.create_order(symbol, 'market', side, float(amount_str), params=params)
        except ccxt.InsufficientFunds as e:
            logger.error(f"Nicht genuegend Guthaben: {e}")
            raise
        except Exception as e:
            logger.error(f"Fehler bei Market Order: {e}", exc_info=True)
            raise

    def place_trigger_market_order(self, symbol: str, side: str, amount: float,
                                    trigger_price: float, reduce: bool = False):
        """TP/SL Trigger Order (reduceOnly)."""
        try:
            amount_str = self.amount_to_precision(symbol, amount)
            trigger_str = self.price_to_precision(symbol, trigger_price)
            params = {
                'triggerPrice': trigger_str,
                'reduceOnly': reduce,
                'productType': 'USDT-FUTURES',
                'marginMode': 'isolated',
                'hedged': True,
            }
            logger.info(f"Trigger Market Order: {side.upper()} {amount_str} {symbol} @ {trigger_str} hedged=True")
            return self.exchange.create_order(symbol, 'market', side, float(amount_str), params=params)
        except Exception as e:
            logger.error(f"Fehler bei Trigger Order: {e}", exc_info=True)
            raise

    def close_position(self, symbol: str):
        """Schliesst eine offene Position sofort via Market Order (Sicherheits-Fallback,
        z.B. wenn nach einem gefuellten Entry der SL nicht platziert werden konnte --
        niemals eine ungeschuetzte Position stehen lassen)."""
        try:
            positions = self.fetch_open_positions(symbol)
            if not positions:
                logger.warning(f"Keine offene Position zum Schliessen fuer {symbol}.")
                return None
            pos = positions[0]
            pos_side = pos['side']
            close_side = 'sell' if pos_side == 'long' else 'buy'
            amount = float(pos.get('contracts') or pos.get('contractSize'))
            logger.info(f"Schliesse {pos_side} Position fuer {symbol} ({amount} Kontrakte).")
            return self.place_market_order(symbol, close_side, amount, reduce=True)
        except Exception as e:
            logger.error(f"Fehler beim Schliessen der Position: {e}")
            raise
