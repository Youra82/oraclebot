# src/oraclebot/strategy/live_trade.py
# Fuehrt ein von signal.py berechnetes Handelssignal tatsaechlich auf Bitget aus.
# Getrennt von signal.py (reine Berechnung, keine Seiteneffekte) und von predict_next_candle.py
# (Orchestrierung), damit die Order-Logik isoliert gemockt/getestet werden kann.
#
# Sicherheitsprinzip (wie bei mbot/dbot/ltbbot): niemals eine ungeschuetzte Position stehen
# lassen. Reihenfolge ist bewusst Entry -> SL -> TP (nicht umgekehrt), weil reduceOnly-Trigger
# fuer SL/TP erst gegen eine tatsaechlich bestehende Position sauber definiert sind. Schlaegt
# die SL-Platzierung nach einem gefuellten Entry fehl, wird die Position sofort per Market
# Order wieder geschlossen, statt ungeschuetzt offen zu bleiben.
import logging
import os

from oraclebot.strategy import anti_martingale
from oraclebot.strategy.signal import compute_position_size
from oraclebot.utils.telegram import send_message

logger = logging.getLogger(__name__)

MIN_NOTIONAL_USDT = 5.0

DEFAULT_ANTI_MARTINGALE_STATE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'artifacts', 'state', 'anti_martingale_state.json')


def execute_live_trade(exchange, signal: dict, symbol: str, strat_cfg: dict, telegram_cfg: dict,
                        state_path: str = None) -> dict:
    """Prueft offene Positionen, platziert bei freiem Signal einen Market-Entry + SL/TP-Trigger.

    Args:
        exchange: oraclebot.utils.exchange.Exchange (authentifiziert).
        signal: Rueckgabe von compute_trade_signal() (braucht 'direction', 'entry',
            'stop_loss', 'sl_distance', 'tp_distance' -- oder 'direction': None fuer kein Signal).
        symbol: Handelspaar, z.B. 'BTC/USDT:USDT'.
        strat_cfg: settings.json strategy_settings (leverage, margin_mode, risk_per_trade_pct,
            optional anti_martingale_enabled/anti_martingale_base_pct/
            anti_martingale_growth_factor/anti_martingale_streak_target).
        telegram_cfg: secret.json telegram-Block (fuer Fehler-/Erfolgs-Benachrichtigung).
        state_path: Pfad zur Anti-Martingale-Zustandsdatei (Standard: artifacts/state/
            anti_martingale_state.json). Nur zum Testen ueberschreiben.

    Returns:
        dict mit 'action' ('skipped' | 'entered' | 'failed') + Details.
    """
    leverage = strat_cfg.get('leverage', 5)
    margin_mode = strat_cfg.get('margin_mode', 'isolated')
    risk_per_trade_pct = strat_cfg.get('risk_per_trade_pct', 1.0)
    anti_martingale_enabled = strat_cfg.get('anti_martingale_enabled', False)
    state_path = state_path or DEFAULT_ANTI_MARTINGALE_STATE_PATH
    base_pct = strat_cfg.get('anti_martingale_base_pct', 5.0)
    growth_factor = strat_cfg.get('anti_martingale_growth_factor', 2.0)
    streak_target = strat_cfg.get('anti_martingale_streak_target', 3)

    am_state = anti_martingale.load_state(state_path, base_pct) if anti_martingale_enabled else None

    open_positions = exchange.fetch_open_positions(symbol)

    # SL und TP sind zwei UNABHAENGIGE Trigger-Orders (kein OCO-Verbund) -- sobald die Position
    # geschlossen ist (gleich ob durch SL oder TP), bleibt der jeweils NICHT ausgeloeste Trigger
    # als Order-Leiche auf der Boerse stehen und koennte sonst faelschlich gegen eine spaeter
    # eroeffnete neue Position ausloesen (im Live-Test 2026-07-24 beobachtet). Deshalb hier
    # IMMER aufraeumen, sobald keine offene Position mehr da ist -- unabhaengig von Anti-Martingale.
    if not open_positions:
        exchange.cancel_all_orders_for_symbol(symbol)

    # Kein Order-Historie-API in Exchange -- der Ausgang der vorherigen Position (SL oder TP
    # gegriffen) wird stattdessen indirekt aus dem aktuellen Guthaben erschlossen, sobald keine
    # offene Position mehr gefunden wird (siehe anti_martingale.resolve_pending_outcome).
    if anti_martingale_enabled and not open_positions and am_state.get('pending_position'):
        current_balance = exchange.fetch_balance_usdt()
        am_state = anti_martingale.resolve_pending_outcome(
            am_state, current_balance, base_pct, growth_factor, streak_target)
        anti_martingale.save_state(state_path, am_state)

    if open_positions:
        pos = open_positions[0]
        logger.info(f"Position bereits offen fuer {symbol}: {pos.get('side')} @ {pos.get('entryPrice')} "
                    f"(PnL {pos.get('unrealizedPnl', 0):.2f} USDT). Kein neuer Entry.")
        return {'action': 'skipped', 'reason': 'position_open'}

    if signal.get('direction') is None:
        logger.info(f"Kein Handelssignal ({signal.get('reason')}). Kein Live-Trade.")
        return {'action': 'skipped', 'reason': signal.get('reason', 'no_signal')}

    balance = exchange.fetch_balance_usdt()
    if balance < MIN_NOTIONAL_USDT:
        logger.warning(f"Guthaben zu niedrig ({balance:.2f} USDT). Kein Live-Trade.")
        return {'action': 'skipped', 'reason': 'insufficient_balance'}

    exchange.set_margin_mode(symbol, margin_mode)
    exchange.set_leverage(symbol, leverage, margin_mode)

    side = signal['direction']
    entry_side = 'buy' if side == 'long' else 'sell'
    if anti_martingale_enabled:
        margin = anti_martingale.compute_margin(balance, am_state)
        contracts = (margin * leverage) / signal['entry']
        logger.info(f"Anti-Martingale aktiv: Einsatz={am_state['stake_pct']:.2f}% vom Guthaben "
                    f"(Serie={am_state.get('consecutive_wins', 0)}/{streak_target}).")
    else:
        contracts = compute_position_size(balance, risk_per_trade_pct, signal['entry'], signal['stop_loss'])

    # Margin-Cap: risiko-basierte Groesse darf die verfuegbare Margin nicht ueberschreiten
    # (kann bei sehr kleiner SL-Distanz passieren) -- 1% Puffer wie bei mbot.
    max_contracts_by_margin = (balance * leverage) / signal['entry'] * 0.99
    if contracts > max_contracts_by_margin:
        logger.warning(f"Kontrakte {contracts:.6f} > Margin-Cap {max_contracts_by_margin:.6f}. Reduziere.")
        contracts = max_contracts_by_margin

    min_amount = exchange.fetch_min_amount_tradable(symbol)
    if contracts < min_amount:
        logger.warning(f"Menge {contracts:.6f} unter Boersen-Minimum {min_amount:.6f}. Kein Live-Trade.")
        return {'action': 'skipped', 'reason': 'below_min_amount'}

    notional = contracts * signal['entry']
    if notional < MIN_NOTIONAL_USDT:
        logger.warning(f"Notional {notional:.2f} USDT unter Minimum {MIN_NOTIONAL_USDT} USDT. Kein Live-Trade.")
        return {'action': 'skipped', 'reason': 'below_min_notional'}

    logger.info(f"Platziere Live-Entry: {side.upper()} {contracts:.6f} {symbol} "
                f"| Hebel {leverage}x | Guthaben {balance:.2f} USDT | Risiko {risk_per_trade_pct}%")

    try:
        entry_order = exchange.place_market_order(symbol, entry_side, contracts, margin_mode=margin_mode)
    except Exception as e:
        logger.error(f"Entry fehlgeschlagen: {e}")
        return {'action': 'failed', 'reason': 'entry_order_failed'}

    entry_price = float(entry_order.get('average') or entry_order.get('price') or signal['entry'])
    if entry_price <= 0:
        entry_price = signal['entry']
    filled = float(entry_order.get('filled') or entry_order.get('amount') or contracts)
    if filled <= 0:
        filled = contracts

    # SL/TP mit denselben Preis-Abstaenden wie im Signal berechnet, aber neu am TATSAECHLICHEN
    # Fuellpreis verankert (der bei einer Market Order leicht vom Signal-Entry abweichen kann).
    sl_distance = signal['sl_distance']
    tp_distance = signal['tp_distance']
    if side == 'long':
        sl_price = entry_price - sl_distance
        tp_price = entry_price + tp_distance
    else:
        sl_price = entry_price + sl_distance
        tp_price = entry_price - tp_distance
    exit_side = 'sell' if side == 'long' else 'buy'

    try:
        exchange.place_trigger_market_order(symbol, exit_side, filled, sl_price, reduce=True)
        logger.info(f"SL platziert @ {sl_price:.2f}")
    except Exception as e:
        logger.error(f"SL konnte nicht platziert werden: {e}. Schliesse Position sofort!")
        try:
            exchange.close_position(symbol)
        except Exception as ce:
            logger.critical(f"KONNTE POSITION NICHT SCHLIESSEN nach fehlgeschlagenem SL: {ce}. MANUELL PRUEFEN!")
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                     f"ACHTUNG oraclebot: SL-Platzierung fuer {symbol} fehlgeschlagen. Position wurde "
                     f"sicherheitshalber geschlossen (oder Schliessen schlug ebenfalls fehl -- bitte manuell pruefen).")
        return {'action': 'failed', 'reason': 'sl_placement_failed'}

    try:
        exchange.place_trigger_market_order(symbol, exit_side, filled, tp_price, reduce=True)
        logger.info(f"TP platziert @ {tp_price:.2f}")
    except Exception as e:
        logger.error(f"TP konnte nicht platziert werden: {e}. Position bleibt durch SL geschuetzt.")

    if anti_martingale_enabled:
        expected_win_balance = balance + tp_distance * filled
        expected_loss_balance = balance - sl_distance * filled
        am_state = anti_martingale.record_pending_position(am_state, balance, expected_win_balance, expected_loss_balance)
        anti_martingale.save_state(state_path, am_state)

    sl_dist_pct = abs(entry_price - sl_price) / entry_price * 100
    tp_dist_pct = abs(tp_price - entry_price) / entry_price * 100
    size_info = f"Einsatz: {am_state['stake_pct']:.2f}% (Anti-Martingale)" if anti_martingale_enabled \
        else f"Risiko: {risk_per_trade_pct}%"
    message = (
        f"oraclebot LIVE-TRADE: {symbol}\n"
        f"Richtung: {side.upper()}\n"
        f"Entry: {entry_price:.2f}\n"
        f"SL: {sl_price:.2f} (-{sl_dist_pct:.2f}%)\n"
        f"TP: {tp_price:.2f} (+{tp_dist_pct:.2f}%)\n"
        f"Menge: {filled:.6f} | Hebel: {leverage}x\n"
        f"Guthaben: {balance:.2f} USDT | {size_info}"
    )
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    return {'action': 'entered', 'side': side, 'entry_price': entry_price, 'sl_price': sl_price,
            'tp_price': tp_price, 'contracts': filled}
