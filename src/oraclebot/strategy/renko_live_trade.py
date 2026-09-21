# src/oraclebot/strategy/renko_live_trade.py
# Order-Ausfuehrung fuer die Portfolio-weite Renko-Breakout-Strategie. Anders als
# strategy/live_trade.py (Barriere-Strategie, festes SL+TP bei Entry) hat diese Strategie KEIN
# festes TP -- der Exit ist der erste vollstaendig ausgebildete Gegen-Brick (siehe
# horizontal_breakout_signal.py), erkannt und ausgefuehrt vom aufrufenden Cron-Lauf
# (scripts/run_renko_breakout.py), nicht von einer boersenseitigen Trigger-Order.
#
# Sicherheitsnetz: OHNE jede SL-Trigger-Order waere eine offene Position bei einem Ausfall
# dieses Prozesses (Cron haengt, Server down, Bug) UNBEGRENZT lange ungeschuetzt offen -- anders
# als bei der Barriere-Strategie, wo der Broker selbst SL/TP durchsetzt. Deshalb wird bei jedem
# Entry zusaetzlich ein Sicherheits-Stop platziert (Default 3.0%, deutlich ausserhalb der im
# Backtest beobachteten Verlust-Groessenordnung von ~Brick-Groesse/0.4-0.45%) -- er soll im
# Normalbetrieb NIE ausloesen, sondern nur einen Totalausfall abfangen.
#
# WICHTIG (Fund 2026-09-21, zweifach korrigiert): bei hohem Hebel liquidiert Bitget selbst schon
# VOR einem zu weit entfernten Sicherheits-Stop (echte Liquidationsdistanz ~ 1/Hebel -
# Wartungsmargin-Satz - Gebuehr). ERSTER Fund: ein anfangs verwendeter 5%-Stop haette bei 40x nie
# ausgeloest (Distanz nur ~2%). ZWEITER Fund: selbst bei angepasstem Stop war 40x Hebel an sich zu
# hoch angesetzt -- die BTC-Wartungsmarge aus margin_safety.py (0.40%) galt NICHT fuer diese
# Altcoins (Bitgets echte unterste Stufe: NEAR/DOT/ADA/AVAX/SUI 0.66%, SOL 0.50%, XRP 0.40%, per
# publicMixGetV2MixMarketQueryPositionLever), damit lag die reale Liquidationsdistanz bei 40x nur
# ~1.78-2.04% entfernt -- nur ~30% Puffer zum schlechtesten je in 2544 historischen Trades
# beobachteten Max-Adverse-Excursion-Wert (1.37%). Hebel deshalb auf 20x reduziert (Distanz
# 4.28-4.54%, ~3x Puffer zum historischen Worst Case). safety_stop_pct MUSS klar UNTER der bei
# der gewaehlten `leverage` real zu erwartenden (symbolspezifischen!) Liquidationsdistanz liegen,
# sonst ist er reine Dekoration. Bei isoliertem Margin ist der Schaden so oder so gedeckelt (max.
# die fuer den Trade eingesetzte Margin), nur eben ueber die Liquidation statt ueber den Trigger.
import logging
import os

from oraclebot.strategy import anti_martingale
from oraclebot.utils.telegram import send_message

logger = logging.getLogger(__name__)

MIN_NOTIONAL_USDT = 5.0

DEFAULT_AM_STATE_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'artifacts', 'state', 'renko_anti_martingale_state.json')


def open_renko_position(exchange, symbol: str, direction: str, entry_price_hint: float,
                         cfg: dict, telegram_cfg: dict, am_state_path: str = None) -> dict:
    """Platziert einen Market-Entry + weiten Sicherheits-Stop fuer die Renko-Breakout-Strategie.

    Args:
        direction: 'long' oder 'short' (aus renko_portfolio_state.detect_fresh_entry()).
        entry_price_hint: Brick-Schluss des Ausloeser-Bricks (nur fuer Logging/Groessenschaetzung
            -- die tatsaechliche Positionsgroesse wird am REALEN Fuellpreis verankert).
        cfg: renko_breakout_settings-Block (leverage, margin_mode, safety_stop_pct,
            anti_martingale_base_pct/growth_factor/streak_target).

    Returns:
        dict mit 'action' ('entered' | 'failed' | 'skipped') + Details.
    """
    am_state_path = am_state_path or DEFAULT_AM_STATE_PATH
    leverage = cfg.get('leverage', 10)
    margin_mode = cfg.get('margin_mode', 'isolated')
    safety_stop_pct = cfg.get('safety_stop_pct', 5.0)
    base_pct = cfg.get('anti_martingale_base_pct', 1.0)
    growth_factor = cfg.get('anti_martingale_growth_factor', 1.5)
    streak_target = cfg.get('anti_martingale_streak_target', 3)

    balance = exchange.fetch_balance_usdt()
    if balance < MIN_NOTIONAL_USDT:
        logger.warning(f"Renko: Guthaben zu niedrig ({balance:.2f} USDT). Kein Entry ({symbol}).")
        return {'action': 'skipped', 'reason': 'insufficient_balance'}

    am_state = anti_martingale.load_state(am_state_path, base_pct)

    exchange.set_margin_mode(symbol, margin_mode)
    exchange.set_leverage(symbol, leverage, margin_mode)

    entry_side = 'buy' if direction == 'long' else 'sell'
    margin = anti_martingale.compute_margin(balance, am_state)
    contracts = (margin * leverage) / entry_price_hint

    max_contracts_by_margin = (balance * leverage) / entry_price_hint * 0.99
    if contracts > max_contracts_by_margin:
        contracts = max_contracts_by_margin

    min_amount = exchange.fetch_min_amount_tradable(symbol)
    if contracts < min_amount:
        logger.warning(f"Renko: Menge {contracts:.6f} unter Boersen-Minimum {min_amount:.6f} ({symbol}).")
        return {'action': 'skipped', 'reason': 'below_min_amount'}

    notional = contracts * entry_price_hint
    if notional < MIN_NOTIONAL_USDT:
        logger.warning(f"Renko: Notional {notional:.2f} USDT unter Minimum ({symbol}).")
        return {'action': 'skipped', 'reason': 'below_min_notional'}

    logger.info(f"Renko: Platziere Entry {direction.upper()} {contracts:.6f} {symbol} "
                f"| Hebel {leverage}x | Einsatz {am_state['stake_pct']:.2f}% vom Guthaben")
    try:
        entry_order = exchange.place_market_order(symbol, entry_side, contracts, margin_mode=margin_mode)
    except Exception as e:
        logger.error(f"Renko: Entry fehlgeschlagen ({symbol}): {e}")
        return {'action': 'failed', 'reason': 'entry_order_failed'}

    entry_price = float(entry_order.get('average') or entry_order.get('price') or entry_price_hint)
    if entry_price <= 0:
        entry_price = entry_price_hint
    filled = float(entry_order.get('filled') or entry_order.get('amount') or contracts)
    if filled <= 0:
        filled = contracts

    exit_side = 'sell' if direction == 'long' else 'buy'
    safety_stop_price = (entry_price * (1 - safety_stop_pct / 100.0) if direction == 'long'
                         else entry_price * (1 + safety_stop_pct / 100.0))
    try:
        exchange.place_trigger_market_order(symbol, exit_side, filled, safety_stop_price, reduce=True)
        logger.info(f"Renko: Sicherheits-Stop platziert @ {safety_stop_price:.6f} ({safety_stop_pct}%)")
    except Exception as e:
        logger.error(f"Renko: Sicherheits-Stop konnte nicht platziert werden ({symbol}): {e}. "
                     f"Schliesse Position sofort statt sie ungeschuetzt zu lassen!")
        try:
            exchange.close_position(symbol)
        except Exception as ce:
            logger.critical(f"Renko: KONNTE POSITION NICHT SCHLIESSEN nach fehlgeschlagenem "
                             f"Sicherheits-Stop ({symbol}): {ce}. MANUELL PRUEFEN!")
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'),
                     f"ACHTUNG oraclebot Renko: Sicherheits-Stop fuer {symbol} fehlgeschlagen. "
                     f"Position wurde sicherheitshalber geschlossen (oder Schliessen schlug "
                     f"ebenfalls fehl -- bitte manuell pruefen).")
        return {'action': 'failed', 'reason': 'safety_stop_failed'}

    # Die Renko-Strategie hat kein festes TP/SL-Preisziel (anders als barrier_signal.py), daher
    # koennen hier keine praezisen erwarteten Endbetraege wie bei der Barriere-Strategie berechnet
    # werden. resolve_pending_outcome() vergleicht aber nur "naeher an Win- oder Loss-Erwartung"
    # -- bei SYMMETRISCHEN Ankern (+X/-X um das Guthaben vor dem Trade) reduziert sich das
    # rechnerisch exakt auf einen Vorzeichen-Test des tatsaechlichen Guthaben-Deltas (dist_to_win
    # - dist_to_loss = -2*delta fuer kleines delta), die konkrete Groesse von X ist daher fuer die
    # Win/Loss-Klassifikation irrelevant -- ein beliebiger positiver symmetrischer Platzhalter reicht.
    expected_win_balance = balance + margin * leverage * 0.10
    expected_loss_balance = balance - margin * leverage * 0.10
    am_state = anti_martingale.record_pending_position(am_state, balance, expected_win_balance, expected_loss_balance)
    anti_martingale.save_state(am_state_path, am_state)

    message = (
        f"oraclebot Renko-Breakout ENTRY: {symbol}\n"
        f"Richtung: {direction.upper()}\n"
        f"Entry: {entry_price:.6f}\n"
        f"Sicherheits-Stop: {safety_stop_price:.6f} ({safety_stop_pct}%, Backstop -- regulaerer "
        f"Exit erfolgt beim ersten Gegen-Brick)\n"
        f"Menge: {filled:.6f} | Hebel: {leverage}x\n"
        f"Guthaben: {balance:.2f} USDT | Einsatz: {am_state['stake_pct']:.2f}% (Anti-Martingale)"
    )
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    return {'action': 'entered', 'direction': direction, 'entry_price': entry_price, 'contracts': filled,
            'safety_stop_price': safety_stop_price}


def close_renko_position(exchange, symbol: str, exit_reason: str, telegram_cfg: dict,
                          am_state_path: str = None, base_pct: float = 1.0,
                          growth_factor: float = 1.5, streak_target: int = 3) -> dict:
    """Schliesst eine offene Renko-Position per Market Order (regulaerer Exit beim ersten
    Gegen-Brick, oder Reconciliation falls der Sicherheits-Stop bereits ausgeloest hat) und
    raeumt den verwaisten Sicherheits-Stop-Trigger auf (gleiches Muster wie live_trade.py:
    zwei unabhaengige Order-Typen, keine OCO-Verknuepfung)."""
    am_state_path = am_state_path or DEFAULT_AM_STATE_PATH
    open_positions = exchange.fetch_open_positions(symbol)
    if not open_positions:
        logger.info(f"Renko: Keine offene Position mehr fuer {symbol} (bereits geschlossen, z.B. "
                    f"durch den Sicherheits-Stop) -- nur Aufraeumen.")
        exchange.cancel_all_orders_for_symbol(symbol)
        resolve_am_outcome(exchange, am_state_path, base_pct, growth_factor, streak_target)
        return {'action': 'already_closed'}

    try:
        exchange.close_position(symbol)
    except Exception as e:
        logger.error(f"Renko: Schliessen fehlgeschlagen ({symbol}): {e}")
        return {'action': 'failed', 'reason': 'close_failed'}

    exchange.cancel_all_orders_for_symbol(symbol)
    am_state = resolve_am_outcome(exchange, am_state_path, base_pct, growth_factor, streak_target)

    message = (f"oraclebot Renko-Breakout EXIT: {symbol}\nGrund: {exit_reason}\n"
               f"Naechster Einsatz: {am_state['stake_pct']:.2f}% (Anti-Martingale)")
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)
    return {'action': 'closed'}


def resolve_am_outcome(exchange, am_state_path, base_pct, growth_factor, streak_target) -> dict:
    am_state = anti_martingale.load_state(am_state_path, base_pct)
    if am_state.get('pending_position'):
        current_balance = exchange.fetch_balance_usdt()
        am_state = anti_martingale.resolve_pending_outcome(
            am_state, current_balance, base_pct, growth_factor, streak_target)
        anti_martingale.save_state(am_state_path, am_state)
    return am_state
