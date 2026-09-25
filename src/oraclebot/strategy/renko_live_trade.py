# src/oraclebot/strategy/renko_live_trade.py
# Order-Ausfuehrung fuer die Portfolio-weite Renko-Breakout-Strategie. Anders als
# strategy/live_trade.py (Barriere-Strategie, festes SL+TP bei Entry) hat diese Strategie KEIN
# festes TP -- der Exit ist der erste vollstaendig ausgebildete Gegen-Brick (siehe
# horizontal_breakout_signal.py), erkannt und ausgefuehrt vom aufrufenden Prozess
# (scripts/run_renko_realtime.py), nicht von einer boersenseitigen Trigger-Order.
#
# Sicherheitsnetz: OHNE jede SL-Trigger-Order waere eine offene Position bei einem Ausfall
# dieses Prozesses (Prozess haengt, Server down, Bug) UNBEGRENZT lange ungeschuetzt offen --
# anders als bei der Barriere-Strategie, wo der Broker selbst SL/TP durchsetzt. Deshalb wird bei
# jedem Entry zusaetzlich ein Sicherheits-Stop platziert (Default 3.0%, deutlich ausserhalb der
# im Backtest beobachteten Verlust-Groessenordnung von ~Brick-Groesse/0.4-0.45%) -- er soll im
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
#
# MEHRFACH-POSITIONEN (Fund 2026-09-25): jedes Symbol handelt unabhaengig, mehrere Positionen
# koennen gleichzeitig offen sein (siehe scripts/run_renko_realtime.py) -- open_renko_position()
# bemisst die Positionsgroesse dabei automatisch korrekt, weil exchange.fetch_balance_usdt() nur
# das FREIE (nicht durch andere offene Positionen gebundene) Guthaben liefert. close_renko_position()
# braucht jetzt `direction`+`entry_price` vom Aufrufer (dort lokal je Symbol nachgehalten), um
# Gewinn/Verlust direkt am echten Fuellpreis zu erkennen, statt am Kontostand-Delta -- der alte
# Trick war nur bei GENAU EINER offenen Position eindeutig (siehe anti_martingale.py Moduldoc).
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
        dict mit 'action' ('entered' | 'failed' | 'skipped') + Details (bei 'entered' inkl.
        'entry_price'/'contracts' -- der Aufrufer haelt diese lokal je Symbol nach, um spaeter
        close_renko_position() korrekt aufzurufen).
    """
    am_state_path = am_state_path or DEFAULT_AM_STATE_PATH
    leverage = cfg.get('leverage', 10)
    margin_mode = cfg.get('margin_mode', 'isolated')
    safety_stop_pct = cfg.get('safety_stop_pct', 5.0)
    base_pct = cfg.get('anti_martingale_base_pct', 1.0)

    balance = exchange.fetch_balance_usdt()
    if balance < MIN_NOTIONAL_USDT:
        logger.warning(f"Renko: Freies Guthaben zu niedrig ({balance:.2f} USDT). Kein Entry ({symbol}).")
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
                f"| Hebel {leverage}x | Einsatz {am_state['stake_pct']:.2f}% vom freien Guthaben")
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

    message = (
        f"oraclebot Renko-Breakout ENTRY: {symbol}\n"
        f"Richtung: {direction.upper()}\n"
        f"Entry: {entry_price:.6f}\n"
        f"Sicherheits-Stop: {safety_stop_price:.6f} ({safety_stop_pct}%, Backstop -- regulaerer "
        f"Exit erfolgt beim ersten Gegen-Brick)\n"
        f"Menge: {filled:.6f} | Hebel: {leverage}x\n"
        f"Freies Guthaben: {balance:.2f} USDT | Einsatz: {am_state['stake_pct']:.2f}% (Anti-Martingale)"
    )
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    return {'action': 'entered', 'direction': direction, 'entry_price': entry_price, 'contracts': filled,
            'safety_stop_price': safety_stop_price}


def _resolve_outcome_from_history(exchange, symbol: str) -> bool:
    """Bestimmt Gewinn/Verlust einer bereits geschlossenen Position anhand der echten
    Positions-Historie (fetch_closed_positions -> realizedPnl) -- fuer den Fall, dass DIESER
    Prozess die Position nicht selbst geschlossen hat (z.B. Sicherheits-Stop ausgeloest, oder
    Reconcile nach einem Neustart). Konservativ: keine Historie gefunden zaehlt als Verlust
    (Einsatz faellt auf die Basis zurueck, statt faelschlich zu compounden)."""
    try:
        closed = exchange.fetch_closed_positions(symbol, limit=1)
    except Exception as e:
        logger.error(f"Renko: Positions-Historie fuer {symbol} nicht abrufbar: {e}. "
                     f"Werte konservativ als Verlust.")
        return False
    if not closed:
        logger.warning(f"Renko: keine Positions-Historie fuer {symbol} gefunden. Werte konservativ als Verlust.")
        return False
    return float(closed[0].get('realizedPnl') or 0) > 0


def _update_am_state(am_state_path: str, base_pct: float, growth_factor: float,
                      streak_target: int, is_win: bool) -> dict:
    am_state = anti_martingale.load_state(am_state_path, base_pct)
    am_state = anti_martingale.update_after_close(am_state, base_pct, growth_factor, streak_target, is_win)
    anti_martingale.save_state(am_state_path, am_state)
    return am_state


def resolve_am_outcome(exchange, symbol: str, am_state_path: str, base_pct: float,
                        growth_factor: float = 1.5, streak_target: int = 3) -> dict:
    """Fuer den Reconcile-Fall: eine Position wurde extern geschlossen (Sicherheits-Stop,
    manuell, oder ein frueherer Prozess-Absturz vor dem eigenen Schliessen-Aufruf) -- Gewinn/
    Verlust wird ausschliesslich ueber die echte Positions-Historie ermittelt (kein lokal
    bekannter Fuellpreis verfuegbar)."""
    am_state_path = am_state_path or DEFAULT_AM_STATE_PATH
    is_win = _resolve_outcome_from_history(exchange, symbol)
    return _update_am_state(am_state_path, base_pct, growth_factor, streak_target, is_win)


def close_renko_position(exchange, symbol: str, direction: str, entry_price: float, exit_reason: str,
                          telegram_cfg: dict, am_state_path: str = None, base_pct: float = 1.0,
                          growth_factor: float = 1.5, streak_target: int = 3) -> dict:
    """Schliesst eine offene Renko-Position per Market Order (regulaerer Exit beim ersten
    Gegen-Brick) und raeumt den verwaisten Sicherheits-Stop-Trigger auf (gleiches Muster wie
    live_trade.py: zwei unabhaengige Order-Typen, keine OCO-Verknuepfung).

    Args:
        direction/entry_price: vom Aufrufer lokal nachgehaltener Zustand dieser Position (siehe
            open_renko_position()-Rueckgabe) -- noetig, um Gewinn/Verlust direkt am echten
            Fuellpreis zu erkennen (siehe Moduldoc: kein Kontostand-Trick mehr bei mehreren
            gleichzeitig offenen Positionen)."""
    am_state_path = am_state_path or DEFAULT_AM_STATE_PATH
    open_positions = exchange.fetch_open_positions(symbol)
    if not open_positions:
        logger.info(f"Renko: Keine offene Position mehr fuer {symbol} (bereits geschlossen, z.B. "
                    f"durch den Sicherheits-Stop) -- nur Aufraeumen.")
        exchange.cancel_all_orders_for_symbol(symbol)
        is_win = _resolve_outcome_from_history(exchange, symbol)
        _update_am_state(am_state_path, base_pct, growth_factor, streak_target, is_win)
        return {'action': 'already_closed', 'is_win': is_win}

    try:
        close_order = exchange.close_position(symbol)
    except Exception as e:
        logger.error(f"Renko: Schliessen fehlgeschlagen ({symbol}): {e}")
        return {'action': 'failed', 'reason': 'close_failed'}

    exchange.cancel_all_orders_for_symbol(symbol)

    exit_price = 0.0
    if close_order:
        exit_price = float(close_order.get('average') or close_order.get('price') or 0)
    if exit_price > 0 and entry_price > 0:
        sign = 1 if direction == 'long' else -1
        is_win = sign * (exit_price - entry_price) > 0
    else:
        is_win = _resolve_outcome_from_history(exchange, symbol)

    am_state = _update_am_state(am_state_path, base_pct, growth_factor, streak_target, is_win)

    message = (f"oraclebot Renko-Breakout EXIT: {symbol}\nGrund: {exit_reason}\n"
               f"Ergebnis: {'GEWINN' if is_win else 'VERLUST'}\n"
               f"Naechster Einsatz: {am_state['stake_pct']:.2f}% (Anti-Martingale)")
    send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)
    return {'action': 'closed', 'is_win': is_win, 'exit_price': exit_price or None}
