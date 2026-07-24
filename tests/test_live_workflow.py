# tests/test_live_workflow.py
"""
oraclebot Live-Workflow-Test (Anti-Martingale)

Testet, dass execute_live_trade() mit aktivierter Anti-Martingale-Positionsgroesse einen
echten Entry+SL/TP-Zyklus auf Bitget platziert (PEPE, kleines Mindest-Notional, wie im
mbot-Muster) und die Zustandsdatei korrekt aktualisiert.

WICHTIG: Die Zustandsuebergaenge selbst (Win/Loss-Eskalation, 3er-Serien-Reset) sind bereits
vollstaendig und deterministisch in test_anti_martingale.py abgedeckt -- OHNE Boersenverbindung
noetig. Echte Marktbewegungen lassen sich nicht kontrolliert erzwingen (man kann nicht
garantieren, dass der Markt jetzt einen Gewinn oder Verlust produziert), deshalb prueft dieser
Test NUR, dass die Groessen-Berechnung korrekt in echte Bitget-Orders uebersetzt wird -- nicht
die Martingale-Logik selbst.

Benoetigt secret.json mit gueltigen 'oraclebot'-API-Keys, sonst wird uebersprungen. Sendet bei
Erfolg eine echte Telegram-Benachrichtigung (wie in der Produktion).
"""
import json
import os
import sys
import time

import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from oraclebot.strategy.live_trade import execute_live_trade
from oraclebot.utils.exchange import Exchange

SYMBOL = 'PEPE/USDT:USDT'  # Kleine Mindestgroesse, wie im mbot-Live-Test
LEVERAGE = 20
MARGIN_MODE = 'isolated'


def _force_close_if_open(exchange, symbol):
    try:
        exchange.cancel_all_orders_for_symbol(symbol)
        positions = exchange.fetch_open_positions(symbol)
        if positions:
            pos = positions[0]
            side = 'sell' if pos['side'] == 'long' else 'buy'
            amt = abs(float(pos.get('contracts') or pos.get('contractSize', 0)))
            if amt > 0:
                exchange.place_market_order(symbol, side, amt, reduce=True)
                time.sleep(3)
        exchange.cancel_all_orders_for_symbol(symbol)
    except Exception as e:
        print(f'WARNUNG beim Bereinigen: {e}')


@pytest.fixture(scope='module')
def test_setup():
    print('\n--- oraclebot Live-Workflow-Test (Anti-Martingale) ---')

    secret_path = os.path.join(PROJECT_ROOT, 'secret.json')
    if not os.path.exists(secret_path):
        pytest.skip('secret.json nicht gefunden. Ueberspringe Live-Test.')

    with open(secret_path, 'r') as f:
        secrets = json.load(f)

    accounts = secrets.get('oraclebot', [])
    if not accounts or not accounts[0].get('apiKey'):
        pytest.skip("Keine 'oraclebot'-Accounts in secret.json. Ueberspringe Live-Test.")

    try:
        exchange = Exchange(accounts[0])
        if not exchange.markets:
            pytest.fail('Exchange konnte nicht initialisiert werden.')
    except Exception as e:
        pytest.fail(f'Exchange-Fehler: {e}')

    print(f'[Setup] Bereinige Ausgangszustand fuer {SYMBOL}...')
    _force_close_if_open(exchange, SYMBOL)

    state_path = os.path.join(PROJECT_ROOT, 'artifacts', 'state', 'test_anti_martingale_state.json')
    if os.path.exists(state_path):
        os.remove(state_path)

    telegram_config = secrets.get('telegram', {})

    yield exchange, state_path, telegram_config

    print('\n[Teardown] Raeume nach dem Test auf...')
    _force_close_if_open(exchange, SYMBOL)
    if os.path.exists(state_path):
        os.remove(state_path)


def test_anti_martingale_live_entry_on_bitget(test_setup):
    """Ein echter Zyklus: Anti-Martingale-Sizing -> Entry+SL/TP auf Bitget -> Cleanup."""
    exchange, state_path, telegram_config = test_setup

    balance = exchange.fetch_balance_usdt()
    print(f'\nVerfuegbares Guthaben: {balance:.4f} USDT')
    if balance < 5.0:
        pytest.skip(f'Zu wenig Guthaben ({balance:.2f} USDT < 5 USDT) fuer Live-Test.')

    ticker = exchange.exchange.fetch_ticker(SYMBOL)
    price = float(ticker['last'])

    # Signal manuell konstruiert (SL=TP=1%, wie die Produktions-Konfiguration) -- dieser Test
    # prueft die Sizing->Order-Uebersetzung, nicht die Modell-Vorhersage selbst.
    sl_distance = price * 0.01
    tp_distance = price * 0.01
    signal = {
        'direction': 'long', 'entry': price, 'stop_loss': price - sl_distance,
        'take_profit': price + tp_distance, 'sl_distance': sl_distance,
        'tp_distance': tp_distance, 'confidence': 0.9,
    }

    strat_cfg = {
        'leverage': LEVERAGE, 'margin_mode': MARGIN_MODE,
        'anti_martingale_enabled': True, 'anti_martingale_base_pct': 5.0,
        'anti_martingale_growth_factor': 2.0, 'anti_martingale_streak_target': 3,
    }

    print(f'[Schritt 1/3] Entry: LONG {SYMBOL} @ ~{price:.10f} | Anti-Martingale Basis=5%')
    result = execute_live_trade(exchange, signal, SYMBOL, strat_cfg, telegram_config, state_path=state_path)
    print(f'Ergebnis: {result}')
    assert result['action'] == 'entered', f"Erwartet 'entered', bekam: {result}"

    # Grobe Plausibilitaetspruefung der Kontraktmenge (keine exakte Gleichheit -- Bitgets
    # Mengen-Praezisionsrundung fuer PEPE kann sichtbar von der reinen Formel abweichen).
    expected_contracts = (balance * 0.05 * LEVERAGE) / price
    assert result['contracts'] > 0
    assert abs(result['contracts'] - expected_contracts) / expected_contracts < 0.20

    time.sleep(2)
    print('[Schritt 2/3] Pruefe offene Position...')
    positions = exchange.fetch_open_positions(SYMBOL)
    assert positions, 'Position nicht gefunden nach Entry.'
    print(f"Position offen: {positions[0]['side'].upper()} | Kontrakte: {positions[0].get('contracts')}")

    with open(state_path) as f:
        state = json.load(f)
    assert state['pending_position'] is not None
    assert state['pending_position']['balance_before'] == balance
    print(f'Anti-Martingale-Zustand nach Entry: {state}')

    # Cleanup: Position sauber schliessen (nicht auf ein reales SL/TP warten -- unvorhersehbar,
    # wie lange das dauert; Fixture-Teardown ist das Sicherheitsnetz falls das hier fehlschlaegt).
    print('[Schritt 3/3] Schliesse Position...')
    _force_close_if_open(exchange, SYMBOL)
    time.sleep(2)
    final_pos = exchange.fetch_open_positions(SYMBOL)
    assert len(final_pos) == 0, 'Position sollte geschlossen sein.'

    print('\n--- LIVE-TEST ERFOLGREICH ---')
