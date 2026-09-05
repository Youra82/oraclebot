# scripts/predict_next_barrier.py
# Live-Inferenz + Trading fuer das Barriere-Modell (4h-Kadenz statt taeglich): laedt die
# aktuellste ABGESCHLOSSENE Referenzkerze (Standard 4h), sagt vorher ob zuerst +barrier_pct%
# oder -barrier_pct% erreicht wird, und platziert bei ausreichender Konfidenz einen Live-Trade.
# Analog zu predict_next_candle.py, aber deutlich einfacher (kein Transformer, kein
# Multi-Timeframe-Fenster, keine Preis-Rekonstruktion -- das Barriere-Modell braucht nur die
# aktuellste Referenzkerze).
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import pandas as pd

from oraclebot.data.features import compute_features
from oraclebot.model.barrier_model import BarrierPredictor
from oraclebot.strategy.barrier_signal import compute_barrier_signal
from oraclebot.utils.barrier_gate import check_barrier_gate, mark_barrier_run_complete
from oraclebot.utils.config import load_barrier_config
from oraclebot.utils.config import load_settings as load_settings_json
from oraclebot.utils.data_fetch import fetch_ohlcv_incremental, resample_ohlcv
from oraclebot.utils.telegram import send_message

TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}
# Genug Kerzen fuers laengste Feature-Warmup (EMA-50/MACD) je Timeframe. 1M/1w nutzen in
# feature_settings_by_timeframe deutlich kleinere Fenster (siehe settings.json) -- 60 Kerzen
# reichen dort.
MIN_CANDLES_BY_TF = {'1M': 60, '1w': 60, '1d': 120, '4h': 120, '1h': 120, '15m': 120}


def _drop_incomplete_last_candle(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Wie in predict_next_candle.py -- das Modell hat im Training nur abgeschlossene Kerzen
    gesehen (siehe No-Lookahead-Regel in barrier_targets.py)."""
    if df.empty:
        return df
    now = pd.Timestamp.now(tz='UTC')
    last_open = df.index[-1]
    close_time = last_open + pd.Timedelta(minutes=TIMEFRAME_MINUTES[timeframe])
    return df.iloc[:-1] if now < close_time else df


def _log_feature_block(label: str, ts, names: list, values: list) -> None:
    """Loggt einen Feature-Block als eine JSON-Zeile (Timestamp + Name->Wert, 8 Nachkommastellen).

    Eingefuehrt 2026-09-05 zur Live-vs-Offline-Diagnose: ein rekonstruierter Backtest mit der
    exakt gleichen, aus Git extrahierten Modell-Datei ergab fuer dieselbe historische
    Referenzkerze (identischer Entry-Preis) die GEGENTEILIGE Richtung bei abweichender Konfidenz
    (Live: down_first 66.4% vs. Offline-Rekonstruktion: up_first 63.2%, Kerze vom 2026-08-31
    08:00 UTC, 4 Stunden alt zum Entscheidungszeitpunkt -- also kein "Kerze noch nicht
    settled"-Effekt). Da nur Konfidenz/Richtung geloggt wurden, liess sich nicht feststellen,
    welcher der ~21 Werte je Block (Referenz + 5 Kontext-Timeframes) dafuer verantwortlich ist.
    Diese Zeilen ermoeglichen den direkten Abgleich mit den 'features' aus dem Offline-Datensatz
    (artifacts/datasets/barrier_*.jsonl) fuer dieselbe Referenzkerze."""
    payload = {'block': label, 'ts': str(ts), **{n: round(float(v), 8) for n, v in zip(names, values)}}
    logger.info(f"FEATURE-DUMP {json.dumps(payload, sort_keys=False)}")


def load_secrets(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true',
                         help="Ignoriert das Zeitfenster-Gate und den Perioden-Marker, laeuft "
                              "sofort. Fuer manuelles Testen -- markiert die Periode NICHT als erledigt.")
    parser.add_argument('--simulate-now', type=str, default=None,
                         help="Ueberschreibt die fuers Gate verwendete UTC-Zeit ('YYYY-MM-DD HH:MM'). "
                              "Nur fuers Testen des Gate+Marker-Zusammenspiels.")
    parser.add_argument('--marker-path', type=str, default=None)
    args = parser.parse_args()

    settings = load_settings_json()
    barrier_cfg = load_barrier_config(settings)

    symbol = barrier_cfg.get('symbol', 'BTC/USDT:USDT')
    reference_tf = barrier_cfg.get('reference_timeframe', '4h')
    intraday_tf = barrier_cfg.get('intraday_timeframe', '15m')
    context_tfs = barrier_cfg.get('context_timeframes', [])
    barrier_pct = barrier_cfg.get('barrier_pct', 1.0)
    min_confidence = barrier_cfg.get('min_confidence', 0.60)

    now_utc = pd.Timestamp(args.simulate_now, tz='UTC') if args.simulate_now else pd.Timestamp.now(tz='UTC')
    if args.simulate_now:
        logger.warning(f"--simulate-now aktiv: Gate+Marker verwenden {now_utc} statt der echten Systemzeit.")
    marker_path = args.marker_path or os.path.join(
        os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'last_barrier_run.txt')
    if not args.force:
        should_run, skip_reason = check_barrier_gate(now_utc, marker_path, period_hours=4)
        if not should_run:
            print(skip_reason)
            sys.exit(0)

    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets')
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    model_path = os.path.join(artifacts_dir, f"barrier_model_{safe_symbol}_{reference_tf}.pkl")
    if not os.path.exists(model_path):
        logger.error(f"Kein Barriere-Modell gefunden: {model_path}. Erst train_barrier_model.py ausfuehren.")
        sys.exit(1)
    predictor = BarrierPredictor.load(model_path)

    logger.info(f"Lade Marktdaten fuer {symbol} ({reference_tf}, inkrementeller Live-Cache)...")
    cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_{reference_tf}.pkl")
    # Genug Historie fuer das laengste Feature-Warmup (EMA-50/MACD) plus Sicherheitsmarge.
    min_candles = 120
    df = fetch_ohlcv_incremental(symbol, reference_tf, min_candles=min_candles, cache_path=cache_path)
    df = _drop_incomplete_last_candle(df, reference_tf)
    logger.info(f"  {reference_tf}: {len(df)} Kerzen, letzte abgeschlossene: {df.index[-1]}")

    # Sicherheitsnetz gegen stille Cache-/Fetch-Fehler (wie predict_next_candle.py): die letzte
    # abgeschlossene Referenzkerze darf nicht aelter als das 2-fache der Periodenlaenge sein.
    staleness = now_utc - df.index[-1]
    max_staleness = pd.Timedelta(minutes=TIMEFRAME_MINUTES[reference_tf]) * 2
    if staleness > max_staleness and not args.force:
        message = (f"ACHTUNG oraclebot (Barriere-Strategie): letzte abgeschlossene {reference_tf}-Kerze "
                    f"ist {staleness} alt (Grenze: {max_staleness}). Moeglicher Fetch-/Cache-Fehler -- "
                    f"breche ab statt eine Prognose fuer eine veraltete Kerze zu senden.")
        logger.error(message)
        secrets = load_secrets(os.path.join(os.path.dirname(__file__), '..', 'secret.json'))
        telegram_cfg = secrets.get('telegram', {})
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)
        sys.exit(1)

    feat = compute_features(df, **barrier_cfg['feature_settings'])
    if len(feat) == 0:
        logger.error("compute_features() lieferte keine Zeilen (zu wenig Historie fuer Warmup). Breche ab.")
        sys.exit(1)

    from oraclebot.data.features import FEATURE_NAMES
    ref_ts = feat.index[-1]
    entry_price = float(df.loc[ref_ts, 'close'])
    feature_row = feat.loc[ref_ts, FEATURE_NAMES].tolist()
    _log_feature_block(reference_tf, ref_ts, FEATURE_NAMES, feature_row)

    # Kontext-Timeframes (siehe barrier_targets.build_barrier_examples): je Timeframe die letzte
    # VOR/BEI der Referenzkerze abgeschlossene Kerze anhaengen -- entspricht live demselben
    # merge_asof(direction='backward'), das beim Training verwendet wurde.
    feature_kwargs_by_timeframe = barrier_cfg.get('feature_settings_by_timeframe', {})
    for ctx_tf in context_tfs:
        # '1d' wird u.U. zweimal gegen dieselbe Cache-Datei aufgerufen: hier direkt als eigener
        # Kontext-Block UND unten fuer die '1M'-Ableitung mit history_days als min_candles. Beide
        # Aufrufe MUESSEN denselben (groesseren) Wert nutzen -- sonst wuerde der kleinere Aufruf
        # die Cache-Datei per Groessenkappung (min_candles*3) sofort wieder auf den kleineren Wert
        # zurueckschneiden und den 1M-Backfill (siehe unten) bei jedem Lauf zunichtemachen.
        if ctx_tf == '1d' and '1M' in context_tfs:
            ctx_min_candles = barrier_cfg.get('history_days', 1000)
        else:
            ctx_min_candles = MIN_CANDLES_BY_TF.get(ctx_tf, 120)
        logger.info(f"Lade Kontext-Timeframe {ctx_tf}...")
        if ctx_tf == '1M':
            # Bitgets eigener '1M'-Endpunkt ist unzuverlaessig (siehe data_fetch.fetch_all_timeframes
            # fuer die Begruendung -- dort deshalb schon seit 2026-07-26 per resample_ohlcv() aus '1d'
            # abgeleitet statt direkt abgefragt). Dieser Live-Pfad hier fetchte bislang trotzdem noch
            # direkt gegen den '1M'-Endpunkt -- Symptom live: der inkrementelle Fetch lieferte
            # wiederholt NICHTS Neues, der Cache blieb auf derselben Monatskerze eingefroren, waehrend
            # die 4h-Referenzkerze weiterlief, bis die Kontext-Luecke die 60-Tage-Alarmgrenze riss
            # (Telegram-Alarme ab 2026-08-30). Fix: wie beim Training aus '1d' resamplen.
            #
            # Bugfix 2026-09-05 (Live-vs-Offline-Feature-Diff): `min_candles` MUSS exakt
            # `history_days` aus dem Training entsprechen (nicht eine eigene Formel wie vorher
            # `ctx_min_candles * 31`) -- sonst bekommt dieser Live-Cache eine ANDERE Gesamtlaenge
            # an '1d'-Historie als der Offline-Datensatz, aus dem das Modell trainiert wurde. Bei
            # sehr kurzen 1M-Fenstern (feature_settings_by_timeframe: atr/ema/macd_slow=6 Monate)
            # reicht schon ein unterschiedlicher Historien-Start, um die abgeleiteten Monats-
            # Indikatoren spuerbar zu verschieben -- verifiziert: Live zeigte fuer eine Kerze
            # `down_first 66.4%`, dieselbe Modell-Datei auf offline nachgebauten Features
            # `up_first 63.2%`, exakt in diesem Kontext-Block, alle anderen Bloecke identisch.
            d1_cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_1d.pkl")
            d1_min_candles = barrier_cfg.get('history_days', 1000)
            d1_df = fetch_ohlcv_incremental(symbol, '1d', min_candles=d1_min_candles, cache_path=d1_cache_path)
            ctx_df = resample_ohlcv(d1_df, '1M')
        else:
            ctx_cache_path = os.path.join(artifacts_dir, f"ohlcv_live_{safe_symbol}_{ctx_tf}.pkl")
            ctx_df = fetch_ohlcv_incremental(symbol, ctx_tf, min_candles=ctx_min_candles, cache_path=ctx_cache_path)
        ctx_df = _drop_incomplete_last_candle(ctx_df, ctx_tf)
        ctx_kwargs = {**barrier_cfg['feature_settings'], **feature_kwargs_by_timeframe.get(ctx_tf, {})}
        ctx_feat = compute_features(ctx_df, **ctx_kwargs)
        ctx_feat = ctx_feat[ctx_feat.index <= ref_ts]
        if len(ctx_feat) == 0:
            logger.error(f"Kontext-Timeframe {ctx_tf}: keine gueltige Kerze <= Referenzzeitpunkt {ref_ts}. Breche ab.")
            sys.exit(1)
        ctx_ts = ctx_feat.index[-1]
        ctx_gap = ref_ts - ctx_ts
        max_ctx_gap = pd.Timedelta(minutes=TIMEFRAME_MINUTES[ctx_tf]) * 2
        if ctx_gap > max_ctx_gap and not args.force:
            message = (f"ACHTUNG oraclebot (Barriere-Strategie): Kontext-Timeframe {ctx_tf} ist "
                        f"{ctx_gap} hinter der Referenzkerze zurueck (Grenze: {max_ctx_gap}). "
                        f"Moeglicher Fetch-/Cache-Fehler -- breche ab statt mit veraltetem Kontext zu handeln.")
            logger.error(message)
            secrets_early = load_secrets(os.path.join(os.path.dirname(__file__), '..', 'secret.json'))
            telegram_early = secrets_early.get('telegram', {})
            send_message(telegram_early.get('bot_token'), telegram_early.get('chat_id'), message)
            sys.exit(1)
        logger.info(f"  {ctx_tf}: letzte abgeschlossene Kerze <= Referenz: {ctx_ts}")
        ctx_values = ctx_feat.loc[ctx_ts, FEATURE_NAMES].tolist()
        _log_feature_block(ctx_tf, ctx_ts, FEATURE_NAMES, ctx_values)
        feature_row += ctx_values

    predicted_class, confidence = predictor.predict_one(feature_row)
    from oraclebot.data.barrier_targets import BARRIER_LABELS
    logger.info(f"\nReferenzkerze: {feat.index[-1]} | Entry: {entry_price:.2f}")
    logger.info(f"Vorhersage: {BARRIER_LABELS[predicted_class]} (Konfidenz: {confidence:.1%})")

    signal = compute_barrier_signal(predicted_class, confidence, entry_price,
                                     min_confidence=min_confidence, barrier_pct=barrier_pct)

    if signal['direction'] is None:
        logger.info(f"Kein Trade ({signal['reason']}, Konfidenz {signal['confidence']:.1%} < {min_confidence:.1%}).")
    else:
        logger.info(f"Signal: {signal['direction'].upper()} | SL: {signal['stop_loss']:.2f} | TP: {signal['take_profit']:.2f}")

    secret_path = os.path.join(os.path.dirname(__file__), '..', 'secret.json')
    secrets = load_secrets(secret_path)
    telegram_cfg = secrets.get('telegram', {})

    if barrier_cfg.get('live_trading_enabled', False):
        oraclebot_accounts = secrets.get('oraclebot', [])
        if not oraclebot_accounts or not oraclebot_accounts[0].get('apiKey'):
            logger.error("live_trading_enabled=true, aber keine 'oraclebot'-API-Keys in secret.json gefunden.")
        else:
            from oraclebot.strategy.live_trade import execute_live_trade
            from oraclebot.utils.exchange import Exchange
            exchange = Exchange(oraclebot_accounts[0])
            am_state_path = os.path.join(artifacts_dir, '..', 'state', 'barrier_anti_martingale_state.json')
            result = execute_live_trade(exchange, signal, symbol, barrier_cfg, telegram_cfg, state_path=am_state_path)
            logger.info(f"\nLive-Trading-Ergebnis: {result}")
    else:
        logger.info("\n(Dry-Run: barrier_strategy_settings.live_trading_enabled=false -- kein echter Trade.)")

    if settings.get('notification_settings', {}).get('telegram_enabled', False):
        dir_text = signal['direction'].upper() if signal['direction'] else 'KEIN TRADE'
        message = (
            f"oraclebot Barriere-Signal: {symbol} ({reference_tf})\n"
            f"Referenzkerze: {feat.index[-1]}\n"
            f"Entry: {entry_price:.2f}\n"
            f"Vorhersage: {BARRIER_LABELS[predicted_class]} (Konfidenz {confidence:.1%})\n"
            f"Richtung: {dir_text}"
        )
        if signal['direction']:
            message += f"\nSL: {signal['stop_loss']:.2f}\nTP: {signal['take_profit']:.2f}"
        send_message(telegram_cfg.get('bot_token'), telegram_cfg.get('chat_id'), message)

    if not args.force:
        mark_barrier_run_complete(now_utc, marker_path, period_hours=4)
