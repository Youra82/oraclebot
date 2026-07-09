# scripts/fit_tokenizer.py
# Fittet den MarketTokenizer auf echten Daten und zeigt das entdeckte Vokabular
# sowie eine tokenisierte Kerzen-Sequenz (Schritt 5 der Spezifikation).
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')

from oraclebot.data.features import compute_features
from oraclebot.data.tokenizer import MarketTokenizer
from oraclebot.utils.data_fetch import fetch_ohlcv

SYMBOL = 'BTC/USDT:USDT'
TIMEFRAMES = {'1d': 1000, '4h': 1000, '1h': 1000}
N_TOKENS = 48  # klein gehalten fuer die Stichprobe; Produktion siehe settings.json (500-5000)

if __name__ == '__main__':
    feature_frames = []
    for tf, limit in TIMEFRAMES.items():
        print(f"Lade {SYMBOL} {tf} ({limit} Kerzen)...")
        ohlcv = fetch_ohlcv(SYMBOL, tf, limit=limit)
        feats = compute_features(ohlcv)
        print(f"  -> {len(feats)} Feature-Zeilen nach Warmup")
        feature_frames.append(feats)

    # Ein gemeinsames Vokabular ueber alle Timeframes hinweg (Features sind ATR-/Ratio-normiert,
    # also timeframe-uebergreifend vergleichbar).
    import pandas as pd
    pooled = pd.concat(feature_frames)

    tokenizer = MarketTokenizer(n_tokens=N_TOKENS, random_state=42)
    tokenizer.fit(pooled)

    out_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'tokenizer.pkl')
    tokenizer.save(out_path)
    print(f"\nTokenizer gespeichert: {out_path}")

    print(f"\nVokabular ({N_TOKENS} Tokens), sortiert nach Haeufigkeit:\n")
    summary = tokenizer.vocabulary_summary()
    for _, row in summary.head(15).iterrows():
        print(f"Token {row['token_id']:>3} (n={row['count']:>5}): {row['label']}")

    print("\nTokenisierte Sequenz der letzten 20 Tageskerzen:\n")
    daily_feats = feature_frames[0]
    last_20 = daily_feats.iloc[-20:]
    tokens = tokenizer.transform(last_20)
    for ts, token in zip(last_20.index, tokens):
        print(f"  {ts.date()}  ->  Token {token}")
    print("\nAls Sequenz:", ' '.join(f"T{t}" for t in tokens))
