# scripts/build_sample_dataset.py
# Baut einen kleinen Beispiel-Datensatz aus echten BTC-Daten, um Schritt 1-4 der
# Spezifikation end-to-end zu verifizieren (Feature-Encoder -> Targets -> Dataset).
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')

from oraclebot.data.dataset import build_training_examples, save_dataset_jsonl
from oraclebot.utils.data_fetch import fetch_ohlcv

SYMBOL = 'BTC/USDT:USDT'
# Kleine Fenster fuer eine schnelle Stichprobe (Produktions-Defaults siehe settings.json).
WINDOW_SIZES = {'1d': 30, '4h': 60}
FETCH_LIMITS = {'1d': 250, '4h': 500}

if __name__ == '__main__':
    ohlcv_by_timeframe = {}
    for tf, limit in FETCH_LIMITS.items():
        print(f"Lade {SYMBOL} {tf} ({limit} Kerzen)...")
        ohlcv_by_timeframe[tf] = fetch_ohlcv(SYMBOL, tf, limit=limit)
        print(f"  -> {len(ohlcv_by_timeframe[tf])} Kerzen von {ohlcv_by_timeframe[tf].index[0]} bis {ohlcv_by_timeframe[tf].index[-1]}")

    examples = build_training_examples(
        ohlcv_by_timeframe, reference_timeframe='1d', window_sizes=WINDOW_SIZES,
    )

    out_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'sample_btc_1d.jsonl')
    save_dataset_jsonl(examples, out_path)

    print(f"\n{len(examples)} Beispiele gebaut. Erste 2 zur Kontrolle:\n")
    for ex in examples[:2]:
        preview = {k: (v if k in ('date', 'reference_time', 'target') else f"[{len(v)} Kerzen x {len(v[0])} Features]")
                   for k, v in ex.items()}
        print(json.dumps(preview, indent=2))
        print()
