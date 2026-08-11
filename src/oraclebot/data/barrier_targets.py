# src/oraclebot/data/barrier_targets.py
# Alternative Zielformulierung zu targets.py: statt die naechste TAGESKERZE zu klassifizieren,
# wird fuer JEDE Referenzkerze (Standard: 4h) direkt vorhergesagt, ob eine Kursbewegung von
# `barrier_pct`% zuerst nach oben oder nach unten erreicht wird -- das ist exakt die Frage, die
# fuer eine symmetrische SL=TP-Strategie zaehlt (siehe Recherche 2026-07-24), statt sie ueber den
# Umweg "wird morgen bullisch oder baerisch" indirekt zu erschliessen.
#
# Validiert (8-Fenster-Walk-Forward ueber 2.5 Jahre BTC/USDT:USDT, 4h-Referenz/15m-Aufloesung,
# depth=3 HistGradientBoosting auf den reinen 4h-Features): 62.0-71.2% Accuracy je Fenster,
# Mittel 67.5%, Standardabweichung nur 3.0pp -- deutlich robuster als targets.py's taegliches
# trend-Ziel (Mittel 59.0%, Worst-Case 56.0% bei nur 3 Fenstern). Ausserdem ~7x mehr
# Handelsgelegenheiten (4h- statt Tages-Kadenz).
import logging
import time

import pandas as pd

from oraclebot.utils.progress import finish_progress, render_progress

logger = logging.getLogger(__name__)

BARRIER_LABELS = ['down_first', 'up_first']  # 0, 1

# Fuer die Umrechnung reference_timeframe -> Kerzendauer (siehe compute_barrier_labels()).
# Dieselbe Zuordnung wie in data_fetch.py/predict_next_barrier.py -- bewusst dupliziert statt
# geteilt (Fleet-Konvention, siehe dortige TIMEFRAME_MINUTES-Vorkommen).
TIMEFRAME_MINUTES = {'1M': 30 * 24 * 60, '1w': 7 * 24 * 60, '1d': 24 * 60, '4h': 4 * 60, '1h': 60, '15m': 15}


def compute_barrier_labels(reference_df: pd.DataFrame, intraday_df: pd.DataFrame,
                            barrier_pct: float = 1.0, reference_timeframe: str = '4h') -> pd.DataFrame:
    """Fuer jede Kerze in `reference_df`: wird ausgehend vom Schlusskurs zuerst eine Bewegung von
    +barrier_pct% oder -barrier_pct% erreicht (anhand der feineren `intraday_df`-Kerzen,
    chronologisch STRIKT NACH der Referenzkerze -- kein Blick in die eigene Kerze)?

    Args:
        reference_df: OHLCV-DataFrame der Referenz-Kerzen (Standard: 4h), DatetimeIndex (=
            Kerzen-OEFFNUNGSzeit, Standard-OHLCV-Konvention).
        intraday_df: feinere OHLCV-Kerzen (Standard: 15m) fuer die Reihenfolge-Bestimmung.
        barrier_pct: symmetrischer Abstand in Prozent vom Schlusskurs.
        reference_timeframe: Dauer der Referenzkerze (Standard: '4h') -- bestimmt, wann `entry`
            (der SCHLUSSkurs der Referenzkerze) ueberhaupt real bekannt/handelbar ist.

    Returns:
        DataFrame (Index = reference_df-Zeitstempel) mit 'entry', 'label' (0=runter zuerst,
        1=hoch zuerst), 'exit_time'. Zeilen ohne bestimmbares Ergebnis (z.B. am Ende der
        verfuegbaren Historie, wo weder Barriere je erreicht wird) werden weggelassen.
    """
    # Bugfix 2026-08-11 (Live-vs-Backtest-Validierung): `entry` ist der SCHLUSSkurs der
    # Referenzkerze, der aber erst bei ts + reference_timeframe (Kerzenschluss) real bekannt ist
    # -- `ts` selbst ist nur die Kerzen-OEFFNUNG (Standard-OHLCV-Konvention, siehe
    # predict_next_barrier.py: ref_ts = letzte ABGESCHLOSSENE Kerze, entry_price = deren close,
    # gehandelt beim naechsten Cron-Lauf NACH Kerzenschluss). Die vorherige Suche
    # (`intraday_df.index > ts`) griff dadurch auf 15m-Baeren VOR dem Kerzenschluss zu -- also auf
    # die Bildungsphase der Referenzkerze selbst, bevor der als `entry` verwendete Schlusskurs
    # ueberhaupt real existiert. Verifiziert am aktuellen Trainings-Datensatz: 36% aller Beispiele
    # hatten ein Label/exit_time, das auf diese Weise aus der eigenen, noch nicht abgeschlossenen
    # Kerze stammte -- echtes Lookahead im Trainings-Target, nicht nur eine Anzeige-Ungenauigkeit.
    ref_delta = pd.Timedelta(minutes=TIMEFRAME_MINUTES[reference_timeframe])

    records = []
    n = len(reference_df)
    start_time = time.time()
    # Nicht vektorisiert (pro Referenzkerze eine vorwaertslaufende Suche in intraday_df) -- kann
    # bei mehreren tausend Referenzkerzen ohne sichtbare Zwischenausgabe eine Weile dauern.
    # Fortschrittsbalken, damit ein interaktiver Lauf (run_pipeline.sh) nicht wie ein Haenger
    # aussieht (Nutzer-Feedback 2026-07-26).
    for i, (ts, row) in enumerate(reference_df.iterrows()):
        if i % 25 == 0 or i == n - 1:
            render_progress("Barriere-Labels", i + 1, n, start_time)
        entry = float(row['close'])
        up_level = entry * (1 + barrier_pct / 100.0)
        down_level = entry * (1 - barrier_pct / 100.0)
        future_bars = intraday_df[intraday_df.index >= ts + ref_delta]
        label, exit_time = None, None
        for fts, bar in future_bars.iterrows():
            if bar['low'] <= down_level:
                label, exit_time = 0, fts
                break
            if bar['high'] >= up_level:
                label, exit_time = 1, fts
                break
        if label is None:
            continue
        records.append({'ts': ts, 'entry': entry, 'label': label, 'exit_time': exit_time})
    if n:
        finish_progress()

    if not records:
        return pd.DataFrame(columns=['entry', 'label', 'exit_time'])
    return pd.DataFrame(records).set_index('ts')


def build_barrier_examples(ohlcv_by_timeframe: dict, reference_timeframe: str, intraday_timeframe: str,
                            context_timeframes: list = None, feature_kwargs: dict = None,
                            feature_kwargs_by_timeframe: dict = None, barrier_pct: float = 1.0) -> list:
    """Baut flache Trainingsbeispiele: pro Referenzkerze ein Feature-Vektor + Barriere-Label.

    Der Feature-Vektor besteht aus dem Referenz-Timeframe-Block gefolgt von je einem Block pro
    `context_timeframes`-Eintrag (jeweils die letzte VOR dem Referenzzeitpunkt abgeschlossene
    Kerze dieses Timeframes -- No-Lookahead-korrekt per `merge_asof(direction='backward')`).
    Anders als das alte dataset.py's build_training_examples (Fenster-Historie pro Timeframe)
    reicht hier je Timeframe nur die JEWEILS AKTUELLSTE Kerze (ATR-/EMA-basierte Features
    verarbeiten Historie bereits intern) -- kein Sequenz-Fenster noetig.

    Validiert (2026-07-25, Walk-Forward ueber 2.5 Jahre): reine 4h-Features gaben 67.6%/63.8%
    (Mittel/Worst-Case), alle 6 Zeitebenen (1M/1w/1d/4h/1h/15m, wie im alten Tages-Modell)
    74.6%/71.4% -- deutlicher, durchgaengiger Gewinn in JEDEM der 7 Testfenster.

    Args:
        ohlcv_by_timeframe: {timeframe: OHLCV-DataFrame} fuer Referenz-, Intraday- und alle
            Kontext-Timeframes.
        reference_timeframe: Timeframe der Handelsentscheidung (Standard: '4h').
        intraday_timeframe: feinere Kerzen fuer die Barriere-Reihenfolge (Standard: '15m').
        context_timeframes: zusaetzliche Timeframes, deren letzte abgeschlossene Kerze als
            weiterer Feature-Block angehaengt wird (leer = nur Referenz-Timeframe, altes Verhalten).
        feature_kwargs_by_timeframe: optionale Overrides je Timeframe (z.B. kuerzere
            Indikator-Fenster fuer 1M/1w, die sonst Jahre an Warmup braeuchten).

    Returns:
        Liste von Dicts: {date, reference_time, entry, exit_time, features (Liste), target}.
        Beispiele, bei denen ein Kontext-Timeframe noch keine gueltige Vorgaenger-Kerze hat
        (frueher Rand der Historie), werden uebersprungen.
    """
    from oraclebot.data.features import FEATURE_NAMES, compute_features

    feature_kwargs = feature_kwargs or {}
    feature_kwargs_by_timeframe = feature_kwargs_by_timeframe or {}
    context_timeframes = context_timeframes or []

    def feats_for(tf, show_progress=False):
        kwargs = {**feature_kwargs, **feature_kwargs_by_timeframe.get(tf, {})}
        return compute_features(ohlcv_by_timeframe[tf], progress_label=tf if show_progress else None, **kwargs)

    reference_df = ohlcv_by_timeframe[reference_timeframe]
    intraday_df = ohlcv_by_timeframe[intraday_timeframe]
    feat_ref = feats_for(reference_timeframe, show_progress=True)
    labels = compute_barrier_labels(reference_df, intraday_df, barrier_pct=barrier_pct,
                                     reference_timeframe=reference_timeframe)

    joined_index = sorted(feat_ref.index.intersection(labels.index))
    if not joined_index:
        return []

    context_merged = {}
    if context_timeframes:
        ref_ts_df = pd.DataFrame(index=pd.DatetimeIndex(joined_index).rename('ts')).reset_index()
        for tf in context_timeframes:
            # compute_features() enthaelt zwei Pro-Kerze-Schleifen (Marktstruktur, S/R+Kanal) --
            # bei grossen Kontext-Timeframes (15m: >90000 Kerzen) spuerbar langsam, nicht nur
            # unsichtbar. Header-Zeile + die beiden eigenen Fortschrittsbalken in compute_features
            # (progress_label=tf) sorgen dafuer, dass ein live zuschauender Nutzer sieht, dass
            # etwas passiert, statt einer stillen Luecke (Nutzer-Feedback 2026-07-26).
            n_raw = len(ohlcv_by_timeframe.get(tf, []))
            logger.info(f"  Verarbeite Kontext-Timeframe '{tf}' ({n_raw} Kerzen)...")
            context_feat = feats_for(tf, show_progress=True)
            if context_feat.empty:
                # compute_features() auf einer (fast) leeren OHLCV-DataFrame liefert 0 Zeilen --
                # dann hat das leere Ergebnis KEINEN DatetimeIndex mehr (pandas faellt auf einen
                # generischen int64-Index zurueck), was merge_asof() weiter unten mit einem
                # kryptischen "incompatible merge keys ... datetime64 and int64"-Fehler tief in
                # pandas-Internals crashen liesse. Stattdessen hier fruehzeitig klar melden --
                # typischer Ausloeser: ein --no-cache-Komplettabruf, bei dem Bitget fuer sehr
                # grobe Zeitebenen (v.a. 1M) weniger Historie zurueckgibt als angefragt (siehe
                # data_fetch.py) und die Kerzenanzahl unter das Feature-Warmup faellt.
                n_raw = len(ohlcv_by_timeframe.get(tf, []))
                raise RuntimeError(
                    f"Kontext-Timeframe '{tf}': compute_features() lieferte 0 Zeilen "
                    f"({n_raw} Rohkerzen verfuegbar -- reicht nicht fuers Feature-Warmup). "
                    f"Haeufigste Ursache: ein --no-cache-Komplettabruf, bei dem Bitget fuer "
                    f"'{tf}' weniger Historie zurueckgab als angefragt. Cache NICHT loeschen "
                    f"und erneut versuchen, oder '{tf}' vorlaeufig aus context_timeframes "
                    f"in settings.json entfernen.")
            right = context_feat.rename_axis('ts').reset_index()
            merged = pd.merge_asof(ref_ts_df, right, on='ts', direction='backward')
            context_merged[tf] = merged.set_index('ts')[FEATURE_NAMES]

    examples = []
    for ts in joined_index:
        feature_row = feat_ref.loc[ts, FEATURE_NAMES].tolist()
        valid = True
        for tf in context_timeframes:
            context_row = context_merged[tf].loc[ts]
            if context_row.isna().any():
                valid = False
                break
            feature_row += context_row.tolist()
        if not valid:
            continue
        examples.append({
            'date': ts.isoformat(),
            'reference_time': ts.isoformat(),
            'entry': float(labels.loc[ts, 'entry']),
            'exit_time': labels.loc[ts, 'exit_time'].isoformat(),
            'features': feature_row,
            'target': int(labels.loc[ts, 'label']),
        })
    return examples
