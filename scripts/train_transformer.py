# scripts/train_transformer.py
# End-to-End-Pipeline (Spezifikation Schritt 1-7): echte Multi-Timeframe-Daten laden,
# Trainingsbeispiele bauen, FeatureScaler fitten, MarketTransformer trainieren
# (mit Early Stopping + Best-Checkpoint nach Out-of-Sample-Genauigkeit), Beam-Search-Beispiel zeigen.
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import ta
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from oraclebot.data.dataset import build_training_examples, save_dataset_jsonl
from oraclebot.data.features import FEATURE_NAMES, compute_features
from oraclebot.data.scaler import FeatureScaler
from oraclebot.data.targets import TARGET_NAMES
from oraclebot.model.dataset_torch import ContinuousMarketDataset, collate_fn
from oraclebot.model.reconstruct import reconstruct_candle
from oraclebot.model.transformer import MarketTransformer, TARGET_CARDINALITIES, estimate_attention_memory_bytes
from oraclebot.model.tree_ensemble import TREE_TARGETS, TreeEnsemblePredictor, flat_features
from oraclebot.utils.data_fetch import fetch_all_timeframes

# Durchschnittliche Trefferquote von reinem Raten (1/Anzahl Klassen je Zielvariable),
# als Referenzlinie fuer das Robust/Grenzwertig/Overfitted-Verdict.
RANDOM_BASELINE_ACCURACY = sum(1 / card for card in TARGET_CARDINALITIES.values()) / len(TARGET_CARDINALITIES)


def compute_verdict(train_acc: float, val_acc: float, gap_threshold: float = 0.15) -> str:
    """Robust/Grenzwertig/Overfitted-Verdict, analog zu probebot's OOS-Test (70/30 Split,

    In-Sample- vs Out-of-Sample-Vergleich). `train_acc`/`val_acc` sind ueber alle
    Zielvariablen gemittelte Genauigkeiten.
    """
    gap = train_acc - val_acc
    if val_acc > RANDOM_BASELINE_ACCURACY + 0.05 and gap < gap_threshold:
        return 'Robust'
    if val_acc > RANDOM_BASELINE_ACCURACY - 0.05:
        return 'Grenzwertig'
    return 'Overfitted'


def load_settings(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_accuracy(model, loader, device) -> dict:
    """Schnelle Genauigkeits-Auswertung je Zielvariable (fuer Early Stopping, jede Epoche)."""
    model.eval()
    correct = {name: 0 for name in TARGET_NAMES}
    total = 0
    with torch.no_grad():
        for features, targets in loader:
            features = {tf: t.to(device) for tf, t in features.items()}
            logits, _ = model(features, targets=None)
            batch_size = next(iter(targets.values())).size(0)
            total += batch_size
            for name in TARGET_NAMES:
                preds = logits[name].argmax(dim=-1).cpu()
                correct[name] += (preds == targets[name]).sum().item()
    return {name: correct[name] / total for name in TARGET_NAMES} if total else {}


def compute_epoch_diagnostics(model, loader, device, boosted_targets: list) -> dict:
    """Objektive Kennzahlen zum Trainingsverlauf, unabhaengig von der Trainings-Loss selbst --

    fuer die Kollaps-Zuverlaessigkeits-Untersuchung (2026-07-10): erlaubt zu pruefen, ob sich
    spaeter kollabierende von divers bleibenden Laeufen schon frueh (Epoche 1-3) unterscheiden
    lassen, statt nur am Trainingsende festzustellen "kollabiert oder nicht".

    - tf_weight_std: Streuung der GEMITTELTEN Fusion-Attention-Gewichte ueber die Timeframes
      (niedrig = uniforme Fusion, hoch = spezialisiert -- siehe Fusion-Kollaps vom 2026-07-09).
    - trend_pred_std: Streuung der P(bullish)-Einzelvorhersagen ueber das Validierungsset
      (dieselbe Groesse wie der Diversitaets-Regularisierer im Training, hier aber als reine
      MESSUNG auf ungesehenen Daten, nicht als Trainingsziel).
    - Gewichtsnormen der geboosteten Parameter (Fusion-Attention, geboostete Koepfe), um zu
      sehen ob/wie schnell sie sich vom Init entfernen.
    """
    model.eval()
    all_tf_weights, all_trend_probs = [], []
    with torch.no_grad():
        for features, targets in loader:
            features = {tf: t.to(device) for tf, t in features.items()}
            logits, tf_weights = model(features, targets=None)
            all_tf_weights.append(tf_weights.cpu())
            all_trend_probs.append(F.softmax(logits['trend'], dim=-1)[:, -1].cpu())
    tf_weights_cat = torch.cat(all_tf_weights, dim=0)
    trend_probs_cat = torch.cat(all_trend_probs, dim=0)

    diagnostics = {
        'tf_weight_std': float(tf_weights_cat.mean(dim=0).std().item()),
        'trend_pred_std': float(trend_probs_cat.std(unbiased=False).item()),
        'fusion_attn_weight_norm': float(model.fusion.attention.in_proj_weight.norm().item()),
    }
    for name in boosted_targets:
        idx = model.decoder.step_names.index(name)
        diagnostics[f'{name}_head_weight_norm'] = float(model.decoder.heads[idx][-1].weight.norm().item())
    return diagnostics


def _brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """Brier Score (mittlere quadratische Abweichung Wahrscheinlichkeit vs. One-Hot-Ziel). Kleiner = besser."""
    one_hot = np.eye(num_classes)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def _expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: gewichtete Differenz zwischen Konfidenz und tatsaechlicher

    Trefferquote ueber Konfidenz-Buckets. 0 = perfekt kalibriert.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    n = len(labels)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not mask.any():
            continue
        ece += (mask.sum() / n) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece)


def evaluate_with_calibration(model, loader, device) -> dict:
    """Vollstaendige Auswertung je Zielvariable: Genauigkeit, Top-3-Trefferquote, Brier Score, ECE.

    Kalibrierungs-Metriken zeigen, ob die vom Modell ausgegebenen Wahrscheinlichkeiten
    tatsaechlich etwas bedeuten (z.B. "60% bullisch" sollte auch ~60% der Zeit eintreffen) --
    reine Argmax-Genauigkeit sagt darueber nichts aus.
    """
    model.eval()
    all_probs = {name: [] for name in TARGET_NAMES}
    all_labels = {name: [] for name in TARGET_NAMES}
    with torch.no_grad():
        for features, targets in loader:
            features = {tf: t.to(device) for tf, t in features.items()}
            logits, _ = model(features, targets=None)
            for name in TARGET_NAMES:
                all_probs[name].append(F.softmax(logits[name], dim=-1).cpu().numpy())
                all_labels[name].append(targets[name].numpy())

    result = {}
    for name in TARGET_NAMES:
        probs = np.concatenate(all_probs[name])
        labels = np.concatenate(all_labels[name])
        card = TARGET_CARDINALITIES[name]
        preds = probs.argmax(axis=1)
        top_k = min(3, card)
        top_k_preds = np.argsort(-probs, axis=1)[:, :top_k]
        result[name] = {
            'accuracy': float((preds == labels).mean()),
            'top3_accuracy': float(np.mean([label in row for label, row in zip(labels, top_k_preds)])),
            'brier': _brier_score(probs, labels, card),
            'ece': _expected_calibration_error(probs, labels),
        }
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', default=None, help='Einzelnes Symbol (ueberschreibt --symbols/settings.json auf genau eins)')
    parser.add_argument('--symbols', default=None, help='Kommagetrennte Liste von Symbolen (Standard: dataset_settings.symbols aus settings.json). Beispiele werden gepoolt fuer mehr Trainingsdaten.')
    parser.add_argument('--history-days', type=int, default=None, help='Ueberschreibt training_settings.history_days aus settings.json')
    parser.add_argument('--epochs', type=int, default=None, help='Ueberschreibt training_settings.epochs (max. Epochen; Early Stopping kann frueher abbrechen)')
    parser.add_argument('--d-model', type=int, default=None, help='Ueberschreibt model_settings.d_model (z.B. fuer schnelle Smoke-Tests)')
    parser.add_argument('--max-examples', type=int, default=None, help='Begrenzt die Anzahl Trainingsbeispiele (Smoke-Test)')
    parser.add_argument('--no-cache', action='store_true', help='Ignoriert den OHLCV-Cache und fetcht alle Timeframes neu')
    parser.add_argument('--batch-size', type=int, default=None, help='Ueberschreibt training_settings.batch_size')
    parser.add_argument('--force-unsafe', action='store_true', help='Ignoriert die Speicher-Sicherheitspruefung (NICHT auf diesem Rechner verwenden, siehe feedback_cpu_memory_safety)')
    args = parser.parse_args()

    settings_path = os.path.join(os.path.dirname(__file__), '..', 'settings.json')
    settings = load_settings(settings_path)
    ds_cfg = settings['dataset_settings']
    model_cfg = settings['model_settings']
    train_cfg = settings['training_settings']

    history_days = args.history_days or train_cfg['history_days']
    epochs = args.epochs or train_cfg['epochs']
    d_model = args.d_model or model_cfg['d_model']
    batch_size = args.batch_size or train_cfg['batch_size']
    timeframes = model_cfg['timeframes']
    n_features = len(FEATURE_NAMES)

    torch.set_num_threads(train_cfg.get('num_threads', 4))

    # Sicherheitspruefung: ein frueherer Versuch mit batch_size=32/seq_len=2000 hat den
    # Trainings-PC durch RAM-Erschoepfung dreimal zum Absturz gebracht (siehe
    # feedback_cpu_memory_safety in der Projekt-Memory). Vor jedem Lauf abschaetzen statt hoffen.
    estimated_bytes = estimate_attention_memory_bytes(
        ds_cfg['window_sizes'], timeframes, batch_size, model_cfg['nhead'], model_cfg['num_encoder_layers'],
        d_model=d_model, dim_feedforward=model_cfg['dim_feedforward'])
    max_bytes = train_cfg.get('max_attention_memory_bytes', 1024 ** 3)
    logger.info(f"Geschaetzter Attention-Speicherbedarf: {estimated_bytes / 1024**2:.0f} MB (Limit: {max_bytes / 1024**2:.0f} MB)")
    if estimated_bytes > max_bytes and not args.force_unsafe:
        raise RuntimeError(
            f"Geschaetzter Speicherbedarf ({estimated_bytes / 1024**2:.0f} MB) ueberschreitet das Sicherheitslimit "
            f"({max_bytes / 1024**2:.0f} MB). batch_size/window_sizes reduzieren oder --force-unsafe setzen "
            f"(nicht empfohlen -- siehe feedback_cpu_memory_safety)."
        )

    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = ds_cfg.get('symbols', ['BTC/USDT:USDT'])

    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets')
    feature_kwargs_by_tf = ds_cfg.get('feature_settings_by_timeframe', {})
    val_split = train_cfg['val_split']

    # Pro Symbol einzeln fetchen, Features bauen und CHRONOLOGISCH (pro Symbol!) in Training/
    # Validierung splitten, bevor alles zusammen gepoolt wird -- ein einzelner Split auf der
    # gepoolten Liste wuerde sonst ggf. nur ein Symbol in der Validierung landen lassen, je
    # nachdem in welcher Reihenfolge die Symbole verarbeitet wurden.
    ohlcv_by_symbol = {}
    all_train_examples, all_val_examples = [], []
    all_feature_frames = []

    for symbol in symbols:
        logger.info(f"\n=== {symbol} ===")
        ohlcv_by_timeframe = fetch_all_timeframes(symbol, timeframes, history_days, cache_dir=cache_dir, use_cache=not args.no_cache)
        ohlcv_by_symbol[symbol] = ohlcv_by_timeframe

        examples = build_training_examples(
            ohlcv_by_timeframe, reference_timeframe=ds_cfg['reference_timeframe'],
            window_sizes=ds_cfg['window_sizes'], feature_kwargs=ds_cfg['feature_settings'],
            feature_kwargs_by_timeframe=feature_kwargs_by_tf,
            target_kwargs=ds_cfg['target_settings'],
        )
        for ex in examples:
            ex['symbol'] = symbol
        if len(examples) < 10:
            logger.warning(f"Nur {len(examples)} Beispiele fuer {symbol} -- wird trotzdem mit gepoolt.")

        n_val = max(1, int(len(examples) * val_split))
        all_train_examples.extend(examples[:-n_val])
        all_val_examples.extend(examples[-n_val:])

        for tf, df in ohlcv_by_timeframe.items():
            feats = compute_features(df, **{**ds_cfg['feature_settings'], **feature_kwargs_by_tf.get(tf, {})})
            if len(feats):
                all_feature_frames.append(feats)

    if args.max_examples:
        all_train_examples = all_train_examples[:args.max_examples]
    examples = all_train_examples + all_val_examples
    if len(examples) < 10:
        raise RuntimeError(f"Nur {len(examples)} Trainingsbeispiele insgesamt -- zu wenig Historie geladen (history_days erhoehen).")

    symbols_tag = '_'.join(s.replace('/', '_').replace(':', '_') for s in symbols)
    dataset_path = os.path.join(cache_dir, f"{symbols_tag}_full.jsonl")
    save_dataset_jsonl(examples, dataset_path)

    logger.info(f"\nFitte FeatureScaler auf gepoolten Features aller Timeframes und Symbole ({', '.join(symbols)})...")
    pooled = pd.concat(all_feature_frames)
    scaler = FeatureScaler().fit(pooled)
    scaler.save(os.path.join(cache_dir, 'scaler_full.pkl'))

    train_examples, val_examples = all_train_examples, all_val_examples
    logger.info(f"\n{len(train_examples)} Trainings- / {len(val_examples)} Validierungsbeispiele "
                f"(ueber {len(symbols)} Symbol(e): {', '.join(symbols)})")

    train_ds = ContinuousMarketDataset(train_examples, scaler, timeframes)
    val_ds = ContinuousMarketDataset(val_examples, scaler, timeframes)
    train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(train_ds)), shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_ds)), shuffle=False, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Welche Decoder-Koepfe eine eigene (hoehere) LR + staerkere Init + Diversitaets-
    # Regularisierung bekommen. trend war der urspruengliche Fall (kein Vorwissen aus
    # vorherigen Schritten, am anfaelligsten fuer Kollaps auf die Trainings-Mehrheitsklasse,
    # 2026-07-09); close_position/upper_wick/lower_wick zeigten im OOS-Test dieselbe Schwaeche
    # (36-48% statt deutlich ueber Zufall), vermutlich derselbe Mechanismus, nur schwaecher.
    boosted_targets = train_cfg.get('boosted_targets', ['trend'])
    model = MarketTransformer(
        n_features=n_features, timeframes=timeframes, window_sizes=ds_cfg['window_sizes'],
        d_model=d_model, nhead=model_cfg['nhead'], num_encoder_layers=model_cfg['num_encoder_layers'],
        dim_feedforward=model_cfg['dim_feedforward'], dropout=model_cfg['dropout'],
        boosted_targets=boosted_targets,
    ).to(device)
    # Eigene (hoehere) Lernrate fuer die TimeframeFusion-Attention: mit einer einzigen
    # gemeinsamen LR blieb sie ueber viele Epochen bei fast uniformen Gewichten haengen
    # (alle Timeframes ~14%, unabhaengig vom Input) -- ein 5-Epochen-Diagnoselauf mit 10x LR
    # nur fuer diese Parameter zeigte ab Epoche 4 klare Spezialisierung (15m-Gewicht 14%->37%)
    # und erstmals wieder unterschiedliche statt konstante Trend-Vorhersagen. Ohne diesen
    # Boost reicht die normale Basis-LR nicht, um die Fusion aus ihrem Init-nahen Fixpunkt
    # herauszubewegen -- der Repraesentations-Kollaps vom 2026-07-09-Checkpoint (Trend-
    # Vorhersage praktisch konstant = Trainings-Klassenpriorisierung, egal welches Beispiel).
    # Dieselbe eigene, hoehere LR bekommen auch alle konfigurierten `boosted_targets`-Koepfe.
    base_lr = train_cfg['learning_rate']
    fusion_lr = base_lr * train_cfg.get('fusion_lr_multiplier', 1.0)
    fusion_param_ids = {id(p) for p in model.fusion.parameters()}
    boosted_head_params = {name: list(model.decoder.heads[model.decoder.step_names.index(name)].parameters())
                            for name in boosted_targets}
    boosted_ids = fusion_param_ids | {id(p) for params in boosted_head_params.values() for p in params}
    other_params = [p for p in model.parameters() if id(p) not in boosted_ids]
    param_groups = [{'params': other_params, 'lr': base_lr}, {'params': model.fusion.parameters(), 'lr': fusion_lr}]
    param_groups += [{'params': params, 'lr': fusion_lr} for params in boosted_head_params.values()]
    optimizer = torch.optim.Adam(param_groups)
    logger.info(f"Optimizer: Basis-LR={base_lr}, Fusion-Attention-LR={fusion_lr}, "
                f"geboostete Koepfe ({', '.join(boosted_targets)})-LR={fusion_lr} "
                f"({train_cfg.get('fusion_lr_multiplier', 1.0)}x)")

    # LR-Warmup NUR fuer die Basis-Parameter: fuer die geboosteten Gruppen (Fusion, geboostete
    # Koepfe) drosselt der Warmup genau die ersten Epochen, in denen sie den Symmetriebruch aus
    # ihrem uniformen/Bias-Fixpunkt heraus schaffen muessen (im 5-Epochen-Diagnoselauf mit sofort
    # voller 10x-LR trat die Spezialisierung schon ab Epoche 4 auf; im vollen Lauf MIT Warmup
    # fuer beide Gruppen blieb sie trotz 19 Epochen komplett kollabiert). Geboostete Gruppen
    # bekommen daher von Anfang an die volle LR, nur die Basis-Parameter werden sanft hochgefahren.
    warmup_epochs = max(1, train_cfg.get('lr_warmup_epochs', 0))
    warmup_lambda = (lambda epoch: min(1.0, (epoch + 1) / warmup_epochs)) if train_cfg.get('lr_warmup_epochs', 0) else (lambda epoch: 1.0)
    constant_lambda = lambda epoch: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [warmup_lambda] + [constant_lambda] * (len(param_groups) - 1))
    grad_clip_norm = train_cfg.get('grad_clip_norm', 0.0)
    diversity_weight = train_cfg.get('diversity_weight', 0.0)
    diversity_weights = {name: diversity_weight for name in boosted_targets} if diversity_weight > 0 else {}

    checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'market_transformer.pt')
    best_checkpoint_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'market_transformer_best.pt')
    checkpoint_every = train_cfg.get('checkpoint_every_epochs', 5)
    patience = train_cfg.get('early_stopping_patience', 5)
    min_delta = train_cfg.get('early_stopping_min_delta', 0.0)

    best_val_acc = -1.0
    best_epoch = 0
    patience_counter = 0
    epoch_diagnostics_log = []
    diagnostics_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'training_diagnostics.json')

    logger.info(f"\nTrainiere auf {device} fuer max. {epochs} Epochen (d_model={d_model}, "
                f"Early-Stopping-Geduld={patience})...")
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        for features, targets in train_loader:
            features = {tf: t.to(device) for tf, t in features.items()}
            targets = {name: t.to(device) for name, t in targets.items()}
            optimizer.zero_grad()
            total_loss, _, _ = model.compute_loss(features, targets, diversity_weights=diversity_weights)
            total_loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            epoch_losses.append(total_loss.item())
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        scheduler.step()

        val_accuracies = evaluate_accuracy(model, val_loader, device)
        avg_val_acc = sum(val_accuracies.values()) / len(val_accuracies)
        last_lrs = scheduler.get_last_lr()
        logger.info(f"  Epoche {epoch + 1}/{epochs}: Trainings-Loss = {avg_loss:.4f}, "
                    f"OOS-Genauigkeit = {avg_val_acc:.1%}, LR(Basis)={last_lrs[0]:.6f}, LR(Boost)={last_lrs[1]:.6f}")

        # Objektive Verlaufs-Kennzahlen fuer die Kollaps-Zuverlaessigkeits-Untersuchung
        # (2026-07-10) -- unabhaengig von avg_val_acc gespeichert, damit man nachtraeglich
        # pruefen kann ob sich kollabierende Laeufe schon frueh (Epoche 1-3) unterscheiden.
        diag = compute_epoch_diagnostics(model, val_loader, device, boosted_targets)
        diag['epoch'] = epoch + 1
        diag['avg_val_acc'] = avg_val_acc
        epoch_diagnostics_log.append(diag)
        with open(diagnostics_path, 'w', encoding='utf-8') as f:
            json.dump(epoch_diagnostics_log, f, indent=2)

        if avg_val_acc > best_val_acc + min_delta:
            best_val_acc = avg_val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            patience_counter += 1

        # Periodisches "letzter Stand"-Zwischenspeichern, damit ein unterbrochener Lauf
        # (z.B. Neustart) nicht alles verliert -- unabhaengig vom Best-Checkpoint.
        if (epoch + 1) % checkpoint_every == 0 or (epoch + 1) == epochs:
            torch.save(model.state_dict(), checkpoint_path)

        if patience_counter >= patience:
            logger.info(f"\nEarly Stopping bei Epoche {epoch + 1} (keine Verbesserung seit {patience} Epochen). "
                        f"Bestes Ergebnis: Epoche {best_epoch} mit {best_val_acc:.1%} OOS-Genauigkeit.")
            break

    logger.info(f"\nLade besten Checkpoint (Epoche {best_epoch}, {best_val_acc:.1%} OOS-Genauigkeit) fuer die finale Auswertung...")
    model.load_state_dict(torch.load(best_checkpoint_path))

    logger.info(f"\nIn-Sample-Auswertung (Training, {len(train_examples)} Beispiele) je Zielvariable:")
    train_metrics = evaluate_with_calibration(model, train_loader, device)
    for name, m in train_metrics.items():
        logger.info(f"  {name}: acc={m['accuracy']:.1%}, top3={m['top3_accuracy']:.1%}, brier={m['brier']:.3f}, ece={m['ece']:.3f}")

    logger.info(f"\nOut-of-Sample-Auswertung (70/30-Split, {len(val_examples)} nie im Training gesehene Beispiele) je Zielvariable:")
    val_metrics = evaluate_with_calibration(model, val_loader, device)
    for name, m in val_metrics.items():
        logger.info(f"  {name}: acc={m['accuracy']:.1%}, top3={m['top3_accuracy']:.1%}, brier={m['brier']:.3f}, ece={m['ece']:.3f}")

    avg_train_acc = sum(m['accuracy'] for m in train_metrics.values()) / len(train_metrics)
    avg_val_acc = sum(m['accuracy'] for m in val_metrics.values()) / len(val_metrics)
    verdict = compute_verdict(avg_train_acc, avg_val_acc)
    logger.info(f"\n70/30 OOS-Verdict (bester Checkpoint): In-Sample {avg_train_acc:.1%} vs Out-of-Sample {avg_val_acc:.1%} "
                f"(Random-Baseline: {RANDOM_BASELINE_ACCURACY:.1%}) -> {verdict}")

    logger.info(f"\nBester Checkpoint gespeichert: {best_checkpoint_path}")
    logger.info(f"Letzter Zwischenstand gespeichert: {checkpoint_path}")

    # Hybrid-Ansatz (2026-07-10, TREE_TARGETS-Umfang spaeter am selben Tag korrigiert): siehe
    # tree_ensemble.py fuer die vollstaendige Begruendung inkl. der Korrektur (Accuracy allein
    # hatte den Kollaps von range/inside_outside_day zunaechst verschleiert). Nur gap_yn/
    # high_first bleiben beim Transformer.
    logger.info(f"\nTrainiere Baum-Ensemble fuer {TREE_TARGETS} (Hybrid-Ansatz)...")
    tree_ensemble = TreeEnsemblePredictor().fit(train_examples, scaler, timeframes)
    tree_ensemble_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'datasets', 'tree_ensemble.pkl')
    tree_ensemble.save(tree_ensemble_path)
    logger.info(f"Baum-Ensemble gespeichert: {tree_ensemble_path}")

    X_train_flat = np.stack([flat_features(ex, scaler, timeframes) for ex in train_examples])
    X_val_flat = np.stack([flat_features(ex, scaler, timeframes) for ex in val_examples])
    tree_train_acc = {t: tree_ensemble.models[t].score(X_train_flat, np.array([ex['target'][t] for ex in train_examples]))
                       for t in TREE_TARGETS}
    tree_val_acc = {t: tree_ensemble.models[t].score(X_val_flat, np.array([ex['target'][t] for ex in val_examples]))
                     for t in TREE_TARGETS}
    logger.info(f"  RandomForest In-Sample: { {k: f'{v:.1%}' for k, v in tree_train_acc.items()} }")
    logger.info(f"  RandomForest Out-of-Sample: { {k: f'{v:.1%}' for k, v in tree_val_acc.items()} }")

    logger.info("\nBeispiel-Vorhersage (Hybrid: Beam Search + RandomForest) fuer das letzte Validierungsbeispiel:")
    last_example = val_examples[-1]
    last_features, last_targets = val_ds[len(val_ds) - 1]
    last_features = {tf: t.unsqueeze(0).to(device) for tf, t in last_features.items()}
    prediction = model.predict_beam(last_features, beam_width=model_cfg['beam_width'])
    tree_prediction = tree_ensemble.predict(last_example, scaler, timeframes)
    for t in TREE_TARGETS:
        prediction[t] = tree_prediction[t]
        prediction['step_probabilities'][t] = tree_prediction['tree_probabilities'][t]
    logger.info(f"  Symbol: {last_example['symbol']}")
    logger.info(f"  Vorhergesagt: { {k: v for k, v in prediction.items() if k in TARGET_NAMES} }")
    logger.info(f"  Tatsaechlich: { {name: last_targets[name].item() for name in TARGET_NAMES} }")
    logger.info(f"  Timeframe-Gewichte: {prediction['timeframe_weights']}")

    # Preis-Koordinaten rekonstruieren: prev_close/ATR aus der Referenz-Tageskerze (reference_time),
    # kategoriale Vorhersage -> High/Body-oben/Body-unten/Low (siehe reconstruct.py). Muss vom
    # richtigen Symbol kommen, nicht nur vom zuletzt geladenen (bei Multi-Symbol-Training).
    daily_df = ohlcv_by_symbol[last_example['symbol']][ds_cfg['reference_timeframe']]
    ref_time = pd.Timestamp(last_example['reference_time'])
    daily_up_to_ref = daily_df.loc[:ref_time]
    prev_close = float(daily_up_to_ref['close'].iloc[-1])
    atr_series = ta.volatility.AverageTrueRange(
        high=daily_up_to_ref['high'], low=daily_up_to_ref['low'], close=daily_up_to_ref['close'],
        window=ds_cfg['target_settings']['atr_window']).average_true_range()
    atr_value = float(atr_series.iloc[-1])

    coords = reconstruct_candle(
        prev_close=prev_close, atr=atr_value, trend=prediction['trend'], range_cat=prediction['range'],
        close_position_cat=prediction['close_position'], upper_wick_cat=prediction['upper_wick'],
        lower_wick_cat=prediction['lower_wick'])
    logger.info(f"\nRekonstruierte Preis-Koordinaten (Anker: Vortages-Close={prev_close:.2f}, ATR={atr_value:.2f}):")
    logger.info(f"  High (Docht oben):    {coords['high']:.2f}")
    logger.info(f"  Body oben:            {coords['body_top']:.2f}")
    logger.info(f"  Body unten:           {coords['body_bottom']:.2f}")
    logger.info(f"  Low (Docht unten):    {coords['low']:.2f}")
    logger.info(f"  (Open={coords['open']:.2f}, Close={coords['close']:.2f}, "
                f"close_position_consistent={coords['close_position_consistent']})")
