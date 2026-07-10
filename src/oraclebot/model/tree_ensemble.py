# src/oraclebot/model/tree_ensemble.py
# Hybrid-Ansatz (2026-07-10): der Transformer-Decoder kollabiert bei schwach-signalhaltigen
# Zielgroessen (trend, close_position, upper_wick, lower_wick -- alle nahe an oder unter
# sklearn-Baseline-Vorsprung von +4-14pp) in 80% der Trainingslaeufe auf die Trainings-
# Klassenpriorisierung (siehe seed_reliability-Untersuchung). RandomForest auf denselben
# Features hat in KEINEM Test heute kollabiert und war beim trend-Ziel sogar treffsicherer
# (43.6-58.2% vs. Transformer-Bestwert 57.6%, meist deutlich darunter). Grund: Baum-Splits
# erzwingen strukturell eine echte Unterscheidung -- "immer dieselbe Klasse" ist fuer einen
# Entscheidungsbaum kein erreichbarer, bequemer Gradientenabstieg-Fluchtpunkt wie fuer ein
# per Cross-Entropy trainiertes neuronales Netz bei schwachem Signal.
#
# Uebernimmt daher NUR die schwachen Zielgroessen; range/gap_yn/inside_outside_day/high_first
# bleiben beim Transformer (dort kollabiert nichts, siehe OOS-Tests).
import pickle

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from oraclebot.data.features import FEATURE_NAMES

TREE_TARGETS = ['trend', 'close_position', 'upper_wick', 'lower_wick']


def flat_features(example: dict, scaler, timeframes: list) -> np.ndarray:
    """Letzte (aktuellste) Zeile jedes Timeframe-Fensters, skaliert, aneinandergehaengt --

    dieselbe Merkmalsbasis, mit der die sklearn-Baseline-Tests (2026-07-10) das schwache-aber-
    reale Signal fuer trend/close_position gefunden haben. Bewusst NICHT die volle Sequenz
    (der Transformer nutzt die -- und kollabiert trotzdem; ein einfacherer, robusterer Lerner
    auf dem aktuellsten Zustand ist hier der Punkt, nicht mehr Information).
    """
    parts = []
    for tf in timeframes:
        arr = scaler.transform_array(np.array(example[tf], dtype=np.float32))
        parts.append(arr[-1])
    return np.concatenate(parts)


class TreeEnsemblePredictor:
    """Ein RandomForestClassifier pro TREE_TARGETS-Ziel."""

    def __init__(self, n_estimators: int = 300, max_depth: int = 5, random_state: int = 0):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.models = {}

    def fit(self, examples: list, scaler, timeframes: list) -> 'TreeEnsemblePredictor':
        X = np.stack([flat_features(ex, scaler, timeframes) for ex in examples])
        for target in TREE_TARGETS:
            y = np.array([ex['target'][target] for ex in examples])
            model = RandomForestClassifier(
                n_estimators=self.n_estimators, max_depth=self.max_depth,
                class_weight='balanced', random_state=self.random_state,
            )
            model.fit(X, y)
            self.models[target] = model
        return self

    def predict(self, example: dict, scaler, timeframes: list) -> dict:
        """Gibt {target: klasse, target_probabilities: {klasse: p}} fuer jedes TREE_TARGETS-Ziel."""
        x = flat_features(example, scaler, timeframes).reshape(1, -1)
        result = {}
        probabilities = {}
        for target, model in self.models.items():
            proba = model.predict_proba(x)[0]
            class_id = int(np.argmax(proba))
            result[target] = class_id
            probabilities[target] = float(proba[class_id])
        result['tree_probabilities'] = probabilities
        return result

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'TreeEnsemblePredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)
