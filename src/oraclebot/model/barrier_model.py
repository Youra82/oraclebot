# src/oraclebot/model/barrier_model.py
# Vorhersage-Modell fuer das Barriere-Ziel (barrier_targets.py): ein einzelnes
# HistGradientBoostingClassifier (depth=3) auf den reinen Referenzkerzen-Features -- kein
# Multi-Timeframe-Fenster wie beim Transformer/tree_ensemble.py noetig, da hier ausschliesslich
# die aktuellste Kerze der gewaehlten Referenz-Zeitebene (Standard 4h) als Input dient.
#
# Architektur-Wahl (depth=3, HistGBM, kein Ensemble): direkt aus der Recherche vom 24.07.2026
# uebernommen -- max_depth 2-6 lieferte fast identische Genauigkeit (66.0-66.9%), depth=3 als
# Mittelweg gewaehlt. Ein heterogenes Ensemble (HistGBM+RF+LogReg) verbesserte beim taeglichen
# trend-Ziel die rohe Genauigkeit, aber NICHT das Handelsergebnis -- deshalb hier bewusst bei
# einem einzelnen Modell belassen, bis eine eigene Untersuchung fuer DIESES Ziel etwas anderes
# zeigt.
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from oraclebot.data.features import FEATURE_NAMES


class BarrierPredictor:
    """Sagt vorher, ob ausgehend von einer Referenzkerze zuerst die obere oder untere
    Preis-Barriere erreicht wird (siehe barrier_targets.py fuer die Zieldefinition)."""

    def __init__(self, max_depth: int = 3, max_iter: int = 100, random_state: int = 0):
        self.max_depth = max_depth
        self.max_iter = max_iter
        self.random_state = random_state
        self.model = None
        self.scaler = None

    def fit(self, X: np.ndarray, y: np.ndarray, scaler) -> 'BarrierPredictor':
        """X: unskalierte Feature-Matrix (n, len(FEATURE_NAMES)). scaler: bereits auf den
        Trainingsdaten gefitteter FeatureScaler (wird mitgespeichert, damit predict_proba/
        predict_one konsistent dieselbe Skalierung verwenden)."""
        self.scaler = scaler
        self.model = HistGradientBoostingClassifier(
            max_depth=self.max_depth, max_iter=self.max_iter,
            class_weight='balanced', random_state=self.random_state)
        self.model.fit(self.scaler.transform_array(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.scaler.transform_array(X))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return self.model.score(self.scaler.transform_array(X), y)

    def predict_one(self, feature_row) -> tuple:
        """feature_row: Feature-Vektor (Laenge len(FEATURE_NAMES)) EINER Kerze, unskaliert.
        Gibt (vorhergesagte_klasse, confidence) zurueck."""
        x = np.array(feature_row, dtype=np.float32).reshape(1, len(FEATURE_NAMES))
        proba = self.predict_proba(x)[0]
        cls = int(np.argmax(proba))
        return cls, float(proba[cls])

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> 'BarrierPredictor':
        with open(path, 'rb') as f:
            return pickle.load(f)
