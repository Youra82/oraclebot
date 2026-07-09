# src/oraclebot/data/scaler.py
# Kontinuierliche Feature-Standardisierung als Modell-Eingabe.
#
# Ersetzt den K-Means-Tokenizer als Eingabe fuer den Transformer: Diskretisierung
# (Feature-Vektor -> ein Cluster-Index) verliert bei kontinuierlichen Finanzsignalen
# echte Information (z.B. RSI=61.2 vs RSI=61.8 koennten im selben Cluster landen) --
# anders als bei Sprache, wo ein diskretes Vokabular die natuerliche Repraesentation ist.
# Der MarketTokenizer (tokenizer.py) bleibt als eigenstaendiges Analyse-Werkzeug fuer
# Marktregime/Vokabular-Exploration erhalten, wird aber nicht mehr als Modell-Input verwendet.
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler

from oraclebot.data.features import FEATURE_NAMES


class FeatureScaler:
    """Standardisiert (Mittelwert 0, Std 1) den FEATURE_NAMES-Vektor fuer den Transformer-Input."""

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, feature_df) -> 'FeatureScaler':
        return self.fit_array(feature_df[FEATURE_NAMES].values)

    def fit_array(self, X: np.ndarray) -> 'FeatureScaler':
        self.scaler.fit(np.asarray(X, dtype=np.float64))
        self._fitted = True
        return self

    def transform(self, feature_df) -> np.ndarray:
        return self.transform_array(feature_df[FEATURE_NAMES].values)

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Gibt die standardisierten Werte als float32-Array zurueck (batch, len(FEATURE_NAMES))."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler wurde noch nicht gefittet.")
        return self.scaler.transform(np.asarray(X, dtype=np.float64)).astype(np.float32)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'scaler': self.scaler, 'fitted': self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'FeatureScaler':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        obj = cls()
        obj.scaler = state['scaler']
        obj._fitted = state['fitted']
        return obj
