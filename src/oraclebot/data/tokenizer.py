# src/oraclebot/data/tokenizer.py
# Tokenisierung: Markt-Token-Feature-Vektor -> diskreter Vokabular-Index (Clustering statt Vorgabe von Hand).
#
# Wir clustern direkt im interpretierbaren 15-dim Feature-Raum aus features.py, statt zuerst
# ein gelerntes neuronales Embedding zu trainieren. Das entspricht Schritt 5 der Spezifikation,
# nur dass der "Embedding-Schritt" hier trivial ist (die Features SIND bereits die Repraesentation).
# Ein gelerntes Embedding kommt erst mit dem Transformer (nn.Embedding(vocab_size, dim)) in Schritt 6 --
# das Clustering hier liefert nur die diskreten Token-IDs, auf denen dieses Embedding dann aufsetzt.
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from oraclebot.data.features import FEATURE_NAMES


class MarketTokenizer:
    """Clustert Markt-Token-Feature-Vektoren zu einem diskreten Vokabular (K-Means).

    J = sum_i ||x_i - mu_{k(i)}||^2  (K-Means-Zielfunktion, siehe Spezifikation Schritt 5).
    """

    def __init__(self, n_tokens: int = 512, random_state: int = 42):
        self.n_tokens = n_tokens
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_tokens, random_state=random_state, n_init=10)
        self._fitted = False

    def fit(self, feature_df: pd.DataFrame) -> 'MarketTokenizer':
        """Fittet Scaler + K-Means auf den FEATURE_NAMES-Spalten eines Feature-DataFrames."""
        return self.fit_array(feature_df[FEATURE_NAMES].values)

    def transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Ordnet jeder Kerze den Index ihres naechstgelegenen Vokabular-Tokens zu."""
        return self.transform_array(feature_df[FEATURE_NAMES].values)

    def fit_transform(self, feature_df: pd.DataFrame) -> np.ndarray:
        self.fit(feature_df)
        return self.transform(feature_df)

    def fit_array(self, X: np.ndarray) -> 'MarketTokenizer':
        """Wie fit(), nimmt aber direkt eine (N, len(FEATURE_NAMES))-Matrix statt eines DataFrames."""
        # float64 erzwingen: KMeans verlangt intern denselben Dtype bei fit() und predict();
        # float32-Input (z.B. aus PyTorch-Tensoren) wuerde sonst zu einem Cython-Buffer-Mismatch fuehren.
        X_scaled = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        self.kmeans.fit(X_scaled)
        self._fitted = True
        return self

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        """Wie transform(), nimmt aber direkt eine (N, len(FEATURE_NAMES))-Matrix statt eines DataFrames.

        Wird beim Modell-Training benutzt: die Dataset-Fenster aus dataset.py sind bereits
        rohe Feature-Arrays (keine DataFrames mehr), da sie aus JSONL geladen werden.
        """
        if not self._fitted:
            raise RuntimeError("MarketTokenizer wurde noch nicht gefittet.")
        X_scaled = self.scaler.transform(np.asarray(X, dtype=np.float64))
        return self.kmeans.predict(X_scaled)

    def describe_token(self, token_id: int) -> dict:
        """Beschreibt ein Token durch die (rueck-skalierten) Feature-Werte seines Cluster-Zentrums."""
        if not self._fitted:
            raise RuntimeError("MarketTokenizer wurde noch nicht gefittet.")
        centroid_scaled = self.kmeans.cluster_centers_[token_id].reshape(1, -1)
        centroid = self.scaler.inverse_transform(centroid_scaled)[0]
        values = dict(zip(FEATURE_NAMES, centroid))

        trend_label = 'bullish' if values['trend_state'] > 0.5 else 'bearish' if values['trend_state'] < -0.5 else 'neutral'
        vola_label = 'hoch' if values['atr_range'] > 1.3 else 'niedrig' if values['atr_range'] < 0.7 else 'normal'
        momentum_label = 'steigend' if values['momentum'] > 0.2 else 'fallend' if values['momentum'] < -0.2 else 'neutral'

        return {
            'token_id': token_id,
            'label': f"Trend {trend_label} | Vola {vola_label} ({values['atr_range']:.2f}x ATR) | Momentum {momentum_label}",
            'centroid': values,
        }

    def vocabulary_summary(self) -> pd.DataFrame:
        """Listet alle Tokens mit ihrer Nutzungshaeufigkeit (auf den Fit-Daten) und Beschreibung."""
        if not self._fitted:
            raise RuntimeError("MarketTokenizer wurde noch nicht gefittet.")
        labels, counts = np.unique(self.kmeans.labels_, return_counts=True)
        rows = []
        for token_id, count in zip(labels, counts):
            desc = self.describe_token(int(token_id))
            rows.append({'token_id': int(token_id), 'count': int(count), 'label': desc['label']})
        return pd.DataFrame(rows).sort_values('count', ascending=False).reset_index(drop=True)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'n_tokens': self.n_tokens, 'random_state': self.random_state,
                         'scaler': self.scaler, 'kmeans': self.kmeans, 'fitted': self._fitted}, f)

    @classmethod
    def load(cls, path: str) -> 'MarketTokenizer':
        with open(path, 'rb') as f:
            state = pickle.load(f)
        tokenizer = cls(n_tokens=state['n_tokens'], random_state=state['random_state'])
        tokenizer.scaler = state['scaler']
        tokenizer.kmeans = state['kmeans']
        tokenizer._fitted = state['fitted']
        return tokenizer
