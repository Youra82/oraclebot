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
# KORREKTUR (2026-07-10, spaeter am selben Tag): urspruenglich nur trend/close_position/
# upper_wick/lower_wick uebernommen, weil range/gap_yn/inside_outside_day/high_first anhand
# ihrer Accuracy-Zahlen (44-59%, 72-77%) als "kollabiert nicht" eingestuft wurden -- OHNE die
# Vorhersage-VERTEILUNG direkt zu pruefen, wie bei trend. Ein User-Vergleich der Chart-Kerzen
# mit den echten Kerzen deckte auf: range war zu 131/132 auf einem einzigen Bucket kollabiert
# (Accuracy sah trotzdem "brauchbar" aus, weil das die zweithaeufigste echte Klasse war) und
# inside_outside_day war zu 132/132 auf der Mehrheitsklasse kollabiert (die "75.8% Accuracy"
# war schlicht der Mehrheitsklassen-Anteil). Lehre: Accuracy allein erkennt Kollaps NICHT
# zuverlaessig -- nur ein direkter Verteilungs-Check tut das. gap_yn ist eine echte Ausnahme
# (die echten Labels sind in dieser Datenmenge selbst zu 100% eine Klasse -- "immer 0" ist dort
# die korrekte Antwort, kein Kollaps). high_first zeigte als einziges Ziel echte, mit der
# Realitaet uebereinstimmende Diversitaet und bleibt beim Transformer.
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from oraclebot.data.features import FEATURE_NAMES

TREE_TARGETS = ['trend', 'range', 'close_position', 'upper_wick', 'lower_wick', 'inside_outside_day']

# HistGradientBoosting zeigt fuer 'trend' einen robusteren Worst-Case als RandomForest bei
# identischem Mittelwert -- bestaetigt in zwei unabhaengigen Tests (2026-07-13): Split-Ratio
# (70/30/60/40/50/50 x 10 Seeds: worst-case 57.5% vs. RF 51.7%) und Zeit-Walk-Forward (5
# expandierende Zeitfenster x 5 Seeds: worst-case 50.6% vs. RF 46.1%), zusaetzlich vollstaendig
# deterministisch (0 Varianz ueber Seeds). Fuer die uebrigen Ziele nicht getestet -> dort bleibt RF.
HISTGBM_TARGETS = {'trend'}

# max_depth-Override je Ziel: Walk-Forward-Test (2026-07-24, 3 chronologische Testfenster auf
# BTC/USDT:USDT) zeigte fuer 'trend' einen robusten Gewinn von depth=5 auf depth=3 -- besser
# oder gleich gut in JEDEM der 3 Fenster (Mittel 56.2%->59.0%, Worst-Case 53.7%->56.0%), nicht
# nur im Durchschnitt. Plausibel bei nur ~130-400 Trainingsbeispielen pro Fenster: tiefere
# Baeume (depth=5+) neigen eher zum Auswendiglernen statt Verallgemeinern. Nur fuer 'trend'
# getestet -- die uebrigen TREE_TARGETS bleiben bei self.max_depth (5), bis sie separat
# validiert werden.
TARGET_MAX_DEPTH = {'trend': 3}


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
            depth = TARGET_MAX_DEPTH.get(target, self.max_depth)
            if target in HISTGBM_TARGETS:
                model = HistGradientBoostingClassifier(
                    max_depth=depth, max_iter=100,
                    class_weight='balanced', random_state=self.random_state,
                )
            else:
                model = RandomForestClassifier(
                    n_estimators=self.n_estimators, max_depth=depth,
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
