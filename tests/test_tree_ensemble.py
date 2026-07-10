import numpy as np

from oraclebot.data.targets import TARGET_NAMES
from oraclebot.model.transformer import TARGET_CARDINALITIES
from oraclebot.model.tree_ensemble import TREE_TARGETS, TreeEnsemblePredictor, flat_features

TIMEFRAMES = ['1d', '4h']
WINDOW_SIZES = {'1d': 5, '4h': 10}
N_FEATURES = 21


class DummyScaler:
    def transform_array(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)


def make_examples(n=40, seed=0):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        ex = {tf: rng.normal(size=(WINDOW_SIZES[tf], N_FEATURES)).tolist() for tf in TIMEFRAMES}
        ex['target'] = {name: int(rng.integers(0, TARGET_CARDINALITIES[name])) for name in TARGET_NAMES}
        examples.append(ex)
    return examples


def test_flat_features_uses_last_row_per_timeframe():
    scaler = DummyScaler()
    ex = {tf: np.arange(WINDOW_SIZES[tf] * N_FEATURES).reshape(WINDOW_SIZES[tf], N_FEATURES).tolist() for tf in TIMEFRAMES}
    features = flat_features(ex, scaler, TIMEFRAMES)
    assert features.shape == (len(TIMEFRAMES) * N_FEATURES,)
    expected_1d_last_row = np.array(ex['1d'][-1], dtype=np.float32)
    np.testing.assert_array_equal(features[:N_FEATURES], expected_1d_last_row)


def test_fit_and_predict_covers_all_tree_targets():
    scaler = DummyScaler()
    examples = make_examples()
    predictor = TreeEnsemblePredictor(n_estimators=10, max_depth=3).fit(examples, scaler, TIMEFRAMES)

    prediction = predictor.predict(examples[0], scaler, TIMEFRAMES)
    for target in TREE_TARGETS:
        assert target in prediction
        assert 0 <= prediction[target] < TARGET_CARDINALITIES[target]
        assert target in prediction['tree_probabilities']
        assert 0.0 <= prediction['tree_probabilities'][target] <= 1.0


def test_save_and_load_roundtrip(tmp_path):
    scaler = DummyScaler()
    examples = make_examples()
    predictor = TreeEnsemblePredictor(n_estimators=10, max_depth=3).fit(examples, scaler, TIMEFRAMES)

    path = str(tmp_path / 'tree_ensemble.pkl')
    predictor.save(path)
    loaded = TreeEnsemblePredictor.load(path)

    original = predictor.predict(examples[0], scaler, TIMEFRAMES)
    restored = loaded.predict(examples[0], scaler, TIMEFRAMES)
    for target in TREE_TARGETS:
        assert original[target] == restored[target]
