import numpy as np

from oraclebot.data.features import FEATURE_NAMES
from oraclebot.data.scaler import FeatureScaler
from oraclebot.model.barrier_model import BarrierPredictor


def make_separable_data(n=200, seed=0):
    """Feature 0 traegt das gesamte Signal (Klasse = Vorzeichen von Feature 0 + Rauschen),
    alle anderen Features sind reines Rauschen -- ein lernbares, aber nicht triviales Problem."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, len(FEATURE_NAMES))).astype(np.float32)
    y = (X[:, 0] + rng.normal(0, 0.3, n) > 0).astype(int)
    return X, y


def test_fit_and_predict_proba_shapes():
    X, y = make_separable_data()
    scaler = FeatureScaler().fit_array(X)
    predictor = BarrierPredictor(max_depth=3).fit(X, y, scaler)
    proba = predictor.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_learns_separable_signal_above_random_baseline():
    X_train, y_train = make_separable_data(n=400, seed=1)
    X_test, y_test = make_separable_data(n=200, seed=2)
    scaler = FeatureScaler().fit_array(X_train)
    predictor = BarrierPredictor(max_depth=3).fit(X_train, y_train, scaler)
    acc = predictor.score(X_test, y_test)
    assert acc > 0.7  # deutlich ueber Zufall (50%), da Feature 0 stark separierend ist


def test_predict_one_returns_class_and_confidence_between_half_and_one():
    X, y = make_separable_data()
    scaler = FeatureScaler().fit_array(X)
    predictor = BarrierPredictor(max_depth=3).fit(X, y, scaler)
    cls, conf = predictor.predict_one(X[0])
    assert cls in (0, 1)
    assert 0.5 <= conf <= 1.0


def test_save_and_load_roundtrip_preserves_predictions(tmp_path):
    X, y = make_separable_data()
    scaler = FeatureScaler().fit_array(X)
    predictor = BarrierPredictor(max_depth=3).fit(X, y, scaler)
    path = str(tmp_path / 'model.pkl')
    predictor.save(path)

    loaded = BarrierPredictor.load(path)
    np.testing.assert_array_equal(predictor.predict_proba(X), loaded.predict_proba(X))
