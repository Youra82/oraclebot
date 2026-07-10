import numpy as np
import torch

from oraclebot.data.targets import TARGET_NAMES
from oraclebot.model.dataset_torch import ContinuousMarketDataset, collate_fn
from oraclebot.model.transformer import TARGET_CARDINALITIES, MarketTransformer

TIMEFRAMES = ['1d', '4h']
WINDOW_SIZES = {'1d': 5, '4h': 10}
N_FEATURES = 19


class DummyScaler:
    """Deterministisches Fake statt eines echten FeatureScaler, fuer schnelle Modell-Tests."""

    def transform_array(self, X: np.ndarray) -> np.ndarray:
        return X.astype(np.float32)


def make_model(d_model=32, nhead=4, num_encoder_layers=2):
    return MarketTransformer(
        n_features=N_FEATURES, timeframes=TIMEFRAMES, window_sizes=WINDOW_SIZES,
        d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers,
        dim_feedforward=64, dropout=0.0,
    )


def make_batch(batch_size=4):
    features = {
        tf: torch.randn(batch_size, WINDOW_SIZES[tf], N_FEATURES)
        for tf in TIMEFRAMES
    }
    targets = {name: torch.randint(0, TARGET_CARDINALITIES[name], (batch_size,)) for name in TARGET_NAMES}
    return features, targets


def make_examples(n=6, seed=0):
    rng = np.random.default_rng(seed)
    examples = []
    for _ in range(n):
        ex = {tf: rng.normal(size=(WINDOW_SIZES[tf], N_FEATURES)).tolist() for tf in TIMEFRAMES}
        ex['target'] = {name: int(rng.integers(0, TARGET_CARDINALITIES[name])) for name in TARGET_NAMES}
        examples.append(ex)
    return examples


def test_forward_pass_shapes():
    model = make_model()
    features, _ = make_batch(batch_size=4)
    logits, timeframe_weights = model(features)

    for name in TARGET_NAMES:
        assert logits[name].shape == (4, TARGET_CARDINALITIES[name])
    assert timeframe_weights.shape == (4, len(TIMEFRAMES))
    # Softmax-Attention-Gewichte: nicht-negativ, und zusammen mit dem (nicht zurueckgegebenen)
    # Selbst-Attention-Anteil des CLS-Tokens summieren sie sich zu 1 -- hier also < 1.
    assert (timeframe_weights >= 0).all()
    assert (timeframe_weights.sum(dim=1) <= 1.0 + 1e-4).all()


def test_compute_loss_is_finite_and_positive():
    model = make_model()
    features, targets = make_batch(batch_size=4)
    total_loss, per_head_losses, _ = model.compute_loss(features, targets)

    assert torch.isfinite(total_loss)
    assert total_loss.item() > 0
    assert set(per_head_losses.keys()) == set(TARGET_NAMES)


def test_trend_diversity_weight_changes_loss_but_not_per_head_losses():
    """Der Diversitaets-Term wirkt nur auf `total`, nicht auf die einzelnen Zielgroessen-Losses

    (die bleiben reines Cross-Entropy, unveraendert) -- und aendert `total` gegenueber
    Gewicht=0 (bestaetigt, dass der Term ueberhaupt greift).
    """
    model = make_model()
    features, targets = make_batch(batch_size=8)

    torch.manual_seed(0)
    total_plain, losses_plain, _ = model.compute_loss(features, targets, trend_diversity_weight=0.0)
    torch.manual_seed(0)
    total_diverse, losses_diverse, _ = model.compute_loss(features, targets, trend_diversity_weight=0.5)

    assert torch.isfinite(total_diverse)
    assert not torch.isclose(total_plain, total_diverse)
    for name in TARGET_NAMES:
        assert torch.isclose(losses_plain[name], losses_diverse[name])


def test_training_step_reduces_loss_on_tiny_batch():
    torch.manual_seed(0)
    model = make_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features, targets = make_batch(batch_size=8)

    losses = []
    for _ in range(30):
        optimizer.zero_grad()
        total_loss, _, _ = model.compute_loss(features, targets)
        total_loss.backward()
        optimizer.step()
        losses.append(total_loss.item())

    assert losses[-1] < losses[0]


def test_predict_beam_returns_valid_categories():
    model = make_model()
    features = {tf: torch.randn(1, WINDOW_SIZES[tf], N_FEATURES) for tf in TIMEFRAMES}

    result = model.predict_beam(features, beam_width=3)

    for name in TARGET_NAMES:
        assert 0 <= result[name] < TARGET_CARDINALITIES[name]
    assert set(result['timeframe_weights'].keys()) == set(TIMEFRAMES)
    assert result['log_prob'] <= 0.0  # Summe von Log-Wahrscheinlichkeiten <= 0
    assert set(result['step_probabilities'].keys()) == set(TARGET_NAMES)
    assert all(0.0 <= p <= 1.0 for p in result['step_probabilities'].values())


def test_beam_width_one_matches_greedy_decoding():
    model = make_model()
    features = {tf: torch.randn(1, WINDOW_SIZES[tf], N_FEATURES) for tf in TIMEFRAMES}

    beam_result = model.predict_beam(features, beam_width=1)
    logits, _ = model.forward(features, targets=None)
    greedy_result = {name: logits[name].argmax(dim=-1).item() for name in TARGET_NAMES}

    for name in TARGET_NAMES:
        assert beam_result[name] == greedy_result[name]


def test_continuous_dataset_and_collate_produce_correct_shapes():
    examples = make_examples(n=6)
    dataset = ContinuousMarketDataset(examples, DummyScaler(), TIMEFRAMES)
    batch = [dataset[i] for i in range(len(dataset))]
    features, targets = collate_fn(batch)

    for tf in TIMEFRAMES:
        assert features[tf].shape == (6, WINDOW_SIZES[tf], N_FEATURES)
    for name in TARGET_NAMES:
        assert targets[name].shape == (6,)


def test_end_to_end_continuous_dataset_through_model():
    examples = make_examples(n=6)
    dataset = ContinuousMarketDataset(examples, DummyScaler(), TIMEFRAMES)
    features, targets = collate_fn([dataset[i] for i in range(len(dataset))])

    model = make_model()
    total_loss, _, _ = model.compute_loss(features, targets)
    assert torch.isfinite(total_loss)
