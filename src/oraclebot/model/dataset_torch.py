# src/oraclebot/model/dataset_torch.py
# Bruecke zwischen den rohen JSONL-Trainingsbeispielen (dataset.py) und PyTorch-Tensoren:
# standardisiert jedes Timeframe-Fenster mit einem gefitteten FeatureScaler (kontinuierlich,
# kein Tokenizer/Clustering -- siehe transformer.py fuer die Begruendung).
import numpy as np
import torch
from torch.utils.data import Dataset

from oraclebot.data.targets import TARGET_NAMES


class ContinuousMarketDataset(Dataset):
    """Wandelt Trainingsbeispiele (rohe Feature-Fenster pro Timeframe + Ziel-Dict) in

    standardisierte Float-Tensoren um, passend fuer MarketTransformer.
    """

    def __init__(self, examples: list, scaler, timeframes: list):
        self.examples = examples
        self.scaler = scaler
        self.timeframes = timeframes

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        example = self.examples[idx]
        features = {}
        for tf in self.timeframes:
            window = np.array(example[tf], dtype=np.float32)  # (seq_len, n_features)
            scaled = self.scaler.transform_array(window)
            features[tf] = torch.tensor(scaled, dtype=torch.float32)
        targets = {name: torch.tensor(int(example['target'][name]), dtype=torch.long) for name in TARGET_NAMES}
        return features, targets


def collate_fn(batch):
    features_batch, targets_batch = zip(*batch)
    timeframes = features_batch[0].keys()
    collated_features = {tf: torch.stack([item[tf] for item in features_batch]) for tf in timeframes}
    collated_targets = {name: torch.stack([item[name] for item in targets_batch]) for name in TARGET_NAMES}
    return collated_features, collated_targets
