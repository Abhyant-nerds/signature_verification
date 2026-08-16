from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from signature_verification.preprocessing import PreprocessConfig, preprocess_signature


@dataclass(frozen=True)
class Pair:
    left_path: str
    right_path: str
    label: int
    negative_type: str


def generate_pairs(
    metadata: pd.DataFrame,
    count: int,
    positive_fraction: float = 0.5,
    skilled_negative_fraction: float = 0.7,
    seed: int = 42,
) -> list[Pair]:
    """Generate 1-labelled matches and 0-labelled skilled/random mismatches."""
    if not 0 < positive_fraction < 1:
        raise ValueError("positive_fraction must be between zero and one")
    rng = np.random.default_rng(seed)
    genuine = metadata[metadata.sample_type == "GENUINE"]
    forgeries = metadata[metadata.sample_type == "FORGERY"]
    writers = sorted(genuine.writer_id.unique())
    if len(writers) < 2:
        raise ValueError("At least two writers are needed to generate random negatives")
    pairs: list[Pair] = []
    n_positive = round(count * positive_fraction)
    for _ in range(n_positive):
        writer = rng.choice(writers)
        samples = genuine[genuine.writer_id == writer].image_path.to_numpy()
        left, right = rng.choice(samples, size=2, replace=False)
        pairs.append(Pair(str(left), str(right), 1, "NONE"))
    for _ in range(count - n_positive):
        writer = rng.choice(writers)
        left = rng.choice(genuine[genuine.writer_id == writer].image_path.to_numpy())
        available_forgery = forgeries[forgeries.target_writer_id == writer]
        if not available_forgery.empty and rng.random() < skilled_negative_fraction:
            right = rng.choice(available_forgery.image_path.to_numpy())
            kind = "SKILLED"
        else:
            other_writer = rng.choice([value for value in writers if value != writer])
            right = rng.choice(genuine[genuine.writer_id == other_writer].image_path.to_numpy())
            kind = "RANDOM"
        pairs.append(Pair(str(left), str(right), 0, kind))
    rng.shuffle(pairs)
    return pairs


class SignaturePairDataset(Dataset):
    def __init__(self, pairs: list[Pair], preprocess_config: PreprocessConfig | None = None):
        self.pairs = pairs
        self.preprocess_config = preprocess_config or PreprocessConfig()

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        left, _, _ = preprocess_signature(pair.left_path, self.preprocess_config)
        right, _, _ = preprocess_signature(pair.right_path, self.preprocess_config)
        return left, right, np.float32(pair.label), pair.negative_type
