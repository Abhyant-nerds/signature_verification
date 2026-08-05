from pathlib import Path

from signature_verification.datasets import build_cedar_metadata
from signature_verification.pairs import generate_pairs


def test_pair_generation_has_valid_classes():
    metadata = build_cedar_metadata(Path("data/raw/cedar/signatures"))
    pairs = generate_pairs(metadata[metadata.split == "train"], 100, seed=7)
    assert len(pairs) == 100
    assert sum(pair.label for pair in pairs) == 50
    assert {pair.negative_type for pair in pairs if pair.label == 0} <= {"SKILLED", "RANDOM"}
    assert all(pair.left_path != pair.right_path for pair in pairs)
