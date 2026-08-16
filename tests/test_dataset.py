from pathlib import Path

from signature_verification.datasets import build_cedar_metadata

DATA_ROOT = Path("data/raw/cedar/signatures")


def test_cedar_adapter_counts_and_labels():
    metadata = build_cedar_metadata(DATA_ROOT)
    assert len(metadata) == 2640
    assert metadata.writer_id.nunique() == 55
    assert (metadata.sample_type == "GENUINE").sum() == 1320
    assert (metadata.sample_type == "FORGERY").sum() == 1320


def test_writer_splits_do_not_overlap():
    metadata = build_cedar_metadata(DATA_ROOT)
    groups = {name: set(part.writer_id) for name, part in metadata.groupby("split")}
    assert groups["train"].isdisjoint(groups["val"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["val"].isdisjoint(groups["test"])
