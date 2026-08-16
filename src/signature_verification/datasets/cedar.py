from __future__ import annotations

import hashlib
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

GENUINE_RE = re.compile(r"^original_(\d+)_(\d+)\.png$", re.IGNORECASE)
FORGERY_RE = re.compile(r"^forgeries_(\d+)_(\d+)\.png$", re.IGNORECASE)
REQUIRED_COLUMNS = [
    "sample_id", "image_path", "writer_id", "target_writer_id", "sample_type",
    "forgery_type", "dataset", "split",
]


def _image_rows(folder: Path, pattern: re.Pattern[str], sample_type: str) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(folder.glob("*.png")):
        match = pattern.match(path.name)
        if not match:
            continue
        writer, sample = (int(value) for value in match.groups())
        rows.append(
            {
                "sample_id": f"CEDAR_{writer:02d}_{sample_type[0]}{sample:02d}",
                "image_path": path.resolve().as_posix(),
                "writer_id": f"CEDAR_{writer:02d}",
                "target_writer_id": f"CEDAR_{writer:02d}",
                "sample_type": sample_type,
                "forgery_type": "NONE" if sample_type == "GENUINE" else "SKILLED",
                "dataset": "CEDAR",
                "split": "",
            }
        )
    return rows


def split_writers(
    metadata: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign complete writers to deterministic train/validation/test splits."""
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must be positive and sum to less than one")
    writers = np.array(sorted(metadata["writer_id"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(writers)
    n_train = round(len(writers) * train_ratio)
    n_val = round(len(writers) * val_ratio)
    assignments = {
        **{writer: "train" for writer in writers[:n_train]},
        **{writer: "val" for writer in writers[n_train : n_train + n_val]},
        **{writer: "test" for writer in writers[n_train + n_val :]},
    }
    result = metadata.copy()
    result["split"] = result["writer_id"].map(assignments)
    return result.sort_values(["split", "writer_id", "sample_id"]).reset_index(drop=True)


def build_cedar_metadata(
    root: str | Path,
    output_path: str | Path | None = None,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    root = Path(root)
    genuine_dir, forgery_dir = root / "full_org", root / "full_forg"
    if not genuine_dir.is_dir() or not forgery_dir.is_dir():
        raise FileNotFoundError(f"Expected full_org and full_forg under {root}")
    rows = _image_rows(genuine_dir, GENUINE_RE, "GENUINE")
    rows += _image_rows(forgery_dir, FORGERY_RE, "FORGERY")
    metadata = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if metadata.empty:
        raise ValueError(f"No CEDAR signature PNGs found under {root}")
    if metadata["sample_id"].duplicated().any():
        raise ValueError("Duplicate CEDAR sample IDs found")
    metadata = split_writers(metadata, train_ratio, val_ratio, seed)
    if output_path:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        metadata.to_csv(output, index=False)
    return metadata


def audit_cedar(metadata: pd.DataFrame, decode_images: bool = True) -> dict:
    split_writers_map = {
        split: set(group["writer_id"]) for split, group in metadata.groupby("split")
    }
    overlap = any(
        split_writers_map[a] & split_writers_map[b]
        for a, b in (("train", "val"), ("train", "test"), ("val", "test"))
    )
    unreadable: list[str] = []
    dimensions: list[tuple[int, int]] = []
    if decode_images:
        for value in metadata["image_path"]:
            image = cv2.imread(value, cv2.IMREAD_GRAYSCALE)
            if image is None:
                unreadable.append(value)
            else:
                dimensions.append((int(image.shape[1]), int(image.shape[0])))
    counts = metadata.groupby(["split", "sample_type"]).size().unstack(fill_value=0)
    return {
        "samples": len(metadata),
        "writers": int(metadata["writer_id"].nunique()),
        "writers_by_split": {
            key: len(value) for key, value in split_writers_map.items()
        },
        "samples_by_split_and_type": counts.to_dict(orient="index"),
        "writer_overlap": overlap,
        "unreadable_images": unreadable,
        "min_width": min((x[0] for x in dimensions), default=None),
        "max_width": max((x[0] for x in dimensions), default=None),
        "min_height": min((x[1] for x in dimensions), default=None),
        "max_height": max((x[1] for x in dimensions), default=None),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
