from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from signature_verification.config import load_config
from signature_verification.datasets import audit_cedar, build_cedar_metadata
from signature_verification.evaluation import calibrate_threshold, evaluate_scores, load_model
from signature_verification.pairs import SignaturePairDataset, generate_pairs
from signature_verification.training import train_experiment
from signature_verification.training.trainer import score_loader


def _write_json(value: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2), encoding="utf-8")


def audit_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build and audit canonical CEDAR metadata")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--skip-decode", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metadata = build_cedar_metadata(
        config["paths"]["data_root"], config["paths"]["metadata"],
        config["data"]["train_ratio"], config["data"]["val_ratio"],
        config["data"]["seed"],
    )
    report = audit_cedar(metadata, decode_images=not args.skip_decode)
    _write_json(report, Path(config["paths"]["metadata"]).with_name("cedar_audit.json"))
    print(json.dumps(report, indent=2))
    if report["writer_overlap"] or report["unreadable_images"]:
        raise SystemExit(1)


def train_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the Siamese signature model")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metadata = pd.read_csv(config["paths"]["metadata"])
    result = train_experiment(
        metadata, config, smoke=args.smoke,
        pretrained=False if args.no_pretrained else None,
    )
    print(json.dumps(result, indent=2))


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Calibrate on validation and evaluate test writers")
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--pairs", type=int, default=4000)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    metadata = pd.read_csv(config["paths"]["metadata"])
    checkpoint = Path(config["paths"]["artifacts"]) / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Baseline checkpoint not found: {checkpoint}. Run `uv run sv-train` first.")
    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_data.get("run_type") != "baseline":
        raise SystemExit("Refusing to evaluate a smoke checkpoint as the final baseline.")
    model, preprocess_cfg, device = load_model(checkpoint)
    output: dict = {}
    threshold = None
    for offset, split in enumerate(("val", "test")):
        pairs = generate_pairs(
            metadata[metadata.split == split], args.pairs,
            config["data"]["positive_fraction"],
            config["data"]["skilled_negative_fraction"],
            config["data"]["seed"] + 100 + offset,
        )
        loader = DataLoader(
            SignaturePairDataset(pairs, preprocess_cfg),
            batch_size=config["training"]["batch_size"], shuffle=False,
            num_workers=config["training"]["num_workers"],
        )
        scores, labels, kinds = score_loader(model, loader, device)
        if split == "val":
            calibration = calibrate_threshold(labels, scores)
            threshold = calibration["threshold"]
            output["calibration"] = calibration
        output[split] = evaluate_scores(labels, scores, threshold, kinds)
    artifact_dir = Path(config["paths"]["artifacts"])
    _write_json(output, artifact_dir / "evaluation.json")
    _write_json({"threshold": threshold, "method": "validation_eer"}, artifact_dir / "thresholds.json")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    audit_main()
