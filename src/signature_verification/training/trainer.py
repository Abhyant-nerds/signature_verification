from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from signature_verification.losses import ContrastiveLoss
from signature_verification.models import SiameseNetwork
from signature_verification.pairs import SignaturePairDataset, generate_pairs
from signature_verification.preprocessing import PreprocessConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _preprocess_config(config: dict) -> PreprocessConfig:
    return PreprocessConfig(**config.get("preprocessing", {}))


@torch.no_grad()
def score_loader(
    model, loader, device, description: str | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    model.eval()
    scores, labels, kinds = [], [], []
    batches = tqdm(
        loader,
        desc=description,
        unit="batch",
        leave=False,
        dynamic_ncols=True,
        disable=description is None,
    )
    for left, right, label, kind in batches:
        left, right = left.to(device), right.to(device)
        first, second = model(left, right)
        scores.extend(F.cosine_similarity(first, second).cpu().numpy().tolist())
        labels.extend(label.numpy().tolist())
        kinds.extend(kind)
    return np.asarray(scores), np.asarray(labels, dtype=int), kinds


def train_experiment(
    metadata: pd.DataFrame,
    config: dict,
    smoke: bool = False,
    pretrained: bool | None = None,
) -> dict:
    training = config["training"]
    data_cfg = config["data"]
    model_cfg = config["model"]
    seed = int(training.get("seed", 42))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    artifact_dir = Path(config["paths"]["artifacts"])
    if smoke:
        artifact_dir = artifact_dir.parent / "smoke"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    pair_count = min(128, data_cfg["pairs_per_epoch"]) if smoke else data_cfg["pairs_per_epoch"]
    epochs = 1 if smoke else training["epochs"]
    train_pairs = generate_pairs(
        metadata[metadata.split == "train"], pair_count,
        data_cfg["positive_fraction"], data_cfg["skilled_negative_fraction"], seed,
    )
    val_pairs = generate_pairs(
        metadata[metadata.split == "val"], max(64, pair_count // 5),
        data_cfg["positive_fraction"], data_cfg["skilled_negative_fraction"], seed + 1,
    )
    pp_cfg = _preprocess_config(config)
    batch_size = min(4, training["batch_size"]) if smoke else training["batch_size"]
    train_loader = DataLoader(
        SignaturePairDataset(train_pairs, pp_cfg), batch_size=batch_size, shuffle=True,
        num_workers=training["num_workers"], pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        SignaturePairDataset(val_pairs, pp_cfg), batch_size=batch_size, shuffle=False,
        num_workers=training["num_workers"], pin_memory=device.type == "cuda",
    )
    use_pretrained = model_cfg["pretrained"] if pretrained is None else pretrained
    model = SiameseNetwork(
        model_cfg["encoder"], model_cfg["embedding_dim"], use_pretrained
    ).to(device)
    criterion = ContrastiveLoss(training["margin"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=training["learning_rate"], weight_decay=training["weight_decay"]
    )
    scaler = torch.amp.GradScaler("cuda", enabled=training["mixed_precision"] and device.type == "cuda")
    history, best_loss, stale = [], float("inf"), 0
    started = time.perf_counter()
    epoch_progress = tqdm(
        range(1, epochs + 1), desc="Training", unit="epoch", dynamic_ncols=True
    )
    for epoch in epoch_progress:
        model.train()
        losses = []
        train_progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        for left, right, label, _ in train_progress:
            left, right, label = left.to(device), right.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                first, second = model(left, right)
                loss = criterion(first, second, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), training["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            progress_values = {
                "loss": f"{losses[-1]:.4f}",
                "avg": f"{np.mean(losses):.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}",
            }
            if device.type == "cuda":
                progress_values["gpu"] = f"{torch.cuda.memory_allocated() / 1024**2:.0f}MiB"
            train_progress.set_postfix(progress_values)
        scores, labels, _ = score_loader(
            model, val_loader, device, description=f"Validate {epoch}/{epochs}"
        )
        distances = 1.0 - scores
        val_loss = float(np.mean(0.5 * (
            labels * distances**2 + (1 - labels) * np.maximum(0, training["margin"] - distances)**2
        )))
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        checkpoint = {
            "model_state": model.state_dict(), "model_config": model_cfg,
            "preprocessing": config["preprocessing"], "epoch": epoch, "history": history,
            "run_type": "smoke" if smoke else "baseline",
        }
        torch.save(checkpoint, artifact_dir / "last.pt")
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save(checkpoint, artifact_dir / "best.pt")
        else:
            stale += 1
        epoch_progress.set_postfix(
            train=f"{row['train_loss']:.4f}", val=f"{val_loss:.4f}", best=f"{best_loss:.4f}"
        )
        if stale >= training["patience"]:
            epoch_progress.set_description("Early stopping")
            break
    result = {
        "device": str(device), "epochs_completed": len(history), "best_val_loss": best_loss,
        "elapsed_seconds": time.perf_counter() - started, "history": history,
        "checkpoint": str((artifact_dir / "best.pt").resolve()),
    }
    (artifact_dir / "training_history.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
