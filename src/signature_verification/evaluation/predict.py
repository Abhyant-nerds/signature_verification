from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from signature_verification.models import SiameseNetwork
from signature_verification.preprocessing import PreprocessConfig, preprocess_signature


def load_model(checkpoint_path: str | Path, device: str | None = None):
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=target, weights_only=False)
    cfg = checkpoint["model_config"]
    model = SiameseNetwork(cfg["encoder"], cfg["embedding_dim"], pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model.to(target).eval(), PreprocessConfig(**checkpoint["preprocessing"]), target


@torch.no_grad()
def verify_signatures(model, reference_paths, query_path, threshold, cfg, device):
    query, quality, _ = preprocess_signature(query_path, cfg)
    if not quality.usable:
        return {"decision": "UNUSABLE", "quality_reasons": list(quality.reasons)}
    query_embedding = model.encode(query.unsqueeze(0).to(device))
    scores = []
    for path in reference_paths:
        reference, reference_quality, _ = preprocess_signature(path, cfg)
        if not reference_quality.usable:
            raise ValueError(f"Unusable reference image: {path}")
        embedding = model.encode(reference.unsqueeze(0).to(device))
        scores.append(float(F.cosine_similarity(query_embedding, embedding).item()))
    if len(scores) < 2:
        aggregate = scores[0]
    else:
        aggregate = float(np.mean(sorted(scores, reverse=True)[:2]))
    return {
        "decision": "ACCEPT" if aggregate >= threshold else "REJECT",
        "score": aggregate,
        "reference_scores": scores,
        "threshold": float(threshold),
        "quality_reasons": [],
    }
