from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch


@dataclass(frozen=True)
class PreprocessConfig:
    image_size: int = 224
    crop_margin: int = 8
    min_ink_fraction: float = 0.0005
    max_ink_fraction: float = 0.50
    min_contrast: float = 10.0
    blur_threshold: float = 8.0
    min_dimension: int = 32


@dataclass(frozen=True)
class QualityResult:
    usable: bool
    reasons: tuple[str, ...]
    measurements: dict[str, float]


def _load_grayscale(source: str | Path | np.ndarray) -> np.ndarray:
    if isinstance(source, np.ndarray):
        image = source.copy()
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        image = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise ValueError(f"Unable to decode signature image: {source}")
    return image.astype(np.uint8)


def assess_quality(image: np.ndarray, cfg: PreprocessConfig) -> QualityResult:
    height, width = image.shape
    contrast = float(np.percentile(image, 95) - np.percentile(image, 5))
    threshold = max(0, min(254, int(np.percentile(image, 50) - 20)))
    ink_fraction = float(np.mean(image < threshold))
    blur = float(cv2.Laplacian(image, cv2.CV_64F).var())
    border = np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))
    border_ink_fraction = float(np.mean(border < threshold))
    reasons: list[str] = []
    if min(height, width) < cfg.min_dimension:
        reasons.append("LOW_RESOLUTION")
    if contrast < cfg.min_contrast:
        reasons.append("LOW_CONTRAST")
    if ink_fraction < cfg.min_ink_fraction:
        reasons.append("BLANK")
    if ink_fraction > cfg.max_ink_fraction:
        reasons.append("EXCESSIVE_INK")
    if blur < cfg.blur_threshold:
        reasons.append("BLUR")
    if border_ink_fraction > 0.10:
        reasons.append("POSSIBLE_CLIPPING")
    return QualityResult(
        usable=not reasons,
        reasons=tuple(reasons),
        measurements={
            "width": float(width), "height": float(height), "contrast": contrast,
            "ink_fraction": ink_fraction, "blur": blur,
            "border_ink_fraction": border_ink_fraction,
        },
    )


def crop_whitespace(image: np.ndarray, margin: int = 8) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, ink = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    points = cv2.findNonZero(ink)
    if points is None:
        return image
    x, y, width, height = cv2.boundingRect(points)
    x0, y0 = max(0, x - margin), max(0, y - margin)
    x1 = min(image.shape[1], x + width + margin)
    y1 = min(image.shape[0], y + height + margin)
    return image[y0:y1, x0:x1]


def resize_and_pad(image: np.ndarray, size: int) -> np.ndarray:
    height, width = image.shape
    scale = min(size / width, size / height)
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((size, size), 255, dtype=np.uint8)
    y = (size - resized.shape[0]) // 2
    x = (size - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def preprocess_signature(
    source: str | Path | np.ndarray,
    cfg: PreprocessConfig | None = None,
    reject_unusable: bool = False,
) -> tuple[torch.Tensor, QualityResult, np.ndarray]:
    cfg = cfg or PreprocessConfig()
    image = _load_grayscale(source)
    quality = assess_quality(image, cfg)
    if reject_unusable and not quality.usable:
        raise ValueError(f"Unusable signature: {', '.join(quality.reasons)}")
    processed = resize_and_pad(crop_whitespace(image, cfg.crop_margin), cfg.image_size)
    tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    tensor = tensor.repeat(3, 1, 1)
    return tensor, quality, processed
