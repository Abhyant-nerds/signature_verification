# CEDAR Offline Signature Verification MVP

Notebook-driven writer-independent signature verification using a Siamese ConvNeXt-Tiny model, cosine similarity, and contrastive loss.

## Current status

- CEDAR audit: 2,640 readable images from 55 writers
- Split: 38 train writers, 8 validation writers, 9 test writers
- Writer overlap: zero
- One-epoch no-pretrained smoke training: passed
- Automated tests: 8 passed

The smoke checkpoint only verifies that the implementation runs. It is not an evaluated model and must not be used as the final result.

## Setup

```powershell
uv sync
uv run pytest
uv run jupyter lab
```

The supplied CEDAR files must use this location:

```text
data/raw/cedar/signatures/full_org/
data/raw/cedar/signatures/full_forg/
```

## Workflow

Run the audit before training:

```powershell
uv run sv-audit
```

Run a cheap pipeline smoke test without downloading pretrained weights:

```powershell
uv run sv-train --smoke --no-pretrained
```

Run the baseline experiment:

```powershell
uv run sv-train
```

This uses pretrained ConvNeXt-Tiny weights. The first run may need network access to obtain them. The project pins the official PyTorch 2.7.1 CUDA 12.8 build for Windows/Linux through `uv`; it has been verified on this machine's NVIDIA GTX 1650.

Verify CUDA before a long run:

```powershell
uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

After training, calibrate on validation writers and evaluate test writers:

```powershell
uv run sv-evaluate
```

Generated artifacts are written to `artifacts/baseline/`. Raw data and model artifacts are intentionally ignored by Git.

## Notebooks

Open these in order:

1. `notebooks/01_data_audit.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_train_siamese.ipynb`
4. `notebooks/04_evaluate_calibrate.ipynb`
5. `notebooks/05_demo_prediction.ipynb`

Notebooks call reusable code under `src/signature_verification`; they are not the sole implementation.

## Important evaluation rule

All splitting is performed by writer. Thresholds are selected from validation writers only. Test writers must remain untouched until the final experiment configuration is frozen.

## Limitations

This is an experimental MVP trained on a small research dataset. It is not a banking control, identity-proofing system, or production fraud decision service. CEDAR performance does not establish performance on UK commercial-bank documents, scanners, customers, or attack patterns.
