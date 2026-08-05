# Offline Signature Verification — Notebook MVP Plan

## 1. MVP objective

Build and test a writer-independent offline signature-verification model using an open/research dataset and Jupyter notebooks.

The MVP will:

- ingest one signature dataset;
- preprocess and quality-check signature images;
- split data by writer;
- train a Siamese ConvNeXt-Tiny model with contrastive loss;
- evaluate it on writers not seen during training;
- calibrate decision thresholds from validation data;
- save the trained model, thresholds, plots, and experiment configuration;
- demonstrate predictions inside a notebook.

The MVP will not include FastAPI, a UI, customer enrollment, a database, cloud deployment, or bank integration.

## 2. MVP boundary

### Included

- `uv` and `pyproject.toml` project management
- dataset download/access instructions
- one dataset adapter plus a small synthetic test fixture
- canonical metadata CSV
- writer-level train/validation/test split
- preprocessing and basic image-quality checks
- positive and negative pair generation
- ConvNeXt-Tiny encoder and 256D L2-normalized embeddings
- Siamese training with contrastive loss
- random and skilled-forgery negatives
- optional hard-negative mining after the baseline works
- notebook-based exploration, training, evaluation, and prediction
- FAR, FRR, EER, ROC AUC, GAR, precision-recall, and forgery-type metrics
- validation-based threshold calibration
- local checkpoint and report artifacts

### Deferred

- API and inference service
- customer enrollment and reference storage
- STN unless the baseline has a clear alignment problem
- triplet loss unless contrastive loss is stable
- production security, authentication, monitoring, and deployment

## 3. Dataset: CEDAR

CEDAR is the fixed dataset for this MVP. The dataset will be supplied directly to the project, so no automated downloader is required.

Place the extracted dataset under:

```text
data/raw/cedar/
```

The adapter must discover the supplied layout instead of assuming that a third-party archive uses one particular filename convention. During ingestion, map each file to:

- `writer_id`: identity of the genuine signer;
- `target_writer_id`: signer being imitated; for a CEDAR skilled forgery this is the corresponding genuine writer;
- `sample_type`: `GENUINE` or `FORGERY`;
- `forgery_type`: `NONE` or `SKILLED`.

Before using the images:

1. Preserve the supplied archive unchanged until its checksum has been recorded.
2. Extract it into `data/raw/cedar/` without renaming the original files.
3. Add `data/raw/DATASET.md` containing the source/provider, received date, licence or usage terms, archive checksum, extraction command, and observed directory structure.
4. Run an inventory to discover actual writer IDs, file extensions, genuine counts, and forgery counts.
5. Open a sample of both classes manually to confirm that filename-derived labels are correct.
6. Keep `data/raw/cedar/` out of Git.

The dataset audit, rather than published or assumed counts, is the source of truth for this copy of CEDAR. If the supplied layout differs from the common CEDAR organization, only the adapter should change; the canonical schema and downstream training code should remain unchanged.

## 4. Target repository

```text
signature_verification/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   └── baseline.yaml
├── data/
│   ├── raw/                 # ignored by Git
│   ├── processed/           # ignored by Git
│   ├── splits/
│   └── fixtures/            # tiny non-sensitive test data
├── notebooks/
│   ├── 01_data_audit.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_train_siamese.ipynb
│   ├── 04_evaluate_calibrate.ipynb
│   └── 05_demo_prediction.ipynb
├── src/signature_verification/
│   ├── config.py
│   ├── datasets/
│   ├── preprocessing/
│   ├── pairs/
│   ├── models/
│   ├── losses/
│   ├── training/
│   └── evaluation/
├── tests/
├── artifacts/              # ignored by Git
└── reports/
```

Notebooks should orchestrate and visualize experiments. Reusable logic should live under `src/signature_verification/`, so training does not depend on hidden notebook state.

## 5. Local setup with uv

From PowerShell in the repository root:

```powershell
uv init --python 3.12
uv add torch torchvision timm opencv-python numpy pandas scikit-learn matplotlib seaborn pillow pyyaml tqdm
uv add --dev jupyterlab ipykernel pytest ruff
uv lock
uv sync
uv run python -m ipykernel install --user --name signature-verification --display-name "Signature Verification (uv)"
uv run jupyter lab
```

Notes:

- Commit `pyproject.toml` and `uv.lock`.
- Do not commit `.venv`, raw datasets, processed images, or large checkpoints.
- The generic PyTorch dependency is sufficient to start. If NVIDIA CUDA is required, select and document the appropriate PyTorch package source for the machine rather than manually mixing `pip` and `uv` installs.
- Run commands through `uv run`, for example `uv run pytest` and `uv run jupyter lab`.

## 6. Implementation phases

### Phase 1 — Project and data foundation

Tasks:

1. Create the `uv` project and repository structure.
2. Add `.gitignore` entries for `.venv/`, `data/raw/`, `data/processed/`, and `artifacts/`.
3. Receive and extract CEDAR, then write `data/raw/DATASET.md` with provider, licence, checksum, and extraction instructions.
4. Implement the first adapter to build a canonical CSV with:
   - `sample_id`
   - `image_path`
   - `writer_id`
   - `target_writer_id`
   - `sample_type`
   - `forgery_type`
   - `dataset`
   - `split`
5. Validate missing/corrupt images, duplicate IDs, label consistency, and per-writer counts.
6. Split writers deterministically: 70% train, 15% validation, 15% test.

Notebook output from `01_data_audit.ipynb`:

- dataset counts and class balance;
- genuine/forgery counts per writer;
- example images;
- corrupt/rejected sample list;
- proof that writer IDs do not overlap between splits.

Exit criteria:

- The canonical CSV is reproducible.
- No writer occurs in more than one split.
- Every excluded image has a recorded reason.

### Phase 2 — Preprocessing and quality checks

Tasks:

1. Load images safely and convert them to grayscale.
2. Crop whitespace without cutting strokes.
3. preserve aspect ratio, resize, and pad to 224 × 224.
4. Normalize and convert to PyTorch tensors.
5. Add configurable checks for blank images, blur, low contrast, clipping, and inadequate resolution.
6. Add conservative training augmentation only, such as small rotation, translation, and mild intensity variation.

Notebook output from `02_preprocessing.ipynb`:

- before/after contact sheets;
- quality measurements and rejection reasons;
- augmentation examples;
- rejection rate by split and class.

Exit criteria:

- Output tensors are consistently shaped.
- Stroke geometry is visually preserved.
- Quality failures return explicit reason codes.

### Phase 3 — Pair generation and model

Tasks:

1. Generate genuine–genuine positive pairs.
2. Generate genuine–skilled-forgery negative pairs.
3. Generate genuine–other-writer random negative pairs.
4. Prevent self-pairs, mirrored duplicates, and cross-split pairs.
5. Implement ConvNeXt-Tiny with a 256D projection layer and L2 normalization.
6. Implement cosine similarity and contrastive loss.
7. Add unit tests for pair labels, embedding shape/norm, similarity, and loss.

Exit criteria:

- Pair generation is deterministic and auditable.
- A forward/backward pass succeeds.
- The model can overfit a tiny sample, proving the training path works.

### Phase 4 — Baseline training

Initial configuration:

```yaml
model:
  encoder: convnext_tiny
  pretrained: true
  embedding_dim: 256
  use_stn: false

training:
  image_size: 224
  batch_size: 64
  learning_rate: 0.0001
  optimizer: adamw
  epochs: 30
  mixed_precision: true
  gradient_clip: 1.0
  seed: 42
```

Tasks:

1. Run a CPU-compatible one-batch test.
2. Run a one-epoch smoke test on a small subset.
3. Run a tiny-subset overfit test.
4. Train the full contrastive baseline.
5. Save best and last checkpoints, configuration, manifest hash, seed, losses, and package versions.
6. Add early stopping and resume support.

Notebook output from `03_train_siamese.ipynb`:

- configuration summary;
- training/validation loss curves;
- score distributions during training;
- saved checkpoint paths and run metadata.

Exit criteria:

- Training completes without NaNs.
- The best checkpoint reloads and reproduces validation scores.
- The run can be reproduced from the lockfile, config, and data manifest.

### Phase 5 — Evaluation and threshold calibration

Tasks:

1. Evaluate validation scores and calculate FAR, FRR, EER, ROC AUC, precision-recall, and GAR.
2. Report skilled and random forgery results separately.
3. Select threshold(s) from validation data only; never assume `0.5`.
4. Freeze the checkpoint, preprocessing, and threshold policy.
5. Run the untouched test writers once for the final report.
6. Save metrics as JSON/CSV and figures as PNG.

Notebook output from `04_evaluate_calibrate.ipynb`:

- ROC and precision-recall curves;
- FAR/FRR versus threshold;
- EER and selected threshold;
- genuine, skilled-forgery, and random-forgery score distributions;
- confusion counts and representative errors;
- final unseen-writer test metrics.

Exit criteria:

- Calibration uses validation writers only.
- Test metrics are produced by a repeatable evaluation function.
- Results clearly distinguish skilled from random forgeries.

### Phase 6 — Notebook prediction demo

Tasks:

1. Load the saved checkpoint and thresholds.
2. Select 3–5 genuine reference signatures for one test writer.
3. Embed a questioned signature.
4. Compare it with all reference embeddings using cosine similarity.
5. Aggregate scores using top-two average.
6. Display preprocessing, individual scores, aggregate score, threshold, and predicted result.

Notebook output from `05_demo_prediction.ipynb`:

- one genuine-query example;
- one skilled-forgery example;
- one random-forgery example;
- one unusable-image example;
- a simple callable `verify_in_notebook(reference_paths, query_path)` demonstration.

Exit criteria:

- A fresh kernel can run the notebook top to bottom.
- Predictions use saved artifacts rather than retraining.
- The result includes score and quality status, not only a label.

### Optional Phase 7 — Improve the baseline

Only after the baseline is reproducible:

1. Enable skilled-forgery-weighted sampling.
2. Add online hard-negative mining after epoch 5, refreshed every 2 epochs.
3. Compare with exactly the same writer splits and metrics.
4. Consider STN only if preprocessing/alignment errors are visible.
5. Consider triplet loss only if contrastive training has a documented limitation.

## 7. Suggested four-week schedule

| Week | Work | Deliverable |
|---|---|---|
| 1 | Setup, dataset acquisition, adapter, audit, writer split | Canonical dataset and `01_data_audit.ipynb` |
| 2 | Preprocessing, quality checks, pairs, model tests | Validated inputs and working Siamese forward/backward pass |
| 3 | Smoke tests and full baseline training | Reproducible best checkpoint and training notebook |
| 4 | Evaluation, calibration, prediction demo, documentation | Metrics report and end-to-end notebook demonstration |

Dataset access may run in parallel with development against the tiny fixture. Hard-negative mining is an extension, not a prerequisite for the first MVP result.

## 8. MVP acceptance criteria

The MVP is complete when:

1. `uv sync` creates a working environment from committed project files.
2. One real dataset loads through the canonical adapter.
3. Writer train/validation/test overlap is zero.
4. Preprocessing and quality checks are visually and automatically tested.
5. Contrastive training completes and saves a reloadable ConvNeXt-Tiny checkpoint.
6. Evaluation reports FAR, FRR, EER, ROC AUC, GAR, precision-recall, and separate skilled/random forgery performance.
7. Threshold calibration uses validation data rather than a fixed cutoff.
8. The final notebook compares a query with 3–5 references and shows its score and prediction.
9. `uv run pytest` passes, including a one-epoch smoke test.
10. A README explains setup, dataset placement, notebook order, reproduction, and known limitations.

## 9. Actions you can take now

1. Install `uv` by following <https://docs.astral.sh/uv/getting-started/installation/> and confirm with `uv --version`.
2. Provide the CEDAR archive or extracted directory and confirm its usage/licence terms.
3. Confirm whether training will use CPU, NVIDIA GPU, or Apple Silicon; record GPU model and available memory.
4. Create `data/raw/cedar/`, extract the unchanged CEDAR files there, and keep it out of Git.
5. Save the dataset licence, source URL, archive checksum, and extraction notes in `data/raw/DATASET.md`.
6. Run the `uv` setup commands in Section 5.
7. Open JupyterLab through `uv run jupyter lab` and select the `Signature Verification (uv)` kernel.
8. Begin with the dataset-audit notebook; do not begin full training until the writer-overlap assertion passes.
