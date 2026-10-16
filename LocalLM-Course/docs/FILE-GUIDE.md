# File guide

## Course source files

| Stage | File | Purpose |
|---|---|---|
| Data | scripts/prepare_data.py | Download, clean, split, and verify data |
| Tokenizer | scripts/prepare_tokens.py | Fit BPE and encode all splits |
| Tokenizer | scripts/inspect_tokenizer.py | Inspect token IDs and text round trips |
| Model | scripts/model.py | Model configuration and architecture |
| Model | scripts/inspect_model.py | Check shapes, masking, gradients, and context |
| Training | scripts/training_utils.py | Configuration, batches, optimizer, and updates |
| Training | scripts/check_training_config.py | Check the CPU training configuration |
| Training | scripts/train.py | CPU training, checkpoints, and recovery |
| Recovery | scripts/inspect_checkpoint.py | Inspect saved training state |
| Recovery | scripts/compare_checkpoints.py | Compare uninterrupted and resumed CPU state |
| Evaluation | scripts/inference.py | Shared CPU loading and generation functions |
| Evaluation | scripts/evaluate.py | Score a complete data split |
| Evaluation | scripts/review_samples.py | Generate the five fixed-prompt samples |
| Generation | scripts/generate_text.py | Single-prompt or interactive generation |
| Export | scripts/package_model.py | Package the selected checkpoint and records |
| Lessons 14 and 16 | scripts/train_gpu.py | Shared NVIDIA and AMD trainer |

## Configuration files

For each of `cpu`, `nvidia`, and `amd`, the following files are supplied:

- `<vendor>-baseline.json`: fresh 4,000-update run.
- `<vendor>-check.json`: uninterrupted 20-update functionality check.
- `<vendor>-recovery-stop.json`: 20-update budget, stopped at 10 by CLI.
- `<vendor>-recovery-resumed.json`: resume that check to update 20.

Full-run resume configurations are created from the saved full-run
configuration when needed. Change only `run_name` when resuming.

## Other files

- `requirements-cpu.txt`: CPU PyTorch wheel.
- `requirements-tokenizer.txt`: shared tokenizer package.
- `requirements-model.txt`: NumPy for CPU and NVIDIA environments.
- `requirements-nvidia.txt`: CUDA-enabled PyTorch wheel.
- `requirements-amd.txt`: AMD wheels and compatible packages.
- `model-README.md`: README copied into a model export.
- `templates/run-notes.md`: blank experiment record.
- `templates/selection-notes.md`: blank model-review notes.
- `verify_download.py`: supplied-file checksum verification.
- `DOWNLOAD-MANIFEST.json`: file hashes and build information.
- `reference/source-provenance.json`: course source hashes at packaging.

`data`, `models`, `logs`, `outputs`, `exports`, and `notes` begin empty in
the source bundle. The workflows create their actual contents.
