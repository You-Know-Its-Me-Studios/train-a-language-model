# Download validation

Build date: 2026-10-16.

The packaged course scripts were checked against the lesson source files
byte-for-byte. All 16 course Python files are included without code changes
made during download preparation. Configuration variants preserve the
course settings and use the documented vendor-specific run names.

## Executed checks

Checks ran in a fresh project copy on Linux with Python 3.14.8,
PyTorch 2.10.0+cpu, NumPy 2.4.6, and tokenizers 0.22.2.

- Downloaded the two pinned source files and verified their hashes.
- Prepared 30,496 training, 5,000 validation, and 2,000 test stories.
- Fitted the tokenizer on training stories and encoded all splits.
- Verified counts of 6,080,666, 977,065, and 398,416 tokens.
- Checked tokenizer round trips and story boundaries.
- Inspected model dimensions, tied weights, causal masking, and gradients.
- Checked configuration, batch replay, and an optimizer update.
- Completed an uninterrupted 20-update CPU run.
- Stopped a separate run after update 10 and resumed it to update 20.
- Confirmed exact model, optimizer, CPU RNG, and sampler-state equality
  between the uninterrupted and resumed CPU runs.
- Evaluated the full validation split for best and final checkpoints.
- Generated the five fixed-prompt samples for both candidates.
- Evaluated all 398,415 test targets for the selected check checkpoint.
- Generated text through the normal command-line interface.
- Exported the check checkpoint with actual selection and evaluation records.
- Extracted the inference export, checked its manifest, and generated text.
- Confirmed that the GPU trainer rejects the CPU wheel for NVIDIA and
  rejects Python 3.14 for the pinned AMD path.

The short training and evaluation runs test program operation. They do not
measure the quality of a completed 4,000-update baseline. Their checkpoints
and generated reports are not distributed as student results.

## Package checks

The final source and reference-data archives are extracted and verified
against their included manifests. The reference data is checked with the
packaged preparation scripts. External SHA-256 hashes identify the two ZIPs
and the download README.

Authored instructions are ASCII text. Python source syntax, Bash command
syntax, configuration structure, relative document links, and archive
contents are checked during assembly.

## Not executed in this environment

- Windows PowerShell command execution.
- Windows Python Install Manager installation.
- NVIDIA CUDA execution on an actual GPU.
- AMD Windows driver, WSL2, ROCDXG, and ROCm execution on actual hardware.
- A full 4,000-update training run and quality review.

Perform the vendor-specific checks in the course on the target hardware
before presenting that hardware path as verified.
