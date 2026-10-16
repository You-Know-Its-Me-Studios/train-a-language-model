# LocalLM Course Downloads

Start here after extracting the source ZIP. This project contains the code
and configurations used by the main course and the NVIDIA and AMD
supplements. It trains a small story-continuation model from initialization.

## Use the downloaded project

Extract the ZIP into your Windows user folder. The project path should be:

```text
C:\Users\YourName\LocalLM-Course
```

Do not extract it over an existing training project. Use a separate folder
when comparing this download with your own work.

Open PowerShell and enter:

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
```

If you used another location, set `$courseRoot` to that actual location.
The Python scripts locate the project from their own file location.

## Choose a starting point

- Follow [CPU-QUICKSTART.md](docs/CPU-QUICKSTART.md) for the main course.
- Follow [NVIDIA-QUICKSTART.md](docs/NVIDIA-QUICKSTART.md) for Lesson 14.
- Follow [AMD-QUICKSTART.md](docs/AMD-QUICKSTART.md) for Lesson 16.
- Use [EVALUATE-AND-EXPORT.md](docs/EVALUATE-AND-EXPORT.md) after training.
- Use [RECOVERY.md](docs/RECOVERY.md) for clean stops and recovery.
- Read [FILE-GUIDE.md](docs/FILE-GUIDE.md) to find a particular file.
- Read [VALIDATION.md](docs/VALIDATION.md) for the checks and their limits.

## How this download relates to the lessons

The source files and named configurations are already supplied. When a
lesson asks you to create one of those files, inspect the supplied file
instead. Skip the file-creation block unless you are building your own
project from an empty folder.

In particular, the GPU supplements' configuration-generation blocks will
refuse to overwrite the configuration files included here. That is expected.
Continue with the training commands using the supplied configurations.

The shared GPU trainer is included for both vendors. AMD users do not need
to install the NVIDIA environment or copy code out of the NVIDIA lesson.

The complete updated package builder is included. It records every training
source named in the run fingerprint, including the GPU trainer.

## Data choices

For the complete learning workflow, run the data preparation and tokenizer
steps in the CPU quickstart. The preparation script downloads a bounded
training prefix and the upstream validation file from the pinned source.

The separate reference-data ZIP is optional. It includes source data,
prepared splits, a fitted tokenizer, token files, and their manifests.
It is useful for comparing preparation results or continuing after the data
lessons. Read that ZIP's README before copying its files into a project.

Keep each dataset, tokenizer, and token manifest together. Do not replace
them underneath an existing training run. Independent preparations can have
different environment metadata and manifest hashes.

## Environments

| Path | Python | PyTorch | NumPy | tokenizers |
|---|---|---|---|---|
| CPU on Windows | 3.14 | 2.10.0+cpu | 2.4.6 | 0.22.2 |
| NVIDIA on Windows | 3.14 | 2.10.0+cu128 | 2.4.6 | 0.22.2 |
| AMD in Ubuntu 24.04 / WSL2 | 3.12 | AMD ROCm 7.2.1 build of 2.9.1 | 1.26.4 | 0.22.2 |

These are separate virtual environments. Install only the requirements for
the path you are following. The AMD requirement file contains the complete
wheel build identifiers and official download URLs.

The requirements pin direct dependencies. They are not complete lockfiles
for every transitive dependency, operating system package, or driver.
Training records capture the environment used by the actual run.

## Check the supplied files

After creating the CPU environment, run:

```powershell
$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"

& $coursePython verify_download.py
```

This checks the shipped files against `DOWNLOAD-MANIFEST.json`. It allows
new training outputs but reports changes to the supplied source files.
An intentional edit will therefore cause a hash mismatch.

The verifier uses Python's standard library only. It can also run under
the AMD Python environment.

## Model and run settings

The model has 2,557,888 unique parameters, a 4,096-token BPE vocabulary,
four transformer blocks, six attention heads, and a 256-token context.
The baseline uses microbatches of 8 with 4 accumulation steps.

Each baseline configuration has a 4,000-update budget. Each check or
recovery-check configuration has a 20-update budget. Every configuration
has a separate run name. A completed run folder is never overwritten.

No trained model, validation score, or selected-model record is supplied
as a student result. Those are created by running the course. Short author
test checkpoints are excluded from these downloads.

## Course order after the main lessons

13. End-of-Course Quiz
14. NVIDIA GPU Training
15. NVIDIA GPU Training Quiz
16. AMD GPU Training
17. AMD GPU Training Quiz

The quizzes remain in the Leanpub course. This download contains project
files and operating instructions.

## Attribution and records

Read [DATA-SOURCE.md](docs/DATA-SOURCE.md) for the dataset revision and
published data terms. The course code is separate from the third-party
dataset. This download does not assign new licensing terms to that code.

Preserve the full project to resume training. The inference export created
by `package_model.py` is a smaller deliverable and does not include the
complete training dataset or a recreated training environment.
