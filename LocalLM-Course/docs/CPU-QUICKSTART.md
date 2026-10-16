# CPU quickstart

Use this with the main course. Run one block at a time and stop if a command
fails. The scripts and configurations named below are already supplied.

## 1. Enter the project

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
```

Install the Python Install Manager as described in the course if the
`pymanager` command is not available.

## 2. Create the CPU environment

```powershell
if (Test-Path ".venv-cpu") {
    throw "The CPU environment already exists. Inspect it first."
}

pymanager install 3.14
if ($LASTEXITCODE -ne 0) { throw "Python installation failed." }

pymanager exec -V:3.14 -m venv .venv-cpu
if ($LASTEXITCODE -ne 0) { throw "Environment creation failed." }

$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"

foreach ($name in @("cpu", "tokenizer", "model")) {
    & $coursePython -m pip install --only-binary=:all: `
        -r "requirements-$name.txt"
    if ($LASTEXITCODE -ne 0) {
        throw "Package installation failed: $name"
    }
}

& $coursePython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
```

When reopening PowerShell, repeat step 1 and set `$coursePython` again.
Do not recreate an environment that is already working.

## 3. Verify the source download

```powershell
& $coursePython verify_download.py
```

Expected final line: `Download verification passed.`

## 4. Source and prepare the data

Read `docs/DATA-SOURCE.md`, then run:

```powershell
& $coursePython scripts\prepare_data.py
```

Expected story counts: train 30,496; validation 5,000; test 2,000.

This downloads the exact bounded source files and verifies their SHA-256
hashes. It normalizes whitespace, removes exact duplicates, discards the
incomplete training tail, and creates the split manifests.

If you installed the optional reference data, use this instead:

```powershell
& $coursePython scripts\prepare_data.py --verify
```

## 5. Fit the tokenizer and encode the splits

```powershell
& $coursePython scripts\prepare_tokens.py
```

The tokenizer is fitted on training stories only. It uses a 4,096-token
vocabulary and end-of-text ID 0. The reference build contains 6,080,666
training tokens, 977,065 validation tokens, and 398,416 test tokens.
Use your own generated manifests as the record of your build.

If you installed the optional reference data, use this instead:

```powershell
& $coursePython scripts\prepare_tokens.py --verify
```

Inspect the tokenizer:

```powershell
& $coursePython scripts\inspect_tokenizer.py
```

Expected result: a successful text round trip.

## 6. Inspect the model and training configuration

```powershell
& $coursePython scripts\inspect_model.py
```

Expected results include 2,557,888 parameters, finite gradients, a passing
causal-mask check, and an enforced context limit.

```powershell
& $coursePython scripts\check_training_config.py `
    --config configs\cpu-baseline.json
```

This performs one update as a configuration check without saving a model.

## 7. Run 20 updates

```powershell
& $coursePython scripts\train.py --config configs\cpu-check.json
```

Expected final status: `COMPLETE: 20 updates`.

```powershell
& $coursePython scripts\inspect_checkpoint.py `
    models\cpu-check\checkpoints\final.pt
```

## 8. Check stop and resume

```powershell
& $coursePython scripts\train.py `
    --config configs\cpu-recovery-stop.json `
    --stop-after 10
```

Expected final status: `STOPPED: 10 updates`.

```powershell
& $coursePython scripts\train.py `
    --config configs\cpu-recovery-resumed.json `
    --resume models\cpu-recovery-stop\checkpoints\stopped.pt
```

Expected final status: `COMPLETE: 20 updates`.

```powershell
& $coursePython scripts\compare_checkpoints.py `
    models\cpu-check\checkpoints\final.pt `
    models\cpu-recovery-resumed\checkpoints\final.pt
```

This compares the uninterrupted and resumed CPU runs in the same
environment. It is not a cross-vendor GPU reproducibility test.

## 9. Run the 4,000-update baseline

```powershell
& $coursePython scripts\train.py --config configs\cpu-baseline.json
```

Use `docs/RECOVERY.md` if you need to stop. Keep every segment's run folder.

## 10. Evaluate and export

Follow `docs/EVALUATE-AND-EXPORT.md`. Use `cpu-baseline` as the run name
for an uninterrupted baseline. Do not export the 20-update check as if it
were a completed baseline model.
