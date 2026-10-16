# Evaluate and export

Run this in Windows PowerShell with the CPU environment after completing
training. AMD users first copy the whole completed run folder back from WSL.
These commands create new records and refuse existing output filenames.

## 1. Identify the candidates

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"

$runName = "cpu-baseline"
$candidates = @(
    [pscustomobject]@{
        Label = "$runName-best"
        Checkpoint = "models\$runName\checkpoints\best.pt"
    },
    [pscustomobject]@{
        Label = "$runName-final"
        Checkpoint = "models\$runName\checkpoints\final.pt"
    }
)
```

Use `nvidia-baseline` or `amd-baseline` for those runs. If training resumed,
use the completed segment's run name and add earlier segments' best
checkpoints to `$candidates` before evaluating. For example:

```powershell
# Run this only when this earlier segment exists and is relevant.
# Use its actual folder name and an unused label.
# $candidates += [pscustomobject]@{
#     Label = "cpu-baseline-earlier-best"
#     Checkpoint = "models\cpu-baseline\checkpoints\best.pt"
# }
```

Each run folder tracks its own best sampled validation checkpoint.

## 2. Evaluate validation and generate fixed-prompt samples

```powershell
$records = @()
foreach ($candidate in $candidates) {
    $validation = "logs\$($candidate.Label)-validation.json"
    $samples = "logs\$($candidate.Label)-samples.json"

    & $coursePython scripts\evaluate.py `
        --checkpoint $candidate.Checkpoint `
        --split val --output $validation
    if ($LASTEXITCODE -ne 0) { throw "Validation failed." }

    & $coursePython scripts\review_samples.py `
        --checkpoint $candidate.Checkpoint --output $samples
    if ($LASTEXITCODE -ne 0) { throw "Sample generation failed." }

    $report = Get-Content $validation -Raw | ConvertFrom-Json
    $records += [pscustomobject]@{
        Label = $candidate.Label
        Loss = $report.mean_nll
        Report = $report
        Validation = $validation
        Samples = $samples
    }
}
$records | Select-Object Label, Loss
```

The reference validation split has 977,064 scored targets. The evaluation
scores each target once using the course's disjoint-input-window protocol.
Read the generated stories and record weaknesses in your selection notes.

## 3. Freeze the selection

This baseline selects the lowest full validation mean negative
log-likelihood among the candidates. It does not claim a human quality
review has passed.

```powershell
$chosen = $records | Sort-Object Loss, Label | Select-Object -First 1
$selectionPath = "notes\selected-$runName-model.json"
if (Test-Path $selectionPath) {
    throw "A selection record already exists."
}
New-Item -ItemType Directory -Force "notes" | Out-Null

$selection = [ordered]@{
    selected_utc = [DateTime]::UtcNow.ToString("o")
    checkpoint_path = $chosen.Report.checkpoint_path
    checkpoint_sha256 = $chosen.Report.checkpoint_sha256
    completed_updates = $chosen.Report.completed_updates
    validation_report = $chosen.Validation
    samples_report = $chosen.Samples
    validation_mean_nll = $chosen.Report.mean_nll
    validation_perplexity = $chosen.Report.perplexity
    reason = "Lowest full validation NLL among recorded candidates."
}
$selection | ConvertTo-Json -Depth 10 |
    Set-Content $selectionPath -Encoding UTF8
```

## 4. Evaluate the selected checkpoint on test data

```powershell
$testReport = "logs\$runName-selected-test.json"
& $coursePython scripts\evaluate.py `
    --checkpoint $selection.checkpoint_path `
    --split test --output $testReport
if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed." }
```

The reference test split has 398,415 scored targets. Use validation for
selection. Repeated comparisons on the course test split are not new
independent holdout evaluations.

## 5. Generate and package

```powershell
& $coursePython scripts\generate_text.py `
    --checkpoint $selection.checkpoint_path `
    --prompt "A small bird found a key beside the river." `
    --max-new-tokens 80 --temperature 0.8 --top-k 40 --seed 42
if ($LASTEXITCODE -ne 0) { throw "Generation failed." }
```

Review `model-README.md` before packaging. Describe the actual run and
limitations. Then run:

```powershell
& $coursePython scripts\package_model.py `
    --selection $selectionPath --test-report $testReport `
    --output-dir "exports\$runName-story-model"
if ($LASTEXITCODE -ne 0) { throw "Packaging failed." }
```

Expected outputs are a package folder, ZIP archive, and SHA-256 file.
The export includes the actual checkpoint and selection/evaluation records.

Keep the complete training project separately. Moving an inference export
does not recreate the data and GPU environment needed for training recovery.
