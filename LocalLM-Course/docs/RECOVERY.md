# Stop and recover training

## Clean stop on Windows

In a second PowerShell window, set the actual active run name:

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
$activeRun = "cpu-baseline"
$stopPath = Join-Path $courseRoot "models\$activeRun\STOP"
New-Item -ItemType File -Path $stopPath
```

Use the NVIDIA run name for NVIDIA training. The trainer checks this file
between optimizer updates. Wait for `STOPPED` and the saved `stopped.pt`
message before closing the terminal.

## Resume on Windows

Keep the same source files, inputs, environment, GPU when applicable, and
training settings. Create a new run name from the original configuration:

```powershell
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
$oldRun = "cpu-baseline"
$newRun = "cpu-baseline-resumed-01"
$config = Get-Content "models\$oldRun\config.json" -Raw |
    ConvertFrom-Json
$config.run_name = $newRun
$configPath = "configs\$newRun.json"
if (Test-Path $configPath) { throw "Configuration already exists." }
$config | ConvertTo-Json -Depth 10 |
    Set-Content $configPath -Encoding UTF8
```

For CPU:

```powershell
$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"
& $coursePython scripts\train.py --config $configPath `
    --resume "models\$oldRun\checkpoints\stopped.pt"
```

For NVIDIA, set `$oldRun` and `$newRun` to the NVIDIA run names before
creating the configuration, then use:

```powershell
$gpuPython = Join-Path $courseRoot `
    ".venv-nvidia\Scripts\python.exe"
& $gpuPython scripts\train_gpu.py --vendor nvidia `
    --config $configPath `
    --resume "models\$oldRun\checkpoints\stopped.pt"
```

## Clean stop and resume in AMD WSL

In a second Ubuntu terminal:

```bash
touch "$HOME/LocalLM-Course-AMD/models/amd-baseline/STOP"
```

Wait for the saved checkpoint, then in the training terminal:

```bash
cd "$HOME/LocalLM-Course-AMD"
gpu_python="$PWD/.venv-amd/bin/python"
export HSA_ENABLE_DXG_DETECTION=1

"$gpu_python" - <<'PY'
import json
from pathlib import Path
old = "amd-baseline"
new = "amd-baseline-resumed-01"
config = json.loads(Path(f"models/{old}/config.json").read_text())
config["run_name"] = new
with Path(f"configs/{new}.json").open("x") as stream:
    json.dump(config, stream, indent=2)
    stream.write("\n")
PY
```

```bash
"$gpu_python" scripts/train_gpu.py --vendor amd \
    --config configs/amd-baseline-resumed-01.json \
    --resume models/amd-baseline/checkpoints/stopped.pt
```

## Crash or power interruption

Use the last complete checkpoint that passes `inspect_checkpoint.py`.
A `.pt.tmp` file is not a finished checkpoint. A completed `final.pt` is
already at its configured budget and cannot resume under that same budget.

Do not remove a fingerprint check to force a resume. It protects against
mixing incompatible data, tokenizer mappings, source code, or environments.

Do not change batch sizes or learning rates in an existing resume
configuration. Those changes belong to a new experiment.

Each resumed folder starts its own best-checkpoint tracking. Retain the
earlier segments and include their best candidates in final validation.
