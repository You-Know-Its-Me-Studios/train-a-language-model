# NVIDIA quickstart - Lesson 14

Use this after the main course's data and tokenizer preparation. The shared
GPU trainer and all four NVIDIA configurations are already included.
Run each block separately and stop if it reports an error.

## 1. Enter the project and check the driver

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
nvidia-smi
```

Use a current supported NVIDIA driver for your exact GPU that can run the
CUDA 12.8 PyTorch build. The `nvidia-smi` CUDA field describes driver
capability; `torch.version.cuda` reports the PyTorch build's runtime.

```powershell
nvidia-smi |
    Set-Content "logs\nvidia-driver.txt" -Encoding UTF8
```

## 2. Create the NVIDIA environment

```powershell
if (Test-Path ".venv-nvidia") {
    throw "The NVIDIA environment already exists. Inspect it first."
}
pymanager install 3.14
if ($LASTEXITCODE -ne 0) { throw "Python installation failed." }
pymanager exec -V:3.14 -m venv .venv-nvidia
if ($LASTEXITCODE -ne 0) { throw "Environment creation failed." }
$gpuPython = Join-Path $courseRoot `
    ".venv-nvidia\Scripts\python.exe"

foreach ($name in @("nvidia", "tokenizer", "model")) {
    & $gpuPython -m pip install --only-binary=:all: `
        -r "requirements-$name.txt"
    if ($LASTEXITCODE -ne 0) { throw "Installation failed: $name" }
}
& $gpuPython -m pip check
if ($LASTEXITCODE -ne 0) { throw "Dependency check failed." }
```

## 3. Check GPU operation

```powershell
@'
import torch
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
assert str(torch.__version__) == "2.10.0+cu128"
assert torch.cuda.is_available()
assert torch.version.hip is None
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda:0", requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss)
assert torch.isfinite(x.grad).all()
print("GPU forward and backward check passed.")
'@ | & $gpuPython -
```

```powershell
& $gpuPython -m torch.utils.collect_env |
    Set-Content "logs\nvidia-environment.txt" -Encoding UTF8
```

## 4. Run the model and recovery checks

```powershell
& $gpuPython scripts\train_gpu.py --vendor nvidia `
    --config configs\nvidia-check.json
```

Expected: `COMPLETE: 20 updates`.

```powershell
& $gpuPython scripts\train_gpu.py --vendor nvidia `
    --config configs\nvidia-recovery-stop.json --stop-after 10
```

Expected: `STOPPED: 10 updates`.

```powershell
& $gpuPython scripts\train_gpu.py --vendor nvidia `
    --config configs\nvidia-recovery-resumed.json `
    --resume models\nvidia-recovery-stop\checkpoints\stopped.pt
```

Expected: start after update 10 and finish at update 20.

```powershell
$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"
& $coursePython scripts\inspect_checkpoint.py `
    models\nvidia-recovery-resumed\checkpoints\final.pt
```

## 5. Train the baseline

```powershell
& $gpuPython scripts\train_gpu.py --vendor nvidia `
    --config configs\nvidia-baseline.json
```

This starts a fresh 4,000-update run. Use `docs/RECOVERY.md` for clean stops.
Use `docs/EVALUATE-AND-EXPORT.md` with `nvidia-baseline` after completion.

The trainer uses float32 and the math attention backend. It is a baseline
implementation, not a claim of maximum performance for the GPU.

If memory is insufficient, create a new experiment with microbatch 4 and
accumulation 8. That retains 32 sequences per update. Do not change those
settings when resuming a saved run.

## Sources

- [PyTorch release installation commands](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA driver downloads](https://www.nvidia.com/drivers/)
- [CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/index.html)
