# AMD quickstart - Lesson 16

This path uses Windows 11, WSL2, Ubuntu 24.04, ROCm 7.2.1, ROCDXG 1.2.0,
and AMD's Python 3.12 PyTorch wheels. The shared trainer and AMD
configurations are already included in the source ZIP.

Use the main course to prepare the data in Windows, or install the optional
reference-data bundle first. The Linux project uses a copy of those inputs.

Run each block separately and resolve any error before continuing.

## 1. Check hardware and install the Windows driver

Compare the exact GPU with the ROCm 7.2.x row in AMD's
[ROCDXG compatibility matrix](https://github.com/ROCm/librocdxg).
Use AMD's [ROCm 7.2.1 WSL guide][wsl-guide] to choose a compatible Windows
driver. That guide introduces this route with Adrenalin 26.2.2.

In Windows PowerShell:

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
Set-Location -LiteralPath $courseRoot -ErrorAction Stop
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion |
    Format-List | Out-String |
    Set-Content "logs\amd-windows-driver.txt" -Encoding UTF8
```

The Windows driver provides the GPU interface. Do not install Linux
display drivers or `amdgpu-dkms` inside WSL.

## 2. Install or check WSL2

In PowerShell:

```powershell
wsl --list --verbose
```

If Ubuntu 24.04 is absent, use an administrator PowerShell window:

```powershell
wsl --install -d Ubuntu-24.04
```

Follow first-run prompts to create the Linux user and password. Restart
Windows only if the installation requests it. Then update WSL:

```powershell
wsl --update
wsl --list --verbose
```

Ubuntu 24.04 must show version 2. If it shows version 1, convert it:

```powershell
wsl --set-version Ubuntu-24.04 2
```

Record the WSL version and obtain the Windows project path in Linux form:

```powershell
wsl --version |
    Set-Content "logs\amd-wsl-version.txt" -Encoding UTF8
wsl -d Ubuntu-24.04 -- wslpath -u $courseRoot
```

Save the printed path. It resembles `/mnt/c/Users/YourName/LocalLM-Course`.

Open Ubuntu:

```powershell
wsl -d Ubuntu-24.04
```

## 3. Install Linux prerequisites and ROCm

The following blocks run in Ubuntu Bash:

```bash
sudo apt update
```

```bash
sudo apt install \
    ca-certificates curl rsync \
    python3.12-venv python3-pip python3-setuptools python3-wheel
```

```bash
python3.12 --version
mkdir -p "$HOME/rocm-course-installers"
cd "$HOME/rocm-course-installers"
installer_base="https://repo.radeon.com/amdgpu-install"
installer_path="7.2.1/ubuntu/noble"
installer_file="amdgpu-install_7.2.1.70201-1_all.deb"
curl -fL \
    "$installer_base/$installer_path/$installer_file" \
    -o "$installer_file"
```

```bash
sudo apt install "./$installer_file"
```

```bash
sudo apt update
```

```bash
sudo apt install rocm
```

Use the repository setup and user-space packages only. Skip the native
Linux kernel-driver instructions in the general ROCm installation guide.

## 4. Install ROCDXG and verify the GPU agent

```bash
cd "$HOME/rocm-course-installers"
dxg_base="https://github.com/ROCm/librocdxg/releases/download"
dxg_file="rocdxg-roct_1.2.0_amd64.deb"
curl -fL "$dxg_base/v1.2.0/$dxg_file" -o "$dxg_file"
```

```bash
sudo apt install "./$dxg_file"
```

```bash
sudo ldconfig
export HSA_ENABLE_DXG_DETECTION=1
ls -l /dev/dxg
/opt/rocm/bin/rocminfo
```

ROCm must list the supported GPU as an agent. A CPU agent alone does not
pass this check. Set the DXG environment variable in each new training shell.

## 5. Copy the project to the Linux filesystem

Paste the Linux-form project path saved in step 2:

```bash
read -r -p "Paste the Windows project path: " windows_project
test -f "$windows_project/scripts/train_gpu.py"
test -f "$windows_project/data/tokenized/manifest.json"
```

Create a new destination. If the folder exists, inspect it before proceeding:

```bash
linux_project="$HOME/LocalLM-Course-AMD"
mkdir "$linux_project"
```

Copy into that new folder:

```bash
rsync -a \
    --exclude='.venv*' --exclude='.git/' \
    --exclude='models/' --exclude='logs/' \
    --exclude='outputs/' --exclude='exports/' \
    "$windows_project/" "$linux_project/"
cd "$linux_project"
mkdir -p models logs outputs exports
cp "$windows_project/logs/amd-windows-driver.txt" logs/
cp "$windows_project/logs/amd-wsl-version.txt" logs/
```

## 6. Create the Python environment

```bash
python3.12 -m venv .venv-amd
gpu_python="$PWD/.venv-amd/bin/python"
```

The supplied requirement file points to the same two AMD wheels used in
Lesson 16. It also pins NumPy 1.26.4 and tokenizers 0.22.2:

```bash
"$gpu_python" -m pip install --only-binary=:all: \
    -r requirements-amd.txt
```

```bash
"$gpu_python" -m pip check
"$gpu_python" verify_download.py
"$gpu_python" scripts/prepare_tokens.py --verify
```

Do not install the CPU or NVIDIA PyTorch requirements into `.venv-amd`.

## 7. Check GPU operation

```bash
export HSA_ENABLE_DXG_DETECTION=1
"$gpu_python" - <<'PY'
import numpy as np
import tokenizers
import torch
assert str(torch.__version__) == (
    "2.9.1+rocm7.2.1.lw.gitff65f5bc"
)
assert np.__version__ == "1.26.4"
assert tokenizers.__version__ == "0.22.2"
assert torch.version.hip
assert torch.cuda.is_available()
print("PyTorch:", torch.__version__)
print("HIP runtime:", torch.version.hip)
print("GPU:", torch.cuda.get_device_name(0))
x = torch.randn(64, 64, device="cuda:0", requires_grad=True)
loss = (x @ x).square().mean()
loss.backward()
torch.cuda.synchronize()
assert torch.isfinite(loss)
assert torch.isfinite(x.grad).all()
print("GPU forward and backward check passed.")
PY
```

The ROCm build of PyTorch uses `torch.cuda` and `cuda:0` for AMD devices.
The HIP runtime identifies the AMD backend.

```bash
"$gpu_python" -m torch.utils.collect_env > logs/amd-environment.txt
dpkg-query -W > logs/amd-system-packages.txt
```

## 8. Run the short model and recovery checks

```bash
"$gpu_python" scripts/train_gpu.py --vendor amd \
    --config configs/amd-check.json
```

Expected: `COMPLETE: 20 updates`.

```bash
"$gpu_python" scripts/train_gpu.py --vendor amd \
    --config configs/amd-recovery-stop.json --stop-after 10
```

Expected: `STOPPED: 10 updates`.

```bash
"$gpu_python" scripts/train_gpu.py --vendor amd \
    --config configs/amd-recovery-resumed.json \
    --resume models/amd-recovery-stop/checkpoints/stopped.pt
```

Expected: resume after update 10 and finish at update 20.

```bash
"$gpu_python" scripts/inspect_checkpoint.py \
    models/amd-recovery-resumed/checkpoints/final.pt
```

## 9. Run the baseline

```bash
"$gpu_python" scripts/train_gpu.py --vendor amd \
    --config configs/amd-baseline.json
```

This starts a fresh 4,000-update run. See `docs/RECOVERY.md` for recovery.
In each new Ubuntu terminal, restore the working directory and variables:

```bash
cd "$HOME/LocalLM-Course-AMD"
gpu_python="$PWD/.venv-amd/bin/python"
export HSA_ENABLE_DXG_DETECTION=1
```

## 10. Return the complete run to Windows

Set the actual completed run name. Enter the Windows path again if needed:

```bash
run_name="amd-baseline"
read -r -p "Paste the Windows project path: " windows_project
destination="$windows_project/models/$run_name"
if [ -e "$destination" ]; then
    echo "Destination exists. Inspect it before copying."
else
    mkdir -p "$windows_project/models"
    cp -a "models/$run_name" "$destination"
fi
```

If the run resumed, also preserve and copy the earlier segments. Then copy
the environment records:

```bash
cp logs/amd-environment.txt "$windows_project/logs/"
cp logs/amd-system-packages.txt "$windows_project/logs/"
```

Use the Windows CPU environment with `docs/EVALUATE-AND-EXPORT.md`.
Copying only the checkpoint omits its matching tokenizer and run records.

## Troubleshooting

- No GPU agent: check `/dev/dxg`, the DXG variable, the driver, and the matrix.
- No PyTorch GPU: check the interpreter, exact AMD wheel, and HIP runtime.
- NumPy ABI error: the AMD environment requires NumPy 1.26.4.
- Wheel mismatch: these are `cp312` Linux wheels for Python 3.12.
- Memory failure: use a new experiment with smaller microbatches.
- Fingerprint mismatch: restore the original files and environment.
- Windows model-source mismatch: compare `model.py` with the run snapshot.

Actual AMD GPU execution is still a hardware validation step. The CPU
checks listed in `VALIDATION.md` do not establish AMD driver compatibility.

## Sources

- [ROCDXG setup, release matrix, and legacy DXG variable](https://github.com/ROCm/librocdxg)
- [ROCDXG 1.2.0](https://github.com/ROCm/librocdxg/releases/tag/v1.2.0)
- [ROCm 7.2.1 installation](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/install/quick-start.html)
- [AMD PyTorch 7.2.1 wheel guidance](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/native_linux/install-pytorch.html)
- [Microsoft WSL installation](https://learn.microsoft.com/en-us/windows/wsl/install)

[wsl-guide]: https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html
