# Course downloads

Build: 2026-10-16.

## Files

| File | Use |
|---|---|
| LocalLM-Course-Source.zip | Complete course scripts, configurations, requirements, guides, and templates |
| LocalLM-Course-Reference-Data.zip | Optional source data, prepared splits, tokenizer, and encoded token files |
| SHA256SUMS.txt | SHA-256 values for both ZIPs and this README |

The source archive covers the CPU course, NVIDIA training in Lesson 14,
and AMD training in Lesson 16. The quizzes remain in the Leanpub course.

## Start with the source archive

Assuming the files were downloaded to your Windows Downloads folder, open
PowerShell and run:

```powershell
$downloadFolder = Join-Path $env:USERPROFILE "Downloads"
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"

if (Test-Path $courseRoot) {
    throw "A course project already exists. Choose a separate location."
}

Expand-Archive -LiteralPath `
    (Join-Path $downloadFolder "LocalLM-Course-Source.zip") `
    -DestinationPath $env:USERPROFILE
```

Open `LocalLM-Course/README.md` and follow `docs/CPU-QUICKSTART.md`.
Adjust `$downloadFolder` if your browser saved the files elsewhere.

The scripts and named configurations are already included. Skip lesson
blocks that create those existing files. Continue with setup, data
preparation, inspection, and training commands.

## Optional reference data

For the full learning workflow, source and prepare the data yourself using
the main course. The optional archive supplies a known reference build for
comparison or for continuing after the preparation lessons.

Extract `LocalLM-Course-Reference-Data.zip` separately and read its README.
It explains how to verify and copy the complete data directory into a fresh
source project. It includes source attribution, modification notices, and
the published data license.

Do not replace inputs beneath an existing training run. Do not mix a
tokenizer or manifest from one preparation with another run's checkpoint.

## Check a downloaded ZIP

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath `
    (Join-Path $downloadFolder "LocalLM-Course-Source.zip")

Get-FileHash -Algorithm SHA256 -LiteralPath `
    (Join-Path $downloadFolder "LocalLM-Course-Reference-Data.zip")
```

Compare the values with `SHA256SUMS.txt`. The archives also contain an
individual-file manifest and a Python verification program.

## Validation status

Fresh source acquisition, preparation, tokenizer fitting, 20-update CPU
training, stop/resume, full validation and test evaluation, generation,
and inference export were executed successfully on Linux.

Windows PowerShell execution, NVIDIA hardware execution, AMD/WSL hardware
execution, and full 4,000-update model quality remain to be verified.
See `docs/VALIDATION.md` in the source archive for the exact scope.

No short test checkpoint is presented as a trained course model. Students
create their own training, selection, and export records.
