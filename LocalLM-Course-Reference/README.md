# Optional course reference data

This archive contains the data and tokenizer produced by the course
preparation scripts. It is a companion to `LocalLM-Course-Source.zip`.

Use the main course's preparation steps to learn the complete workflow.
Use this reference archive to compare results or to start after those steps.
No trained model is included.

## Contents

| Directory | Contents |
|---|---|
| data/raw | Verified training prefix and full upstream validation file |
| data/prepared | Normalized story splits and preparation manifest |
| data/tokenized | BPE tokenizer, encoded splits, and token manifest |

The training source is the first 25,000,000 bytes of the pinned upstream
training file. It is not the complete TinyStories training dataset.

| Split | Stories | Tokens |
|---|---:|---:|
| Training | 30,496 | 6,080,666 |
| Validation | 5,000 | 977,065 |
| Test | 2,000 | 398,416 |

The vocabulary contains 4,096 tokens. End-of-text is ID 0. Token files use
little-endian unsigned 16-bit integers. Each story has one end token.

## Install into a fresh source project

Extract this ZIP into a separate location. It creates a folder named
`LocalLM-Course-Reference`.

Do not replace the inputs of an existing training run. Its checkpoints
are tied to the original tokenizer and manifest hashes.

In Windows PowerShell, set the actual extracted reference location:

```powershell
$courseRoot = Join-Path $env:USERPROFILE "LocalLM-Course"
$referenceRoot = Join-Path $env:USERPROFILE `
    "Downloads\LocalLM-Course-Reference"
$coursePython = Join-Path $courseRoot `
    ".venv-cpu\Scripts\python.exe"

foreach ($name in @("raw", "prepared", "tokenized")) {
    if (Test-Path (Join-Path $courseRoot "data\$name")) {
        throw "Existing data found. Use a fresh project."
    }
}

& $coursePython (Join-Path $referenceRoot "verify_download.py")
if ($LASTEXITCODE -ne 0) { throw "Reference verification failed." }

New-Item -ItemType Directory -Force `
    (Join-Path $courseRoot "data") | Out-Null

Get-ChildItem -LiteralPath (Join-Path $referenceRoot "data") |
    Copy-Item -Destination (Join-Path $courseRoot "data") -Recurse

Set-Location -LiteralPath $courseRoot -ErrorAction Stop
& $coursePython scripts\prepare_data.py --verify
if ($LASTEXITCODE -ne 0) { throw "Prepared data check failed." }
& $coursePython scripts\prepare_tokens.py --verify
if ($LASTEXITCODE -ne 0) { throw "Token data check failed." }
```

After installation, use `--verify` instead of rerunning preparation into
these existing folders. The course scripts deliberately refuse to overwrite
prepared inputs.

For AMD, copy the verified Windows source project and its complete `data`
directory into WSL as described in the AMD quickstart.

## Provenance and changes

Read `data/NOTICE.txt`, the upstream dataset card, and the data license.
Each source or transformed data file has a companion modification notice.
Keep these with the data.

The manifests are copied unchanged from the reference build. They record
the actual Python versions and preparation source hashes. A preparation on
another machine can have different manifest metadata. Do not mix a manifest
from one preparation with data or tokenizers from another.

The source download supplies the scripts whose hashes appear in these
manifests. The reference data does not replace the model source or runtime
requirements.

`DOWNLOAD-MANIFEST.json` records the files in this archive. Run the included
`verify_download.py` with Python 3.12 or 3.14 to check them.
