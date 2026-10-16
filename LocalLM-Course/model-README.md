# Local Story Model

This package runs the story-continuation model built in the course.
It uses a selected training checkpoint and its original tokenizer.

The model has 2,557,888 parameters, four transformer blocks, a width
of 192, six attention heads, a 4,096-token BPE vocabulary, and a
256-token context window. It uses RMSNorm, rotary positions,
query/key normalization, SwiGLU, and tied input/output embeddings.

The training text came from the TinyStoriesV2 GPT-4 files at dataset
revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
The course prepared 30,496 training stories, 5,000 validation
stories, and 2,000 test stories.

See the packaged records for the actual update count, training
settings, environment, evaluation results, and selection reason.
The full package list from the export environment is recorded in
`records/inference-environment.txt`.

## Install on Windows

Install the Python Install Manager as described in the course.
Open PowerShell in this extracted package folder, then run:

    pymanager install 3.14
    if ($LASTEXITCODE -ne 0) {
        throw "Python installation failed."
    }

    pymanager exec -V:3.14 -m venv .venv-cpu
    if ($LASTEXITCODE -ne 0) {
        throw "Environment creation failed."
    }

    $python = Join-Path $PWD.Path ".venv-cpu\Scripts\python.exe"
    foreach ($name in @("cpu", "tokenizer", "model")) {
        & $python -m pip install --only-binary=:all: `
            -r "requirements-$name.txt"
        if ($LASTEXITCODE -ne 0) {
            throw "Package installation failed: $name"
        }
    }

    & $python -m pip check
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency verification failed."
    }

## Generate text

    & $python .\scripts\generate_text.py `
        --prompt "A little dog found a red ball near the gate."

For an ongoing console session:

    & $python .\scripts\generate_text.py --interactive

Enter a story beginning at the prompt. Type `/quit` to exit.
Every prompt starts a separate continuation. There is no chat history.
Generation reports are saved under `outputs`.

Defaults are temperature 0.8, top-k 40, seed 42, and 80 new tokens.
Use `--help` to see the available arguments.

## Files and records

- `checkpoints/model.pt`: the unchanged selected checkpoint.
- `tokenizer.json`: the tokenizer from that training run.
- `scripts`: model loading and generation code.
- `requirements-*.txt`: direct package version requirements.
- `records`: selection, evaluation, and training provenance.
- `manifest.json`: SHA-256 hashes of the packaged files.

The checkpoint retains optimizer and recovery state. This package
does not include the dataset or a complete training workspace.
Retain the original project backup to reproduce or resume training.

Paths in historical records refer to the original training project.
The generation program uses the files in this extracted package.

## Limits

This is a small story-continuation model. It can repeat text, lose
track of characters, contradict itself, or stop without an ending.
It was not instruction-tuned and has no factual accuracy guarantee.

Generation stops at end-of-text or the requested token limit.
After the input window fills, only the most recent 256 tokens are
used. Earlier text is no longer available to the next prediction.

The program uses CPU float32 execution with four threads.
Generation uses local files after dependencies are installed.
No hosted model service or API key is required.

The initial software installation requires internet access.
Exact output can vary across execution environments.
This package does not provide an Ollama or GGUF conversion.

## Data source

See the [TinyStories dataset][data]
for the data description and published terms. Raw dataset files are
not included in this package.

[data]: https://huggingface.co/datasets/roneneldan/TinyStories
