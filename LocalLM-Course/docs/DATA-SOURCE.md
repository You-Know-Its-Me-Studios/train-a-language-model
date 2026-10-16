# Data source

The course uses the TinyStories V2 GPT-4 text files from the
[TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories).
The dataset is described in [TinyStories: How Small Can Language Models Be
and Still Speak Coherent English?](https://arxiv.org/abs/2305.07759), by
Ronen Eldan and Yuanzhi Li.

Pinned dataset revision:

```text
f54c09fd23315a6f9c86f9dc80f725de7d8f9c64
```

The upstream dataset card identifies the data license as
[Community Data License Agreement - Sharing, Version 1.0](https://cdla.dev/sharing-1-0/).
The data and the course source code are separate materials.

## Source acquisition

`scripts/prepare_data.py` constructs download URLs from the pinned revision.
It retrieves the first 25,000,000 bytes of `TinyStoriesV2-GPT4-train.txt`
and all 22,502,601 bytes of `TinyStoriesV2-GPT4-valid.txt`.
It checks both downloads against the source hashes embedded in the script.

The training prefix is not the complete upstream training dataset.

## Course transformations

- Split stories at the end-of-text marker.
- Discard the incomplete story at the training prefix boundary.
- Require valid UTF-8 and reject replacement characters.
- Collapse internal whitespace.
- Remove exact normalized-text duplicates across retained stories.
- Shuffle the held-out source stories with seed 42.
- Retain 5,000 validation stories and 2,000 test stories.
- Fit a 4,096-token byte-level BPE tokenizer using training stories only.
- Append one end-of-text token after each encoded story.
- Write token IDs as little-endian unsigned 16-bit integers.

The prepared and tokenized manifests record the transformations, counts,
source hashes, preparation script hashes, and execution environments.

The optional reference-data archive includes prominent modification and
attribution notices alongside these files. Keep those notices with the data.
The dataset's published terms continue to apply to that data.

## Verify rather than rebuild an active run's inputs

Use `prepare_data.py --verify` and `prepare_tokens.py --verify` to check
existing inputs. Rebuilding or replacing manifests or tokenizers changes
the training fingerprint and can prevent a checkpoint from resuming.
