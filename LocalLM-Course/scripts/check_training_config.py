import argparse
import json
import time
from pathlib import Path

import numpy as np
import tokenizers
import torch

from model import StoryModel
from prepare_data import ROOT, digest
from prepare_tokens import OUT, verify
from training_utils import (
    TokenBatches, learning_rate_for, load_config,
    make_optimizer, run_update,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/cpu-baseline.json"
    )
    args = parser.parse_args()
    path = Path(args.config)
    if not path.is_absolute():
        path = ROOT / path
    config = load_config(path)
    if config.run_dir.exists():
        raise FileExistsError(
            "Run folder already exists. Choose an unused run_name."
        )
    if torch.__version__ != "2.10.0+cpu":
        raise RuntimeError("Use the course CPU PyTorch installation.")
    if np.__version__ != "2.4.6":
        raise RuntimeError("Install numpy==2.4.6.")
    if tokenizers.__version__ != "0.22.2":
        raise RuntimeError("Install tokenizers==0.22.2.")
    verify()
    manifest = json.loads(
        (OUT / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["vocab_size"] != config.model.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary differ.")

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    model = StoryModel(config.model).to(
        device="cpu", dtype=torch.float32
    )
    optimizer = make_optimizer(model, config)
    batches = TokenBatches(
        "train", config.micro_batch_size,
        config.model.context_length, config.seed + 1,
    )

    state = batches.generator.get_state()
    inputs, targets = batches.next()
    expected = (
        config.micro_batch_size, config.model.context_length
    )
    if tuple(inputs.shape) != expected or targets.shape != inputs.shape:
        raise RuntimeError("Incorrect batch shapes.")
    if not torch.equal(inputs[:, 1:], targets[:, :-1]):
        raise RuntimeError("Targets are not shifted by one token.")
    batches.generator.set_state(state)
    repeated_inputs, repeated_targets = batches.next()
    if not (
        torch.equal(inputs, repeated_inputs)
        and torch.equal(targets, repeated_targets)
    ):
        raise RuntimeError("Batch generator state did not replay.")
    batches.generator.set_state(state)

    validation = TokenBatches(
        "val", config.micro_batch_size,
        config.model.context_length, config.seed + 2,
    )
    validation.next()
    if not torch.equal(state, batches.generator.get_state()):
        raise RuntimeError("Validation changed the training generator.")

    before = model.token_embedding.weight.detach().clone()
    started = time.perf_counter()
    metrics = run_update(model, optimizer, batches, config, 1)
    seconds = time.perf_counter() - started
    if torch.equal(before, model.token_embedding.weight.detach()):
        raise RuntimeError("The optimizer did not change the weights.")
    for name, parameter in model.named_parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError(f"Non-finite parameter: {name}")

    print("Config SHA256:", digest(path))
    print("Token manifest SHA256:", digest(OUT / "manifest.json"))
    print("Run folder:", config.run_dir)
    print("Device: cpu")
    print("Parameter dtype: float32")
    print("Input shape:", list(inputs.shape))
    print("Effective batch:", config.effective_batch_size)
    print("Predicted tokens per update:", config.tokens_per_update)
    budget = config.max_updates * config.tokens_per_update
    print("Training token budget:", f"{budget:,}")
    boundary = config.decay_after_update
    for update in sorted({1, boundary, boundary + 1}):
        if update <= config.max_updates:
            print(f"LR for update {update}:", learning_rate_for(
                config, update
            ))
    print("Test loss:", f"{metrics['loss']:.4f}")
    print("Gradient norm before clipping:",
          f"{metrics['grad_norm']:.4f}")
    print("Test update seconds:", f"{seconds:.2f}")
    print("Batch replay: passed")
    print("Validation generator isolation: passed")
    print("Optimizer update: passed")
    print("No model checkpoint was saved.")
    print("Training configuration check passed.")


if __name__ == "__main__":
    main()
