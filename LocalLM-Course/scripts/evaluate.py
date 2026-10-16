import argparse
import json
import math
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

from inference import ROOT, digest, environment, load_model
from inference import project_path, save_report


def evaluation_batches(tokens, context, batch_size):
    position = 0
    targets_left = len(tokens) - 1
    while targets_left > 0:
        if targets_left >= context:
            rows = min(batch_size, targets_left // context)
            starts = position + np.arange(rows) * context
            indices = starts[:, None] + np.arange(context + 1)
            values = tokens[indices]
            advance = rows * context
        else:
            values = tokens[position:][None, :]
            advance = targets_left
        sequences = torch.from_numpy(values.astype(np.int64))
        yield sequences[:, :-1], sequences[:, 1:]
        position += advance
        targets_left -= advance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument("--split", choices=("train", "val", "test"),
                        default="val")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive.")
    if project_path(args.output).exists():
        raise FileExistsError("Choose an unused report filename.")
    model, tokenizer, metadata = load_model(
        args.checkpoint, args.tokenizer
    )
    data_dir = ROOT / "data" / "tokenized"
    manifest_path = data_dir / "manifest.json"
    identity = metadata["training_fingerprint"]
    if digest(manifest_path) != identity["token_manifest_sha256"]:
        raise RuntimeError("Token manifest differs from the saved run.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["format"] != "uint16-le":
        raise RuntimeError("Unexpected token file format.")
    data_path = data_dir / f"{args.split}.bin"
    expected = manifest["splits"][args.split]
    if digest(data_path) != expected["sha256"]:
        raise RuntimeError("Evaluation token file changed.")
    if data_path.stat().st_size != 2 * expected["tokens"]:
        raise RuntimeError("Evaluation token file has the wrong size.")
    tokens = np.memmap(data_path, dtype="<u2", mode="r")
    if len(tokens) < 2 or int(tokens.max()) >= model.config.vocab_size:
        raise RuntimeError("Invalid evaluation tokens.")

    context = model.config.context_length
    target_count = len(tokens) - 1
    total_nll = 0.0
    count = 0
    started = time.perf_counter()
    print(f"Evaluating {args.split}: {target_count:,} targets",
          flush=True)
    with torch.inference_mode():
        for index, (inputs, targets) in enumerate(evaluation_batches(
            tokens, context, args.batch_size
        ), start=1):
            logits, _ = model(inputs)
            losses = F.cross_entropy(
                logits.reshape(-1, model.config.vocab_size),
                targets.reshape(-1), reduction="none",
            )
            if not torch.isfinite(losses).all():
                raise RuntimeError("Non-finite evaluation loss.")
            total_nll += losses.double().sum().item()
            count += targets.numel()
            del logits, losses
            if index % 100 == 0 or count == target_count:
                print(f"Scored {count:,}/{target_count:,}", flush=True)
    if count != target_count:
        raise RuntimeError("Not every target was scored exactly once.")
    mean_nll = total_nll / count
    perplexity = (
        math.exp(mean_nll)
        if mean_nll < math.log(sys.float_info.max) else None
    )
    result = {
        **metadata,
        "split": args.split,
        "data_sha256": expected["sha256"],
        "protocol": "disjoint-input-windows-v1",
        "context_length": context,
        "batch_size": args.batch_size,
        "evaluated_targets": count,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity is None,
        "evaluation_seconds": time.perf_counter() - started,
        "environment": environment(),
        "source_sha256": {
            name: digest(ROOT / "scripts" / name)
            for name in ("evaluate.py", "inference.py", "model.py")
        },
    }
    save_report(args.output, result)
    print("Mean NLL:", f"{mean_nll:.6f}")
    print("Perplexity:", perplexity)
    print("Report:", project_path(args.output))


if __name__ == "__main__":
    main()
