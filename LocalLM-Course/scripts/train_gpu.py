import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from importlib.metadata import distributions

import numpy as np
import tokenizers
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from model import StoryModel
from prepare_data import ROOT, digest
from prepare_tokens import OUT, verify
from training_utils import TokenBatches, load_config
from training_utils import make_optimizer, run_update


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def git_output(*arguments):
    try:
        result = subprocess.run(
            ["git", *arguments], cwd=ROOT, capture_output=True,
            text=True, check=True, timeout=10,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def fingerprint(vendor):
    expected = {
        "nvidia": ((3, 14), "2.10.0+cu128", "2.4.6"),
        "amd": (
            (3, 12), "2.9.1+rocm7.2.1.lw.gitff65f5bc", "1.26.4"
        ),
    }
    python, pytorch, numpy = expected[vendor]
    if sys.version_info[:2] != python:
        raise RuntimeError(f"Expected Python {python} for {vendor}.")
    if str(torch.__version__) != pytorch:
        raise RuntimeError(f"Expected torch=={pytorch}.")
    if np.__version__ != numpy or tokenizers.__version__ != "0.22.2":
        raise RuntimeError("NumPy or tokenizers version differs.")
    if vendor == "amd" and platform.system() != "Linux":
        raise RuntimeError("This AMD path requires Linux or WSL2.")
    hip = getattr(torch.version, "hip", None)
    if (vendor == "amd") != bool(hip):
        raise RuntimeError("PyTorch backend does not match the vendor.")
    if not torch.cuda.is_available():
        raise RuntimeError("No supported GPU is available to PyTorch.")
    torch.cuda.set_device(0)
    if vendor == "nvidia":
        torch.backends.cuda.matmul.fp32_precision = "ieee"
    device = torch.cuda.get_device_properties(0)
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "device": "cuda:0",
        "vendor": vendor,
        "gpu": device.name,
        "gpu_memory_bytes": device.total_memory,
        "gpu_arch": str(getattr(device, "gcnArchName", "")),
        "capability": list(torch.cuda.get_device_capability(0)),
        "cuda": torch.version.cuda,
        "hip": hip,
        "dtype": "float32",
        "attention_backend": "math",
        "token_manifest_sha256": digest(OUT / "manifest.json"),
        "tokenizer_sha256": digest(OUT / "tokenizer.json"),
        "sources": {
            name: digest(ROOT / "scripts" / name)
            for name in (
                "model.py", "training_utils.py", "train_gpu.py",
                "prepare_data.py", "prepare_tokens.py",
            )
        },
    }


class DeviceBatches(TokenBatches):
    def next(self):
        inputs, targets = super().next()
        return inputs.to("cuda:0"), targets.to("cuda:0")


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return value


@torch.no_grad()
def validation_loss(model, config):
    was_training = model.training
    model.eval()
    batches = DeviceBatches(
        "val", config.micro_batch_size,
        config.model.context_length, config.seed + 2,
    )
    total = 0.0
    try:
        for _ in range(config.eval_batches):
            inputs, targets = batches.next()
            logits, loss = model(inputs, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite validation loss.")
            total += loss.item()
            del logits, loss
    finally:
        model.train(was_training)
    return total / config.eval_batches


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True
    )
    parser.add_argument("--vendor", choices=("nvidia", "amd"),
                        required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    config_path = project_path(args.config)
    config = load_config(config_path)
    limit = (
        config.max_updates
        if args.stop_after is None else args.stop_after
    )
    if not 1 <= limit <= config.max_updates:
        raise ValueError("Stop point is outside the update budget.")
    if config.run_dir.exists():
        raise FileExistsError("Choose an unused run_name.")
    identity = fingerprint(args.vendor)
    verify()
    manifest = json.loads(
        (OUT / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest["vocab_size"] != config.model.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary differ.")

    torch.set_num_threads(config.threads)
    torch.manual_seed(config.seed)
    model = StoryModel(config.model).to(
        device="cuda:0", dtype=torch.float32
    )
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 2_557_888:
        raise RuntimeError("Unexpected model parameter count.")
    optimizer = make_optimizer(model, config)
    batches = DeviceBatches(
        "train", config.micro_batch_size,
        config.model.context_length, config.seed + 1,
    )
    completed = 0
    parent = None

    if args.resume:
        resume_path = project_path(args.resume)
        checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=True
        )
        if checkpoint.get("trainer") != "gpu-v1":
            raise RuntimeError("Resume requires a GPU checkpoint.")
        if checkpoint["format_version"] != 1:
            raise RuntimeError("Unsupported checkpoint format.")
        saved_config = dict(checkpoint["config"])
        requested_config = asdict(config)
        saved_config.pop("run_name")
        requested_config.pop("run_name")
        if saved_config != requested_config:
            raise RuntimeError("Resume settings differ from saved ones.")
        if checkpoint["fingerprint"] != identity:
            raise RuntimeError("Resume fingerprint does not match.")
        completed = checkpoint["completed_updates"]
        if type(completed) is not int or not 0 <= completed < limit:
            raise ValueError("Checkpoint is not before the stop point.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["gpu_rng_state"], 0)
        batches.generator.set_state(checkpoint["train_rng_state"])
        parent = {
            "path": str(resume_path.resolve()),
            "sha256": digest(resume_path),
            "completed_updates": completed,
        }
        del checkpoint

    run_dir = config.run_dir
    run_dir.mkdir(parents=True, exist_ok=False)
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir()
    sources = run_dir / "sources"
    sources.mkdir()
    for name in identity["sources"]:
        shutil.copy2(ROOT / "scripts" / name, sources / name)
    for name in ("tokenizer.json", "manifest.json"):
        shutil.copy2(OUT / name, run_dir / name)
    write_json(run_dir / "config.json", asdict(config))
    write_json(run_dir / "run.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "config_sha256": digest(config_path),
        "fingerprint": identity,
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_status": git_output("status", "--short"),
        "parent_checkpoint": parent,
        "installed_packages": sorted(
            f"{item.metadata['Name']}=={item.version}"
            for item in distributions()
        ),
    })
    stop_file = run_dir / "STOP"
    best_val = math.inf
    last_val = None
    last_val_update = None

    def record(event, **values):
        row = {
            "event": event,
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "completed_updates": completed,
            **values,
        }
        with (run_dir / "metrics.jsonl").open(
            "a", encoding="utf-8"
        ) as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")

    def save_checkpoint(name):
        for parameter in model.parameters():
            if not torch.isfinite(parameter).all():
                raise RuntimeError("Cannot save non-finite weights.")
        path = checkpoints / name
        temporary = path.with_suffix(".pt.tmp")
        state = {
            "format_version": 1,
            "trainer": "gpu-v1",
            "config": asdict(config),
            "fingerprint": identity,
            "completed_updates": completed,
            "model": cpu_copy(model.state_dict()),
            "optimizer": cpu_copy(optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state(),
            "gpu_rng_state": torch.cuda.get_rng_state(0),
            "train_rng_state": batches.generator.get_state(),
            "best_val": best_val,
            "last_val": last_val,
            "last_val_update": last_val_update,
        }
        with temporary.open("wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        record("checkpoint", file=name, sha256=digest(path))
        print(f"Saved {name} at update {completed}", flush=True)
        if name.startswith("step-"):
            periodic = sorted(
                checkpoints.glob("step-*.pt"),
                key=lambda p: int(p.stem.split("-")[1]),
            )
            for old in periodic[:-config.keep_checkpoints]:
                old.unlink()

    def evaluate():
        nonlocal best_val, last_val, last_val_update
        print(f"Validating at update {completed}...", flush=True)
        torch.cuda.synchronize()
        clock = time.perf_counter()
        last_val = validation_loss(model, config)
        last_val_update = completed
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        improved = last_val < best_val
        record("validation", loss=last_val, seconds=seconds,
               batches=config.eval_batches, improved=improved)
        print(f"Validation loss: {last_val:.4f}", flush=True)
        if improved:
            best_val = last_val
            save_checkpoint("best.pt")

    window_loss = 0.0
    window_seconds = 0.0
    window_count = 0
    metrics = None

    def log_progress():
        nonlocal window_loss, window_seconds, window_count
        if not window_count:
            return
        loss = window_loss / window_count
        seconds = window_seconds / window_count
        speed = config.tokens_per_update / seconds
        record("train", loss=loss, updates=window_count,
               seconds_per_update=seconds, tokens_per_second=speed,
               lr=metrics["lr"], grad_norm=metrics["grad_norm"],
               peak_allocated_bytes=torch.cuda.max_memory_allocated())
        print(
            f"Update {completed}/{config.max_updates} | "
            f"loss {loss:.4f} | lr {metrics['lr']:.6g} | "
            f"grad {metrics['grad_norm']:.4f} | "
            f"{seconds:.2f}s/update | {speed:,.0f} tokens/s",
            flush=True,
        )
        window_loss = window_seconds = 0.0
        window_count = 0

    print(f"Run folder: {run_dir}", flush=True)
    print(f"{identity['gpu']} | float32 | {parameters:,} parameters",
          flush=True)
    print(f"Threads: {config.threads}", flush=True)
    print(f"Starting after update {completed}", flush=True)
    print(f"Stop request file: {stop_file}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    record("start", parent_checkpoint=parent)
    evaluate()

    while completed < limit and not stop_file.exists():
        torch.cuda.synchronize()
        clock = time.perf_counter()
        metrics = run_update(
            model, optimizer, batches, config, completed + 1
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - clock
        completed += 1
        window_loss += metrics["loss"]
        window_seconds += seconds
        window_count += 1
        if completed % config.log_interval == 0 or completed == limit:
            log_progress()
        if (
            completed % config.eval_interval == 0
            or completed == config.max_updates
        ):
            evaluate()
        if completed % config.checkpoint_interval == 0:
            save_checkpoint(f"step-{completed:06d}.pt")

    log_progress()
    finished = completed == config.max_updates
    name = "final.pt" if finished else "stopped.pt"
    save_checkpoint(name)
    elapsed = time.perf_counter() - started
    event = "complete" if finished else "stopped"
    record(event, elapsed_seconds=elapsed, file=name)
    print(f"{event.upper()}: {completed} updates", flush=True)
    print(f"Session elapsed seconds: {elapsed:.2f}", flush=True)
    print(f"Checkpoint: {checkpoints / name}", flush=True)


if __name__ == "__main__":
    with sdpa_kernel(SDPBackend.MATH):
        main()
