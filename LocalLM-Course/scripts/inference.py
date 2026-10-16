import hashlib
import json
import math
import platform
import sys
from pathlib import Path

import numpy as np
import tokenizers
import torch
from tokenizers import Tokenizer

from model import ModelConfig, StoryModel


ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "numpy": np.__version__,
        "tokenizers": tokenizers.__version__,
        "system": platform.system(),
        "machine": platform.machine(),
        "device": "cpu",
        "dtype": "float32",
        "threads": torch.get_num_threads(),
    }


def save_report(path, result):
    path = project_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_model(checkpoint_path, tokenizer_path=None):
    if sys.version_info[:2] != (3, 14):
        raise RuntimeError("Use Python 3.14.")
    if torch.__version__ != "2.10.0+cpu":
        raise RuntimeError("Use torch==2.10.0+cpu.")
    if np.__version__ != "2.4.6" or tokenizers.__version__ != "0.22.2":
        raise RuntimeError("Restore the pinned course packages.")
    torch.set_num_threads(4)
    path = project_path(checkpoint_path)
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["format_version"] != 1:
        raise RuntimeError("Unsupported checkpoint format.")
    identity = saved["fingerprint"]
    model_path = Path(__file__).with_name("model.py")
    if digest(model_path) != identity["sources"]["model.py"]:
        raise RuntimeError("Model source differs from the saved run.")

    tokenizer_path = (
        project_path(tokenizer_path) if tokenizer_path
        else path.parent.parent / "tokenizer.json"
    )
    if digest(tokenizer_path) != identity["tokenizer_sha256"]:
        raise RuntimeError("Tokenizer differs from the saved run.")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    config = ModelConfig(**saved["config"]["model"])
    if tokenizer.get_vocab_size() != config.vocab_size:
        raise RuntimeError("Tokenizer and model vocabulary differ.")
    if tokenizer.token_to_id("<|endoftext|>") != 0:
        raise RuntimeError("Expected end-of-text ID 0.")
    model = StoryModel(config).to(device="cpu", dtype=torch.float32)
    model.load_state_dict(saved["model"], strict=True)
    for parameter in model.parameters():
        if not torch.isfinite(parameter).all():
            raise RuntimeError("Non-finite model weights.")
    model.eval()
    metadata = {
        "checkpoint_path": str(path.resolve()),
        "checkpoint_sha256": digest(path),
        "completed_updates": saved["completed_updates"],
        "training_config": saved["config"],
        "training_fingerprint": identity,
        "tokenizer_sha256": digest(tokenizer_path),
    }
    return model, tokenizer, metadata


@torch.inference_mode()
def generate(model, tokenizer, prompt, *, max_new_tokens=80,
             temperature=0.8, top_k=40, seed=42):
    if not prompt.strip() or "<|endoftext|>" in prompt:
        raise ValueError("Prompt is blank or contains the end marker.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("Temperature must be finite and positive.")
    if not 1 <= top_k <= model.config.vocab_size:
        raise ValueError("top_k must be between 1 and vocabulary size.")
    if max_new_tokens < 1 or not 0 <= seed < 2**32:
        raise ValueError("Invalid token limit or seed.")
    prompt_ids = tokenizer.encode(
        prompt, add_special_tokens=False
    ).ids
    context = model.config.context_length
    if not 1 <= len(prompt_ids) <= context:
        raise ValueError("Prompt must contain 1 to 256 tokens.")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = list(prompt_ids)
    generated = []
    reason = "max_new_tokens"
    model.eval()
    for _ in range(max_new_tokens):
        inputs = torch.tensor([ids[-context:]], dtype=torch.long)
        logits, _ = model(inputs)
        values, indices = torch.topk(logits[0, -1], top_k)
        probabilities = torch.softmax(values / temperature, dim=-1)
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Non-finite sampling probabilities.")
        choice = torch.multinomial(
            probabilities, 1, generator=generator
        ).item()
        token_id = indices[choice].item()
        generated.append(token_id)
        ids.append(token_id)
        if token_id == 0:
            reason = "end_of_text"
            break
    return {
        "prompt": prompt,
        "seed": seed,
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated),
        "generated_token_ids": generated,
        "finish_reason": reason,
        "continuation": tokenizer.decode(generated),
        "text": tokenizer.decode(ids),
    }
